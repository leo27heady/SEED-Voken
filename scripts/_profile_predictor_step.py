"""Rough per-component timing for predictor training step."""
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    cfg_path = ROOT / "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite_predict.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    args = cfg["model"]["init_args"]
    B = cfg["data"]["init_args"]["batch_size"]

    model = VideoHierPredictorModel(**args).cuda().train()
    x = torch.randn(B, 3, 9, 32, 32, device="cuda")

    print(f"batch_size={B}")
    print(f"envelope events={len(list(model.schedule.envelope_order()))}")
    for s, spec in enumerate(model.schedule.stages):
        T = model.schedule.native_t(s)
        ps = model.predictor_stages[s]
        n_layers = len(ps.rev_layers) if ps.rev_layers else len(ps.layers)
        print(
            f"  stage {s}: T={T} HxW={spec.H}x{spec.W} "
            f"shifts={model.schedule.shifts_per_stage(s)} "
            f"tokens={T * spec.H * spec.W} dim={ps.dim} attn={ps.attention_type} layers={n_layers}"
        )

    for _ in range(2):
        with torch.autocast("cuda"):
            model.forward_batch(x, inference_mode="train")
    sync()

    sync()
    t0 = time.perf_counter()
    with torch.autocast("cuda"):
        loss, out = model.forward_batch(x, inference_mode="train")
    sync()
    full = time.perf_counter() - t0

    sync()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda"):
        batch = model.batch_prep.encode_and_schedule(model.vae, x, model.predictor_stages)
    sync()
    prep = time.perf_counter() - t0

    sync()
    t0 = time.perf_counter()
    with torch.autocast("cuda"):
        out2 = model.orchestrator.forward_train(batch)
    sync()
    orc = time.perf_counter() - t0

    sync()
    t0 = time.perf_counter()
    with torch.autocast("cuda"):
        mse = model._decode_horizon_mse(batch, out2.pred_indices)
    sync()
    dec = time.perf_counter() - t0

    print(f"\nTiming (1 train step, after warmup):")
    print(f"  full forward_batch: {full:.2f}s")
    print(f"  VAE encode+prep:    {prep:.2f}s ({100*prep/full:.0f}%)")
    print(f"  orchestrator:       {orc:.2f}s ({100*orc/full:.0f}%)")
    print(f"  decode pred_mse:    {dec:.2f}s ({100*dec/full:.0f}%)")

    # Per envelope event
    print("\nPer envelope event (forward only):")
    parent_by_stage = {}
    stream_carry = {}
    total_ev = 0.0
    for s, k in model.schedule.envelope_order():
        parent = None
        if s > 0:
            ps, pk = model.schedule.parent_for(s, k)
            parent = parent_by_stage.get(ps)
        n_sp = model.schedule.stages[s].n_spatial
        ctx_end = model.schedule._context_end[s]
        t_len = ctx_end + 1
        masks = batch.masks[s][k]
        stage = model.predictor_stages[s]
        if k == 0:
            ctx = batch.context_embed[s]
            stream = None
        else:
            ctx = None
            stream = stream_carry.get((s, k - 1))
        sync()
        t0 = time.perf_counter()
        with torch.autocast("cuda"):
            out = stage.execute_shift(
                k,
                context_embed=ctx,
                stream_state=stream,
                parent=parent,
                masks=masks,
                t_len=t_len,
            )
        sync()
        dt = time.perf_counter() - t0
        total_ev += dt
        parent_by_stage[s] = model.orchestrator._parent_from_output(out, s)
        stream_carry[(s, k)] = (out.o1, out.o2)
        print(f"  (s={s}, k={k}) tokens={t_len*n_sp:4d}  {dt:.2f}s")
    print(f"  sum envelope events: {total_ev:.2f}s")


if __name__ == "__main__":
    main()
