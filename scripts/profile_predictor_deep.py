"""Deep predictor step profiler: forward, backward, per-event, mask build."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel
from src.Open_MAGVIT2.modules.predictor.attention.factorized import build_spatial_window_mask
from src.Open_MAGVIT2.modules.predictor.masks import PyramidMaskBuilder


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
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args["learning_rate"]
    )
    x = torch.randn(B, 3, 9, 32, 32, device="cuda", requires_grad=False)

    print("=== Config ===")
    print(f"batch_size={B}, envelope_events={len(list(model.schedule.envelope_order()))}")
    for s, spec in enumerate(model.schedule.stages):
        T = model.schedule.native_t(s)
        ps = model.predictor_stages[s]
        nl = len(ps.rev_layers) if ps.rev_layers else len(ps.layers)
        print(
            f"  s{s}: T={T} grid={spec.H}x{spec.W} tokens={T*spec.n_spatial} "
            f"dim={ps.dim} attn={ps.attention_type} layers={nl} "
            f"shifts={model.schedule.shifts_per_stage(s)} K={spec.codebook_size}"
        )

    # warmup
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            loss, _, _ = model.forward_batch(x, inference_mode="train")
        loss.backward()
        opt.step()
    sync()

    # full train step
    opt.zero_grad(set_to_none=True)
    sync()
    t0 = time.perf_counter()
    with torch.autocast("cuda"):
        loss, out, _ = model.forward_batch(x, inference_mode="train")
    sync()
    fwd = time.perf_counter() - t0
    sync()
    t0 = time.perf_counter()
    loss.backward()
    sync()
    bwd = time.perf_counter() - t0
    opt.step()
    train_step = fwd + bwd

    print("\n=== Train step (warm) ===")
    print(f"  forward:  {fwd:.3f}s")
    print(f"  backward: {bwd:.3f}s")
    print(f"  total:    {train_step:.3f}s")

    with torch.no_grad(), torch.autocast("cuda"):
        batch = model.batch_prep.encode_and_schedule(model.vae, x, model.predictor_stages)

    sync()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda"):
        model.batch_prep.encode_and_schedule(model.vae, x, model.predictor_stages)
    sync()
    vae_prep = time.perf_counter() - t0

    sync()
    t0 = time.perf_counter()
    with torch.autocast("cuda"):
        model.orchestrator.forward_train(batch)
    sync()
    orc_fwd = time.perf_counter() - t0

    print(f"  vae+prep: {vae_prep:.3f}s")
    print(f"  orch fwd only: {orc_fwd:.3f}s")

    # mask build (full pyramid)
    mb = PyramidMaskBuilder(model.schedule)
    device = torch.device("cuda")
    sync()
    t0 = time.perf_counter()
    for _ in range(10):
        mb.build_all_masks(model.orchestrator.temporal_windows, device)
    sync()
    mask_build = (time.perf_counter() - t0) / 10

    # spatial window mask (Python loops) — per call as in factorized forward
    sync()
    t0 = time.perf_counter()
    for _ in range(100):
        build_spatial_window_mask(256, 4, device)
    sync()
    spat_mask_256 = (time.perf_counter() - t0) / 100

    sync()
    t0 = time.perf_counter()
    for _ in range(100):
        build_spatial_window_mask(64, 4, device)
    sync()
    spat_mask_64 = (time.perf_counter() - t0) / 100

    print("\n=== Mask overhead ===")
    print(f"  build_all_masks (7 events): {mask_build*1000:.1f}ms")
    print(f"  build_spatial_window_mask 256 sw=4: {spat_mask_256*1000:.2f}ms")
    print(f"  build_spatial_window_mask 64 sw=4: {spat_mask_64*1000:.2f}ms")
    fine_calls = 4 * 2 * 2 * 2  # shifts * layers * block_f/g * temporal+spatial... spatial only /2
    fine_spatial_calls = 4 * 2 * 2  # shifts * layers * block_f+block_g
    print(f"  est. fine spatial mask rebuilds/step: {fine_spatial_calls} -> {fine_spatial_calls*spat_mask_256*1000:.0f}ms")

    # per envelope event (forward)
    print("\n=== Per envelope event (forward) ===")
    parent_by_stage = {}
    stream_carry = {}
    ev_total = 0.0
    for s, k in model.schedule.envelope_order():
        parent = None
        if s > 0:
            ps, pk = model.schedule.parent_for(s, k)
            parent = parent_by_stage.get(ps)
        n_sp = model.schedule.stages[s].n_spatial
        ctx_end = model.schedule._context_end[s]
        t_len = ctx_end + 1
        masks = batch.masks[(s, k)]
        stage = model.predictor_stages[s]
        ctx = batch.context_embed[s] if k == 0 else None
        stream = None if k == 0 else stream_carry.get((s, k - 1))
        sync()
        t0 = time.perf_counter()
        with torch.autocast("cuda"):
            o = stage.execute_shift(
                k,
                context_embed=ctx,
                stream_state=stream,
                parent=parent,
                masks=masks,
                t_len=t_len,
            )
        sync()
        dt = time.perf_counter() - t0
        ev_total += dt
        parent_by_stage[s] = model.orchestrator._parent_from_output(o, s)
        if k == 0 or stream is None:
            stream_carry[(s, k)] = (o.o1, o.o2)
        else:
            stream_carry[(s, k)] = (o.o1, o.o2)
        print(f"  (s={s},k={k}) tokens={t_len*n_sp:4d} dim={stage.dim:3d} {dt:.3f}s")
    print(f"  sum events: {ev_total:.3f}s")

    # SDPA vs MHA microbench on fine geometry
    print("\n=== Attention microbench (fine: B*T=72, S=256, D=16, H=4) ===")
    bt, seq, dim, heads = 72, 256, 16, 4
    q = torch.randn(bt, seq, dim, device="cuda", dtype=torch.float16)
    bool_mask = torch.zeros(seq, seq, dtype=torch.bool, device="cuda")
    bool_mask = ~build_spatial_window_mask(256, 4, q.device)

    from torch.nn import MultiheadAttention as MHA
    mha = MHA(dim, heads, batch_first=True).cuda().half()

    sync()
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.autocast("cuda"):
            mha(q, q, q, attn_mask=bool_mask)
    sync()
    mha_masked = (time.perf_counter() - t0) / 20

    sync()
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.autocast("cuda"):
            mha(q, q, q)
    sync()
    mha_full = (time.perf_counter() - t0) / 20

    sync()
    t0 = time.perf_counter()
    for _ in range(20):
        with torch.autocast("cuda"):
            torch.nn.functional.scaled_dot_product_attention(
                q.view(bt, seq, heads, dim // heads).transpose(1, 2),
                q.view(bt, seq, heads, dim // heads).transpose(1, 2),
                q.view(bt, seq, heads, dim // heads).transpose(1, 2),
            )
    sync()
    sdpa = (time.perf_counter() - t0) / 20

    print(f"  MHA + bool mask: {mha_masked*1000:.1f}ms")
    print(f"  MHA no mask:     {mha_full*1000:.1f}ms")
    print(f"  SDPA (flash):    {sdpa*1000:.1f}ms")

    # xformers if available
    try:
        import xformers.ops as xops
        sync()
        t0 = time.perf_counter()
        for _ in range(20):
            with torch.autocast("cuda"):
                xops.memory_efficient_attention(
                    q.view(bt, seq, heads, dim // heads),
                    q.view(bt, seq, heads, dim // heads),
                    q.view(bt, seq, heads, dim // heads),
                )
        sync()
        xf = (time.perf_counter() - t0) / 20
        print(f"  xformers MEA:    {xf*1000:.1f}ms")
    except Exception as e:
        print(f"  xformers: unavailable ({e})")

    # torch.compile smoke
    try:
        stage2 = model.predictor_stages[2]
        compiled = torch.compile(stage2, mode="reduce-overhead")
        ctx = batch.context_embed[2]
        masks = batch.masks[(2, 0)]
        parent = parent_by_stage.get(1)
        for _ in range(2):
            with torch.autocast("cuda"):
                compiled.execute_shift(0, context_embed=ctx, stream_state=None,
                    parent=parent, masks=masks, t_len=3)
        sync()
        t0 = time.perf_counter()
        for _ in range(10):
            with torch.autocast("cuda"):
                compiled.execute_shift(0, context_embed=ctx, stream_state=None,
                    parent=parent, masks=masks, t_len=3)
        sync()
        print(f"\n=== torch.compile fine shift0 (10x): {(time.perf_counter()-t0)/10:.3f}s "
              f"(baseline event ~{ev_total/7:.3f}s total envelope)")
    except Exception as e:
        print(f"\n=== torch.compile: failed ({e})")


if __name__ == "__main__":
    main()
