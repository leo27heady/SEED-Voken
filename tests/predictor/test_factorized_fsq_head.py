"""Factorized per-channel FSQ prediction head (Option A).

Verifies the head is correct (dim, basis matches the tokenizer, per-channel CE/
commit) and that a predictor with output_mode='factorized_fsq' builds, forwards,
and backprops on an FSQ tokenizer. Composite-mode equivalence is covered by
test_golden_refactor.py (must stay bit-equal)."""
import math

import torch

from src.Open_MAGVIT2.modules.predictor.stage import PredictorStage
from src.Open_MAGVIT2.modules.vqvae.hierarchical.fsq import FSQLayerQuantizer


def _stage(levels, dim=32):
    K = 1
    for L in levels:
        K *= L
    return PredictorStage(
        dim=dim, n_heads=4, n_layers=1, codebook_size=K, codebook_dim=4,
        h=4, w=4, max_t=9, max_shifts=4,
        output_mode="factorized_fsq", levels=levels,
    )


def test_head_dim_and_basis_match_fsq():
    levels = [8, 4, 4, 4]
    st = _stage(levels)
    assert st.head_out_dim == sum(levels) == 20
    assert st.output_head[-1].out_features == 20
    # basis must match the tokenizer's FSQ basis exactly (else decode is wrong)
    q = FSQLayerQuantizer(levels=levels)
    assert st._fsq_basis.tolist() == q._basis.tolist() == [1, 8, 32, 128]


def test_uniform_factorized_ce_equals_lnK():
    levels = [8, 4, 4, 4]
    K = 512
    st = _stage(levels)
    logits = torch.zeros(2, 16, sum(levels))  # uniform per channel
    tgt = torch.randint(0, K, (2, 16))
    ce = st.shift_cross_entropy(logits, tgt)
    # sum of per-channel uniform CEs == sum(ln L_i) == ln(prod L_i) == ln K
    assert math.isclose(ce.item(), math.log(K), rel_tol=1e-5)


def test_commit_composes_per_channel_argmax():
    levels = [8, 4, 4, 4]
    basis = [1, 8, 32, 128]
    st = _stage(levels)
    codes = [3, 1, 2, 0]
    logits = torch.full((1, 5, sum(levels)), -10.0)
    off = 0
    for i, L in enumerate(levels):
        logits[:, :, off + codes[i]] = 10.0
        off += L
    idx = st.commit_index(logits)
    expected = sum(c * b for c, b in zip(codes, basis))
    assert (idx == expected).all()
    # and decoding that index with the tokenizer basis recovers the codes
    q = FSQLayerQuantizer(levels=levels)
    dec = [(int(idx[0, 0]) // int(q._basis[i])) % L for i, L in enumerate(levels)]
    assert dec == codes


def test_per_channel_accuracy_perfect_and_zero():
    levels = [4, 4, 4, 4]
    st = _stage(levels)
    K = 256
    tgt = torch.randint(0, K, (2, 8))
    # build logits that put all mass on the TRUE per-channel code -> acc 1.0
    logits = torch.full((2, 8, sum(levels)), -10.0)
    flat_tgt = tgt.reshape(-1)
    off = 0
    for i, L in enumerate(levels):
        code_i = (flat_tgt // int(st._fsq_basis[i])) % L
        rows = torch.arange(flat_tgt.numel())
        block = torch.full((flat_tgt.numel(), L), -10.0)
        block[rows, code_i] = 10.0
        logits.reshape(-1, sum(levels))[:, off:off + L] = block
        off += L
    acc = st.mean_per_channel_accuracy(logits, tgt)
    assert math.isclose(acc.item(), 1.0, abs_tol=1e-6)


def test_factorized_model_builds_forwards_backprops():
    from src.Open_MAGVIT2.models.video_hier_predictor import VideoHierPredictorModel

    model = VideoHierPredictorModel(
        vae_config="configs/Open-MAGVIT2/gpu/shapes3d_fsq_32_S_lite_pyr_smoke.yaml",
        t_context=5, t_total=9,
        predictor={
            "dim": [64, 32, 32], "n_layers": [1, 1, 1], "n_heads": [4, 4, 4],
            "temporal_windows": [-1, 3, 1], "output_mode": "factorized_fsq",
            "attention": {"per_stage": [
                {"type": "full"},
                {"type": "factorized", "t_window": 3},
                {"type": "factorized", "t_window": 1, "spatial_window": 4},
            ]},
        },
        loss={"lambda_pred_mse": 0.0},
    )
    # smoke FSQ levels are [4,4,4] per stage -> head_out 12, K 64
    for s in range(3):
        st = model.predictor_stages[s]
        assert st.output_mode == "factorized_fsq"
        assert st.head_out_dim == sum(st.levels) == 12
        # stage basis must match the tokenizer's quantizer basis
        q = model.vae.hier_quant.blocks[s].quantizer
        assert st._fsq_basis.tolist() == q._basis.tolist()

    video = torch.randn(2, 3, 9, 32, 32)
    total, out, _ = model.forward_batch(video, inference_mode="train")
    assert torch.isfinite(out.loss_ce), "factorized CE is not finite"
    total.backward()
    g = model.predictor_stages[0].output_head[-1].weight.grad
    assert g is not None and torch.isfinite(g).all(), "no finite grad on factorized head"
    # AR path must also run (commit composes per-channel -> composite index)
    with torch.no_grad():
        _, out_ar, _ = model.forward_batch(video, inference_mode="autoregressive")
    assert torch.isfinite(out_ar.loss_ce)
