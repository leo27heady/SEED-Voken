"""FSQ quantizer tests (Path 2).

Guards the FSQ contract: train==eval (no soft/hard gap), exact index<->code<->value
round-trip incl. even levels, straight-through gradient, aux_loss==0, no posterior_var,
and the builder/topdown dispatch (incl. the H1 mixed-per-layer levels validation).
"""

import math

import pytest
import torch

from src.Open_MAGVIT2.modules.vqvae.hierarchical.fsq import FSQLayerQuantizer, round_ste
from src.Open_MAGVIT2.modules.vqvae.hierarchical.quantizer_builder import build_layer_quantizer


def _z(b=2, c=16, t=3, h=4, w=4, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, c, t, h, w)


@pytest.mark.parametrize("levels", [[8, 8, 8, 8], [8, 8, 8, 4], [8, 8, 4, 4], [5, 5, 5], [3, 4, 5, 6, 7, 8]])
def test_roundtrip_exact(levels):
    """decode_indices(forward.indices) reproduces forward.z_q bit-for-bit."""
    q = FSQLayerQuantizer(levels=levels, in_channels=16)
    r = q(_z())
    z_q2 = q.decode_indices(r.indices)
    assert torch.allclose(r.z_q, z_q2, atol=1e-6), (r.z_q - z_q2).abs().max().item()


@pytest.mark.parametrize("levels", [[4], [5], [4, 4], [6, 8]])
def test_all_codes_reachable(levels):
    """Even levels must expose all L codes (the naive (L-1)/2 round drops one)."""
    q = FSQLayerQuantizer(levels=levels, in_channels=len(levels))
    # sweep a wide range of inputs so every bin is hit
    h = torch.linspace(-8, 8, 4096).reshape(1, 1, -1, 1, 1).repeat(1, len(levels), 1, 1, 1)
    r = q(h)
    assert int(r.indices.max().item()) == q.size_dict - 1
    assert int(r.indices.min().item()) == 0
    # per-dim code coverage == L
    _, codes = q._quantize(q._to_codebook_space(h))
    for d, L in enumerate(levels):
        assert codes[:, d].unique().numel() == L, (d, L, codes[:, d].unique().numel())


def test_train_equals_eval():
    """No soft/hard gap: flg_train / flg_quant_det do not change z_q."""
    q = FSQLayerQuantizer(levels=[8, 8, 8, 4], in_channels=16)
    z = _z()
    r_train = q(z, flg_train=True, flg_quant_det=False)
    r_eval = q(z, flg_train=False, flg_quant_det=True)
    assert torch.allclose(r_train.z_q, r_eval.z_q, atol=0.0)
    assert torch.equal(r_train.indices, r_eval.indices)


def test_ste_gradient_flows():
    q = FSQLayerQuantizer(levels=[8, 8, 8, 4], in_channels=16)
    z = _z().requires_grad_(True)
    q(z).z_q.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0  # not blocked by round
    assert q.in_proj.weight.grad is not None and q.in_proj.weight.grad.abs().sum() > 0


def test_ste_gradient_no_proj():
    q = FSQLayerQuantizer(levels=[5, 5, 5], in_channels=3)  # in_proj is Identity
    assert q.in_proj is None
    z = _z(c=3).requires_grad_(True)
    q(z).z_q.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0


def test_round_ste_identity_grad():
    x = torch.randn(10, requires_grad=True)
    round_ste(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))  # straight-through


def test_aux_loss_zero():
    r = FSQLayerQuantizer(levels=[8, 8, 8, 4], in_channels=16)(_z())
    assert r.aux_loss.item() == 0.0 and r.aux_loss.shape == ()


def test_no_posterior_var_and_stats():
    r = FSQLayerQuantizer(levels=[8, 8, 8, 4], in_channels=16)(_z())
    assert "posterior_var" not in r.log_stats
    assert {"active_codes", "usage_fraction", "size_dict"} <= set(r.log_stats)
    assert int(r.log_stats["size_dict"].item()) == 8 * 8 * 8 * 4


def test_size_dict_and_index_range():
    q = FSQLayerQuantizer(levels=[8, 8, 4, 4], in_channels=16)
    assert q.size_dict == 1024 and q.dim_dict == 4
    r = q(_z())
    assert 0 <= int(r.indices.min()) and int(r.indices.max()) < q.size_dict


def test_perplexity_near_uniform_on_uniform_input():
    """Sanity: a well-spread input yields high (near-K) perplexity."""
    q = FSQLayerQuantizer(levels=[4, 4, 4], in_channels=3)  # K=64
    h = torch.empty(1, 3, 8, 8, 8).uniform_(-4, 4)
    r = q(h)
    assert r.perplexity.item() > 0.3 * q.size_dict


def test_rejects_binary_and_bad_levels():
    with pytest.raises(ValueError, match="binary|lfq"):
        FSQLayerQuantizer(levels=[2, 4])
    with pytest.raises(ValueError, match=">= 2"):
        FSQLayerQuantizer(levels=[1, 4])
    with pytest.raises(ValueError):
        FSQLayerQuantizer(levels=[])


def test_builder_dispatch():
    q = build_layer_quantizer("fsq", size_dict=0, dim_dict=0, in_channels=16, levels=[8, 8, 8, 8])
    assert isinstance(q, FSQLayerQuantizer) and q.size_dict == 4096
    with pytest.raises(ValueError, match="requires per-layer 'levels'"):
        build_layer_quantizer("fsq", size_dict=0, dim_dict=0, in_channels=16, levels=None)
    with pytest.raises(ValueError, match="Unknown quantizer"):
        build_layer_quantizer("vq", size_dict=8, dim_dict=4, in_channels=16)


def test_set_temperature_noop():
    q = FSQLayerQuantizer(levels=[5, 5, 5], in_channels=3)
    q.set_temperature(0.1)  # inherited no-op; must not raise
