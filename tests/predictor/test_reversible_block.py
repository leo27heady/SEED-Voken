"""Smoke tests for reversible block and dual-stream routing."""

import torch

from src.Open_MAGVIT2.modules.predictor.reversible_block import ReversibleCouplingBlock


def test_rev_01_forward_invertibility():
    block = ReversibleCouplingBlock(dim=16, n_heads=4, has_cross_attn=False)
    x = torch.randn(2, 32, 16 * 2)
    y = block(x)
    x_hat = block.inverse(y)
    assert torch.allclose(x, x_hat, atol=1e-5)


def test_ds_routing_parent_streams():
    block = ReversibleCouplingBlock(dim=8, n_heads=2, has_cross_attn=True, parent_dim=8)
    parent = torch.randn(1, 10, 8)
    block.set_parent(parent_mode="dual_stream", o1=parent, o2=parent + 1, fused=None, layer_idx=0)
    assert block._cross_kv is parent
    block.set_parent(parent_mode="dual_stream", o1=parent, o2=parent + 1, fused=None, layer_idx=1)
    assert torch.allclose(block._cross_kv, parent + 1)


def test_ds_03_fused_hard_single_kv():
    block = ReversibleCouplingBlock(dim=8, n_heads=2, has_cross_attn=True, parent_dim=8)
    fused = torch.randn(1, 10, 8)
    for layer_idx in range(4):
        block.set_parent(
            parent_mode="fused_hard", o1=None, o2=None, fused=fused, layer_idx=layer_idx
        )
        assert block._cross_kv is fused


def test_rev_02_gradcheck_coupling():
    block = ReversibleCouplingBlock(dim=8, n_heads=2, has_cross_attn=False, custom_backward=False)
    block = block.double()
    x = torch.randn(1, 4, 16, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(block, x, eps=1e-6, atol=1e-4)


def test_ds_04_parent_dim_projection():
    block = ReversibleCouplingBlock(dim=8, n_heads=2, has_cross_attn=True, parent_dim=16)
    parent = torch.randn(1, 10, 16)
    block.set_parent(parent_mode="dual_stream", o1=parent, o2=parent, fused=None, layer_idx=0)
    x = torch.randn(1, 10, 8)
    y = block.F(x, cross_kv=parent, cross_attn_mask=None)
    assert y.shape == x.shape
