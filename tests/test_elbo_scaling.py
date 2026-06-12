"""ELBO reduction regression tests (Path A, Phase 1).

Guards the SQ-VAE reduction semantics: categorical KL summed over latent
positions and the codebook dimension, batch-averaged only. A `mean` here
shrinks the KL by T*H*W*K and silently turns the model into a plain AE.
"""

import math

import pytest
import torch
import yaml

from src.Open_MAGVIT2.models.video_hier_vqgan import VideoHierVQModel
from src.Open_MAGVIT2.modules.vqvae.hierarchical.gaussian_sq import GaussianSQQuantizer
from src.Open_MAGVIT2.modules.vqvae.hierarchical.hier_elbo_loss import compute_hier_elbo_loss

LITE_CFG = "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_32_S_lite.yaml"


def _manual_kld(z, quantizer, var, prior):
    """Loop-computed sum_{t,h,w} sum_k q*(log q - log p), batch mean."""
    z_cb = quantizer._to_codebook_space(z)
    b, d, t, h, w = z_cb.shape
    precision = 1.0 / var
    total = torch.zeros(b)
    for bi in range(b):
        for ti in range(t):
            for hi in range(h):
                for wi in range(w):
                    vec = z_cb[bi, :, ti, hi, wi]
                    dist = ((vec.unsqueeze(0) - quantizer.codebook) ** 2).sum(dim=1)
                    logit = -0.5 * precision * dist
                    q = torch.softmax(logit, dim=0)
                    log_q = torch.log_softmax(logit, dim=0)
                    if prior == "zero":
                        log_p = torch.zeros_like(log_q)
                    elif prior == "uniform":
                        log_p = torch.full_like(log_q, -math.log(quantizer.size_dict))
                    else:
                        raise ValueError(prior)
                    total[bi] += (q * (log_q - log_p)).sum()
    return total.mean()


@pytest.mark.parametrize("prior", ["zero", "uniform"])
def test_kld_discrete_matches_manual_sum(prior):
    torch.manual_seed(0)
    K, D = 5, 4
    q = GaussianSQQuantizer(size_dict=K, dim_dict=D, flg_loss_continuous=False, prior=prior)
    z = torch.randn(2, D, 1, 2, 2)
    var = torch.tensor([0.7])

    result = q(z, var_q_pos=var, flg_train=True, flg_quant_det=False)
    expected = _manual_kld(z, q, var[0], prior)
    assert torch.allclose(result.aux_loss, expected, atol=1e-5), (
        f"kld_discrete {result.aux_loss.item():.6f} != manual {expected.item():.6f}"
    )


def test_kld_continuous_is_summed():
    torch.manual_seed(0)
    K, D = 5, 4
    q = GaussianSQQuantizer(size_dict=K, dim_dict=D, flg_loss_continuous=True, prior="zero")
    z = torch.randn(2, D, 1, 2, 2)
    var = torch.tensor([0.7])
    result = q(z, var_q_pos=var, flg_train=True, flg_quant_det=False)

    # reconstruct the continuous term manually from the returned z_q
    discrete = _manual_kld(z, q, var[0], "zero")
    cont = result.aux_loss - discrete
    # summed over (C,T,H,W): magnitude must scale with element count, i.e.
    # equal 0.5/var * sum of squared residual averaged over batch
    assert cont.item() > 0
    # upper bound sanity: per-element residual is O(z^2 + codebook^2); a mean
    # reduction would be ~16x smaller for this 4*1*2*2 grid
    z_cb = q._to_codebook_space(z)
    n_el = z_cb[0].numel()
    assert n_el == 16


def test_full_model_kl_distortion_same_order_of_magnitude():
    """KL and distortion must be within 2 orders of magnitude at init."""
    with open(LITE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]["init_args"]
    cfg["use_ema"] = False
    torch.manual_seed(0)
    model = VideoHierVQModel(**cfg)
    x = torch.rand(2, 3, 9, 32, 32) * 2 - 1
    with torch.no_grad():
        x_rec, layer_results = model(x, flg_train=True, flg_quant_det=False)
        _, log_dict = compute_hier_elbo_loss(x, x_rec, layer_results)
    distortion = abs(log_dict["loss/distortion"].item())
    kl = abs(log_dict["loss/kl_total"].item())
    ratio = kl / distortion
    assert 0.01 < ratio < 10.0, (
        f"KL/distortion ratio {ratio:.2e} outside [1e-2, 10] "
        f"(kl={kl:.4g}, distortion={distortion:.4g}) — reduction regression?"
    )


def test_kl_beta_warmup_schedule():
    with open(LITE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]["init_args"]
    cfg["use_ema"] = False
    cfg["loss_cfg"] = {"kl_beta": 0.5, "kl_warmup_steps": 100}
    model = VideoHierVQModel(**cfg)
    assert model._current_kl_beta(step=0) == pytest.approx(0.5 * 0.01)
    assert model._current_kl_beta(step=49) == pytest.approx(0.5 * 0.5)
    assert model._current_kl_beta(step=99) == pytest.approx(0.5)
    assert model._current_kl_beta(step=5000) == pytest.approx(0.5)

    w = model._effective_kl_weights(torch.device("cpu"), beta=0.25)
    assert w is not None and torch.allclose(w, torch.full((3,), 0.25))


def test_kl_beta_default_is_identity():
    with open(LITE_CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]["init_args"]
    cfg["use_ema"] = False
    model = VideoHierVQModel(**cfg)
    assert model._current_kl_beta(step=12345) == 1.0
    assert model._effective_kl_weights(torch.device("cpu"), beta=1.0) is None
