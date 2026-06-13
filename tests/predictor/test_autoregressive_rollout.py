"""Inference mode tests: parallel vs autoregressive."""

import torch

from tests.predictor.stack_factory import v2_stack


def _stack():
    vae, prep, orch, stages, _ = v2_stack(n_layers=2, lambda_pred_mse=0.0)
    return vae, prep, orch, stages


def test_inf_01_parallel_oracle_uses_full_context():
    torch.manual_seed(0)
    vae, prep, orch, stages = _stack()
    vae.eval()
    stages.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        train_out = orch.forward_train(batch)
        par_out = orch.forward_parallel(batch)
    assert torch.isfinite(par_out.loss_ce)
    assert par_out.logits[(2, 0)].shape[1] > train_out.logits[(2, 0)].shape[1]


def test_inf_02_ar_differs_from_parallel():
    torch.manual_seed(1)
    vae, prep, orch, stages = _stack()
    vae.eval()
    stages.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        par_out = orch.forward_parallel(batch)
        ar_out = orch.forward_autoregressive(batch)
    assert torch.isfinite(ar_out.loss_ce)
    assert ar_out.ar_context_len is not None
    # Parallel oracle sees full native-T tokens; AR shift-0 uses context length only.
    assert par_out.logits[(2, 0)].shape[1] > ar_out.logits[(2, 0)].shape[1]


def test_inf_03_ar_context_growth():
    torch.manual_seed(2)
    vae, prep, orch, stages = _stack()
    vae.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        init_len = batch.context_embed[2].shape[1]
        ar_out = orch.forward_autoregressive(batch)
    assert ar_out.ar_context_len == init_len + 4 * 32 * 32


def test_inf_04_ar_decode_shape():
    torch.manual_seed(4)
    vae, prep, orch, stages = _stack()
    vae.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        ar_out = orch.forward_autoregressive(batch)
        level_indices = [ar_out.pred_indices[s] for s in range(3)]
        zero_acts = {k: torch.zeros_like(v) for k, v in batch.activations.items()}
        recon = vae.decode_from_indices(
            level_indices, zero_acts, encoder_bottleneck=batch.encoder_bottleneck,
        )
    assert recon.shape == video.shape


def test_inf_05_greedy_stable():
    torch.manual_seed(3)
    vae, prep, orch, stages = _stack()
    vae.eval()
    stages.eval()
    video = torch.randn(1, 3, 13, 64, 64)
    with torch.no_grad():
        batch = prep.encode_and_schedule(vae, video, stages)
        torch.manual_seed(3)
        a = orch.forward_autoregressive(batch)
        torch.manual_seed(3)
        b = orch.forward_autoregressive(batch)
    assert torch.isfinite(a.loss_ce)
    assert torch.allclose(a.loss_ce, b.loss_ce, rtol=0, atol=1e-6)
    for s in a.pred_indices:
        assert torch.equal(a.pred_indices[s], b.pred_indices[s])
