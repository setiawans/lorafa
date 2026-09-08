import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.models import (  # noqa: E402
    attach_lora, build_model, count_params, lora_A_params, lora_B_params,
    lora_layers, lora_W0_params, replace_head,
)
from lorafa.observe import (  # noqa: E402
    OBS_MODES, bn_mode, observe, observed_dim, observed_params,
    project_B_gradient, requires_grad, row_space_projector,
)

torch.manual_seed(0)

def make(arch="resnet20_4", rank=16, alpha=32.0, first_conv=False, dtype=torch.float32):
    model = build_model(arch, 100)
    replace_head(model, 10)
    attach_lora(model, rank, alpha, adapt_first_conv=first_conv, seed=1)
    return model.to(dtype).eval()

def batch(n=1, dtype=torch.float32):
    return torch.randn(n, 3, 32, 32, dtype=dtype), torch.randint(0, 10, (n,))

def test_lora_dimensions():
    expected = {"resnet20_4": (18, 43_008), "resnet18": (16, 61_440)}
    for arch, (L, m) in expected.items():
        model = make(arch)
        assert len(lora_layers(model)) == L
        assert count_params(lora_B_params(model)) == m
        assert model(batch(2)[0]).shape == (2, 10)
    assert len(lora_layers(make(first_conv=True))) == 19

def test_observed_dims_are_nested():
    model = make()
    dims = [observed_dim(model, mode) for mode in OBS_MODES]
    assert dims[0] == 43_008
    assert dims[1] == dims[0] + count_params(model.fc.parameters())
    assert dims[0] < dims[1] < dims[2] < dims[3]

def test_A_gradient_is_zero_at_init():
    model = make()
    x, y = batch()
    with requires_grad(lora_A_params(model)), bn_mode(model, "eval"):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        gA = torch.autograd.grad(loss, lora_A_params(model))
    assert all(torch.count_nonzero(g) == 0 for g in gA)

    for layer in lora_layers(model):
        layer.B.data.normal_(std=0.01)
    with requires_grad(lora_A_params(model)), bn_mode(model, "eval"):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        gA = torch.autograd.grad(loss, lora_A_params(model))
    assert all(torch.count_nonzero(g) > 0 for g in gA)

def test_bn_modes_differ_and_model_state_is_untouched():
    model = make()
    for layer in lora_layers(model):
        layer.B.data.normal_(std=0.01)
    x, y = batch()

    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    before = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone(),
               m.training, m.momentum) for m in bns]
    flags = [p.requires_grad for p in model.parameters()]

    g_train = observe(model, x, y, "full", bn="train")
    g_eval = observe(model, x, y, "full", bn="eval")
    assert not torch.allclose(g_train, g_eval)

    for m, (rm, rv, nbt, training, momentum) in zip(bns, before):
        assert torch.equal(m.running_mean, rm) and torch.equal(m.running_var, rv)
        assert torch.equal(m.num_batches_tracked, nbt)
        assert m.training == training and m.momentum == momentum
    assert not model.training
    assert [p.requires_grad for p in model.parameters()] == flags

def test_observe_is_deterministic():
    model = make()
    x, y = batch()
    assert torch.equal(observe(model, x, y, "lorafa", bn="eval"),
                       observe(model, x, y, "lorafa", bn="eval"))

def test_frozen_params_get_no_grad_under_double_backprop():
    model = make()
    for p in model.parameters():
        p.requires_grad_(False)
    x, y = batch()
    x = x.requires_grad_(True)
    g_star = observe(model, x.detach(), y, "lorafa_fc", bn="eval").detach()
    for _ in range(3):
        g_hat = observe(model, x, y, "lorafa_fc", bn="eval", create_graph=True)
        loss = 1.0 - torch.dot(g_hat, g_star) / (g_hat.norm() * g_star.norm())
        loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None for p in model.parameters())
    assert not any(p.requires_grad for p in model.parameters())

def test_eq7_projection():
    model = make(dtype=torch.float64)
    for layer in lora_layers(model):
        layer.B.data.normal_(std=0.01)
    x, y = batch(dtype=torch.float64)

    params = lora_B_params(model) + lora_W0_params(model)
    with requires_grad(params), bn_mode(model, "eval"):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss, params)
    n = len(lora_layers(model))
    gB, gW = grads[:n], grads[n:]

    for layer, grad_B, grad_W in zip(lora_layers(model), gB, gW):
        G = grad_W.reshape(grad_W.shape[0], -1)
        P = row_space_projector(layer)
        s = layer.scaling
        assert torch.allclose(grad_B, s * G @ layer.A.T, rtol=1e-9, atol=1e-12)
        assert torch.allclose(project_B_gradient(grad_B, layer), s * G @ P, rtol=1e-9, atol=1e-12)
        assert torch.allclose((G @ (torch.eye(P.shape[0], dtype=P.dtype) - P)) @ layer.A.T,
                              torch.zeros_like(grad_B), atol=1e-12)

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")