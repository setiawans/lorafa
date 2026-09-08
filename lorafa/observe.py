from contextlib import contextmanager
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import LoRAConv2d, lora_layers

OBS_MODES = ("lorafa", "lorafa_fc", "lora", "full")
BN_MODES = ("train", "eval")

def observed_params(model: nn.Module, obs_mode: str) -> List[nn.Parameter]:
    if obs_mode not in OBS_MODES:
        raise ValueError(f"unknown obs_mode {obs_mode!r}, choose from {OBS_MODES}")
    layers = lora_layers(model)
    if obs_mode == "lorafa":
        return [m.B for m in layers]
    if obs_mode == "lorafa_fc":
        return [m.B for m in layers] + list(model.fc.parameters())
    if obs_mode == "lora":
        return [p for m in layers for p in (m.A, m.B)] + list(model.fc.parameters())
    lora_factors = {id(p) for m in layers for p in (m.A, m.B)}
    return [p for p in model.parameters() if id(p) not in lora_factors]

@contextmanager
def bn_mode(model: nn.Module, mode: str):
    if mode not in BN_MODES:
        raise ValueError(f"unknown bn_mode {mode!r}, choose from {BN_MODES}")
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [(m.training, m.momentum, m.num_batches_tracked.clone()) for m in bns]
    for m in bns:
        m.train(mode == "train")
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, (training, momentum, nbt) in zip(bns, saved):
            m.train(training)
            m.momentum = momentum
            m.num_batches_tracked.copy_(nbt)

@contextmanager
def requires_grad(params: List[nn.Parameter]):
    saved = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    try:
        yield
    finally:
        for p, r in zip(params, saved):
            p.requires_grad_(r)

def observe(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    obs_mode: str = "lorafa",
    bn: str = "train",
    create_graph: bool = False,
) -> torch.Tensor:
    params = observed_params(model, obs_mode)
    with bn_mode(model, bn), requires_grad(params):
        loss = F.cross_entropy(model(x), y)
        grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in grads])

def observed_dim(model: nn.Module, obs_mode: str) -> int:
    return sum(p.numel() for p in observed_params(model, obs_mode))

def row_space_projector(layer: LoRAConv2d) -> torch.Tensor:
    A = layer.A
    return A.T @ torch.linalg.solve(A @ A.T, A)

def project_B_gradient(grad_B: torch.Tensor, layer: LoRAConv2d) -> torch.Tensor:
    A = layer.A
    return grad_B @ torch.linalg.solve(A @ A.T, A)