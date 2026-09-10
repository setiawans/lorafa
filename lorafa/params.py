from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Normalizer

def linear_decay(sigma0: float, t: int, T: int) -> float:
    return sigma0 * (1.0 - t / max(T - 1, 1))

def make_scheduler(opt, cfg: dict, T: int):
    sched = cfg.get("schedule", {"type": "none"})
    kind = sched.get("type", "none")
    if kind == "none":
        return None
    if kind == "step":
        milestones = [int(T * f) for f in sched.get("milestones", [0.375, 0.625, 0.875])]
        return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=sched.get("gamma", 0.1))
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T, eta_min=sched.get("lr_min", 0.0))
    raise ValueError(f"unknown schedule {kind!r}")

class Parameterization:
    opt: torch.optim.Optimizer
    sched = None

    def image(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def step(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    def end_step(self) -> None:
        if self.sched is not None:
            self.sched.step()

    def lr(self) -> float:
        return self.opt.param_groups[0]["lr"]

    def inject_noise(self, sigma: float) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def final_raw(self) -> torch.Tensor:
        return self.image()[1].detach().clamp(0.0, 1.0)

class PixelParam(Parameterization):
    def __init__(self, x0_norm: torch.Tensor, norm: Normalizer, lr: float,
                 sign_grad: bool = True, optimizer: str = "adam", cfg: Optional[dict] = None, T: int = 1):
        self.norm = norm
        self.sign_grad = sign_grad
        self.lo, self.hi = norm.bounds(x0_norm)
        self.x = x0_norm.detach().clone().clamp(self.lo, self.hi).requires_grad_(True)
        if optimizer == "adam":
            self.opt = torch.optim.Adam([self.x], lr=lr)
        elif optimizer == "sgd":
            self.opt = torch.optim.SGD([self.x], lr=lr)
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}")
        self.sched = make_scheduler(self.opt, cfg or {}, T)

    def image(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.norm.denormalize(self.x)

    def step(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.sign_grad:
            self.x.grad.sign_()
        self.opt.step()
        with torch.no_grad():
            self.x.clamp_(self.lo, self.hi)

    @torch.no_grad()
    def inject_noise(self, sigma: float) -> None:
        self.x.add_(sigma * torch.randn_like(self.x)).clamp_(self.lo, self.hi)

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)

class DIPGenerator(nn.Module):
    def __init__(self, z_channels: int, hidden: int):
        super().__init__()
        h, h2 = hidden, hidden * 2
        self.enc1 = nn.Sequential(ConvBlock(z_channels, h), ConvBlock(h, h))
        self.enc2 = nn.Sequential(ConvBlock(h, h2, 2), ConvBlock(h2, h2))
        self.enc3 = nn.Sequential(ConvBlock(h2, h2, 2), ConvBlock(h2, h2))
        self.bottleneck = nn.Sequential(ConvBlock(h2, h2, 2), ConvBlock(h2, h2))
        self.dec3 = nn.Sequential(ConvBlock(h2 * 2, h2), ConvBlock(h2, h2))
        self.dec2 = nn.Sequential(ConvBlock(h2 * 2, h2), ConvBlock(h2, h))
        self.dec1 = nn.Sequential(ConvBlock(h * 2, h), ConvBlock(h, h))
        self.out = nn.Conv2d(h, 3, 1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="nearest")

    def forward(self, z):
        e1 = self.enc1(z)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(torch.cat([self._up(b, e3), e3], 1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], 1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], 1))
        return torch.sigmoid(self.out(d1))

class DIPParam(Parameterization):
    def __init__(self, shape, norm: Normalizer, lr: float, z_channels: int, hidden: int, device,
                 cfg: Optional[dict] = None, T: int = 1):
        self.norm = norm
        self.net = DIPGenerator(z_channels, hidden).to(device).train()
        self.z0 = torch.randn(shape[0], z_channels, *shape[-2:], device=device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.sched = make_scheduler(self.opt, cfg or {}, T)

    def image(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(self.z0)
        return self.norm.normalize(raw), raw

    def step(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

    @torch.no_grad()
    def inject_noise(self, sigma: float) -> None:
        for p in self.net.parameters():
            p.add_(sigma * torch.randn_like(p))

def make_param(cfg: dict, norm: Normalizer, shape, device, x0_norm: Optional[torch.Tensor] = None) -> Parameterization:
    kind = cfg["param"]
    if kind == "pixel":
        if x0_norm is None:
            x0_norm = norm.normalize(torch.rand(*shape, device=device))
        return PixelParam(x0_norm, norm, cfg["lr"], cfg.get("sign_grad", True), cfg.get("optimizer", "adam"),
                          cfg, cfg["iters"])
    if kind == "dip":
        if x0_norm is not None:
            raise ValueError("ground-truth initialization is only defined for param=pixel")
        return DIPParam(shape, norm, cfg["lr"], cfg["dip"]["z_channels"], cfg["dip"]["hidden"], device,
                        cfg, cfg["iters"])
    raise ValueError(f"unknown param {kind!r}")