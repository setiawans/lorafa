from typing import Dict, Tuple

import torch

def cosine_distance(g_hat: torch.Tensor, g_star: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return 1.0 - torch.dot(g_hat, g_star) / (g_hat.norm() * g_star.norm() + eps)

def total_variation(x_raw: torch.Tensor) -> torch.Tensor:
    dh = (x_raw[:, :, 1:, :] - x_raw[:, :, :-1, :]).abs().mean()
    dw = (x_raw[:, :, :, 1:] - x_raw[:, :, :, :-1]).abs().mean()
    return dh + dw

def attack_loss(
    g_hat: torch.Tensor,
    g_star: torch.Tensor,
    x_raw: torch.Tensor,
    tv_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    cos = cosine_distance(g_hat, g_star)
    tv = total_variation(x_raw)
    total = cos + tv_weight * tv
    return total, {"loss": total.item(), "cos": cos.item(), "tv": tv.item()}