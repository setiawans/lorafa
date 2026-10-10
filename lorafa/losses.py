from typing import Dict, List, Tuple

import torch

COSINE_MODES = ("concat_eps", "concat", "blocks")

def cosine_distance(g_hat: torch.Tensor, g_star: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return 1.0 - torch.dot(g_hat, g_star) / (g_hat.norm() * g_star.norm() + eps)

def normalized_cosine_distance(g_hat: torch.Tensor, g_star: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.dot(g_hat / g_hat.norm(), g_star / g_star.norm())

def block_cosine_distance(g_hat: torch.Tensor, g_star: torch.Tensor, blocks: List[Tuple[str, int]]) -> torch.Tensor:
    sizes = [n for _, n in blocks]
    groups: Dict[str, Tuple[list, list]] = {}
    for (name, _), a, b in zip(blocks, g_hat.split(sizes), g_star.split(sizes)):
        ga, gb = groups.setdefault(name, ([], []))
        ga.append(a)
        gb.append(b)
    return torch.stack([normalized_cosine_distance(torch.cat(a), torch.cat(b)) for a, b in groups.values()]).mean()

def total_variation(x_raw: torch.Tensor) -> torch.Tensor:
    dh = (x_raw[:, :, 1:, :] - x_raw[:, :, :-1, :]).abs().mean()
    dw = (x_raw[:, :, :, 1:] - x_raw[:, :, :, :-1]).abs().mean()
    return dh + dw

def attack_loss(
    g_hat: torch.Tensor,
    g_star: torch.Tensor,
    x_raw: torch.Tensor,
    tv_weight: float,
    cosine: str = "concat_eps",
    blocks: List[Tuple[str, int]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if cosine == "concat_eps":
        cos = cosine_distance(g_hat, g_star)
    elif cosine == "concat":
        cos = normalized_cosine_distance(g_hat, g_star)
    elif cosine == "blocks":
        cos = block_cosine_distance(g_hat, g_star, blocks)
    else:
        raise ValueError(f"unknown cosine {cosine!r}, choose from {COSINE_MODES}")
    tv = total_variation(x_raw)
    total = cos + tv_weight * tv
    return total, {"loss": total.item(), "cos": cos.item(), "tv": tv.item()}