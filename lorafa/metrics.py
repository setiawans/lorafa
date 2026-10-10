from typing import Dict

import torch

_lpips_model = {}

def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = ((x - y) ** 2).flatten(1).mean(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))

def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    from torchmetrics.functional.image import structural_similarity_index_measure
    return structural_similarity_index_measure(x, y, data_range=1.0, reduction="none")

def lpips(x: torch.Tensor, y: torch.Tensor, net: str = "alex") -> torch.Tensor:
    import lpips as _lpips
    key = (net, x.device)
    if key not in _lpips_model:
        _lpips_model[key] = _lpips.LPIPS(net=net, verbose=False).to(x.device).eval()
    with torch.no_grad():
        return _lpips_model[key](x, y, normalize=True).flatten()

@torch.no_grad()
def all_metrics(x_hat: torch.Tensor, x_gt: torch.Tensor) -> Dict[str, torch.Tensor]:
    x_hat, x_gt = x_hat.clamp(0, 1), x_gt.clamp(0, 1)
    return {"psnr": psnr(x_hat, x_gt), "ssim": ssim(x_hat, x_gt), "lpips": lpips(x_hat, x_gt)}

@torch.no_grad()
def trivial_baselines(x_gt: torch.Tensor, dataset_mean: torch.Tensor) -> Dict[str, torch.Tensor]:
    n = x_gt.shape[0]
    return {
        "gray": torch.full_like(x_gt, 0.5),
        "mean_color": x_gt.mean(dim=(2, 3), keepdim=True).expand_as(x_gt).contiguous(),
        "dataset_mean": dataset_mean.to(x_gt.device).expand(n, -1, -1, -1).contiguous(),
    }