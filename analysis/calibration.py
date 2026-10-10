import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.data import dataset_mean_image  # noqa: E402
from lorafa.metrics import all_metrics, trivial_baselines  # noqa: E402
from lorafa.utils import device  # noqa: E402

IMAGES = list(range(100))
OUT = Path("runs/calibration.csv")

def same_label_images(test_set, labels: list) -> torch.Tensor:
    unused = {label: [] for label in set(labels)}
    for index in range(len(IMAGES), len(test_set)):
        if test_set.targets[index] in unused:
            unused[test_set.targets[index]].append(index)
    return torch.stack([test_set[unused[label].pop(0)][0] for label in labels])

def null_row(control: str, img: int, image: torch.Tensor, x_all: torch.Tensor) -> dict:
    others = torch.cat([x_all[:img], x_all[img + 1:]])
    true = all_metrics(image[None], x_all[img:img + 1])
    null = all_metrics(image[None].expand(len(others), -1, -1, -1).contiguous(), others)["lpips"]
    return {"control": control, "img": img, "psnr": float(true["psnr"][0]), "lpips": float(true["lpips"][0]),
            "lpips_null_frac_better": float((null <= true["lpips"][0]).float().mean())}

def main():
    dev = device()
    test_set = torchvision.datasets.CIFAR10("data", train=False, transform=T.ToTensor())
    x_all = torch.stack([test_set[i][0] for i in IMAGES])
    labels = [test_set.targets[i] for i in IMAGES]
    controls = trivial_baselines(x_all, dataset_mean_image("cifar10", "data"))
    controls["same_label_image"] = same_label_images(test_set, labels)
    x_all = x_all.to(dev)
    rows = [null_row(name, i, images[i].to(dev), x_all) for name, images in controls.items() for i in IMAGES]
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

if __name__ == "__main__":
    main()