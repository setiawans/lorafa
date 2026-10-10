import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.data import dataset_mean_image  # noqa: E402
from lorafa.losses import total_variation  # noqa: E402
from lorafa.metrics import all_metrics, trivial_baselines  # noqa: E402

IMAGES = list(range(100))
OUT = Path("runs/baselines.csv")

def main():
    test_set = torchvision.datasets.CIFAR10("data", train=False, transform=T.ToTensor())
    x = torch.stack([test_set[i][0] for i in IMAGES])
    table = pd.DataFrame({
        "img": IMAGES,
        "label": [test_set[i][1] for i in IMAGES],
        "pixel_var": x.flatten(1).var(1).numpy(),
        "tv": [float(total_variation(x[i:i + 1])) for i in IMAGES],
    })
    for control, image in trivial_baselines(x, dataset_mean_image("cifar10", "data")).items():
        for metric, value in all_metrics(image, x).items():
            table[f"{control}_{metric}"] = value.numpy()
    OUT.parent.mkdir(exist_ok=True)
    table.to_csv(OUT, index=False)

if __name__ == "__main__":
    main()