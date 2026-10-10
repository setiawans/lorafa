import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.metrics import all_metrics  # noqa: E402
from lorafa.utils import device  # noqa: E402

ROOT = Path("runs/attack/main")

def null_row(row, x_all: torch.Tensor) -> dict:
    recon = torch.load(row.recon, weights_only=True)["recon"].to(x_all.device)
    others = torch.cat([x_all[:row.img], x_all[row.img + 1:]])
    record = {"epoch": row.epoch, "img": row.img}
    for metric, values in all_metrics(recon.expand(len(others), -1, -1, -1), others).items():
        values = values.cpu().numpy()
        true = getattr(row, metric)
        better = values <= true if metric == "lpips" else values >= true
        record[f"{metric}_null_mean"] = values.mean()
        record[f"{metric}_null_std"] = values.std(ddof=1)
        record[f"{metric}_null_frac_better"] = better.mean()
    return record

def main():
    dev = device()
    test_set = torchvision.datasets.CIFAR10("data", train=False, transform=T.ToTensor())
    x_all = torch.stack([test_set[i][0] for i in range(100)]).to(dev)
    for run_dir in sorted(p for p in ROOT.iterdir() if (p / "best.csv").exists()):
        best = pd.read_csv(run_dir / "best.csv").drop_duplicates(["epoch", "img"], keep="last")
        pd.DataFrame([null_row(row, x_all) for row in best.itertuples()]).to_csv(run_dir / "identification.csv", index=False)

if __name__ == "__main__":
    main()