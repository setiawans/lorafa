import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def load_runs(root: Path, group: str):
    runs = {}
    for best in sorted(root.rglob("best.csv")):
        name = str(best.parent.relative_to(root)).replace("\\", "/")
        if name.startswith(group + "/"):
            df = pd.read_csv(best).drop_duplicates(["epoch", "img"], keep="last")
            runs[name.split("/", 1)[1]] = df
    return runs

def load_gt(indices, root):
    ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=T.ToTensor())
    return {int(i): ds[int(i)][0] for i in indices}

def to_img(t: torch.Tensor):
    return t.clamp(0, 1).permute(1, 2, 0).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="dev")
    ap.add_argument("--runs", default="runs/attack")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--variants", nargs="*", default=None, help="subset and order of variants (default: all)")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs), args.group)
    if not runs:
        raise SystemExit(f"no runs under {args.runs}/{args.group}")
    variants = args.variants or sorted(runs)
    epochs = sorted({int(e) for v in variants for e in runs[v]["epoch"]})
    imgs = sorted({int(i) for v in variants for i in runs[v]["img"]})
    gt = load_gt(imgs, args.data_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img in imgs:
        ncol = 1 + len(variants)
        fig, axes = plt.subplots(len(epochs), ncol, figsize=(1.6 * ncol, 1.75 * len(epochs)), squeeze=False)
        for r, epoch in enumerate(epochs):
            ax = axes[r, 0]
            ax.imshow(to_img(gt[img]), interpolation="nearest")
            ax.set_title("ground truth" if r == 0 else "", fontsize=8)
            ax.set_ylabel(f"epoch {epoch}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            for c, v in enumerate(variants, start=1):
                ax = axes[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(v, fontsize=8)
                row = runs[v][(runs[v]["epoch"] == epoch) & (runs[v]["img"] == img)]
                if row.empty:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                    ax.set_facecolor("#eee")
                    continue
                row = row.iloc[0]
                recon = torch.load(row["recon"], weights_only=True)["recon"][0]
                ax.imshow(to_img(recon), interpolation="nearest")
                ax.set_xlabel(f"{row['psnr']:.1f} dB  lp {row['lpips']:.3f}\nloss {row['final_loss']:.3f}",
                              fontsize=6.5)
        fig.suptitle(f"{args.group} / image {img}", fontsize=10)
        fig.tight_layout()
        path = out_dir / f"gallery_{args.group}_img{img:05d}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"wrote {path}")

if __name__ == "__main__":
    main()
