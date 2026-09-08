import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from scipy import stats
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.checkpoints import read_json  # noqa: E402
from lorafa.data import dataset_mean_image  # noqa: E402
from lorafa.metrics import all_metrics, trivial_baselines  # noqa: E402
from lorafa.utils import load_config  # noqa: E402

METRICS = ["psnr", "ssim", "lpips", "final_loss"]
PAIRS = [
    ("pixel", "pixel_noise"), ("dip", "dip_noise"), ("pixel", "dip"),
    ("pixel", "obs_lorafa_fc"), ("pixel", "obs_lora"), ("pixel", "obs_full"), ("pixel", "bn_train"),
]

def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))

def summarize(df: pd.DataFrame, cols) -> dict:
    out = {"n": len(df)}
    for c in cols:
        v = df[c].astype(float).to_numpy()
        out[f"{c}_mean"] = v.mean()
        out[f"{c}_std"] = v.std(ddof=1) if len(v) > 1 else float("nan")
        out[f"{c}_ci95"] = ci95(v)
    return out

def load_runs(root: Path):
    runs = {}
    for best in sorted(root.rglob("best.csv")):
        run_dir = best.parent
        name = str(run_dir.relative_to(root)).replace("\\", "/")
        curves = None
        if (run_dir / "curves.csv").exists():
            curves = (pd.read_csv(run_dir / "curves.csv")
                      .drop_duplicates(["epoch", "img", "restart", "iter"], keep="last"))
        env = read_json(run_dir / "environment.json") if (run_dir / "environment.json").exists() else {}
        runs[name] = {
            "dir": run_dir,
            "cfg": load_config(run_dir / "config.yaml"),
            "device": env.get("device", "unknown"),
            "mixed": (run_dir / "environment_mixed.json").exists(),
            "best": pd.read_csv(best).drop_duplicates(["epoch", "img"], keep="last").sort_values(["epoch", "img"]),
            "restarts": pd.read_csv(run_dir / "restarts.csv")
                          .drop_duplicates(["epoch", "img", "restart"], keep="last"),
            "curves": curves,
        }
    return runs

def load_gt(indices, root: str) -> torch.Tensor:
    ds = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=T.ToTensor())
    return torch.stack([ds[int(i)][0] for i in indices])

def load_recons(best: pd.DataFrame) -> torch.Tensor:
    return torch.cat([torch.load(p, weights_only=True)["recon"] for p in best["recon"]])

def controls_table(x_gt: torch.Tensor, data_root: str) -> pd.DataFrame:
    rows = []
    for name, img in trivial_baselines(x_gt, dataset_mean_image("cifar10", data_root)).items():
        m = all_metrics(img, x_gt)
        rows.append({"control": name, **summarize(pd.DataFrame({k: v.numpy() for k, v in m.items()}),
                                                   ["psnr", "ssim", "lpips"])})
    return pd.DataFrame(rows)

def mismatch_rows(name: str, epoch: int, best: pd.DataFrame, x_gt: torch.Tensor) -> dict:
    recon = load_recons(best)
    shifted = torch.roll(x_gt, shifts=1, dims=0)
    true = all_metrics(recon, x_gt)
    null = all_metrics(recon, shifted)
    row = {"run": name, "epoch": epoch, "n": len(best)}
    for k in ("psnr", "ssim", "lpips"):
        t, n = true[k].numpy(), null[k].numpy()
        row[f"{k}_true"] = t.mean()
        row[f"{k}_null"] = n.mean()
        row[f"{k}_null_ci95"] = ci95(n)
        row[f"{k}_p_paired_t"] = stats.ttest_rel(t, n).pvalue if len(t) > 1 else float("nan")
    return row

def init_sweep_rows(name: str, r: dict) -> list:
    cfg = r["cfg"]
    if cfg.get("init") != "gt" or r["curves"] is None:
        return []
    rows = []
    first = r["curves"][r["curves"]["iter"] == 0].groupby(["epoch", "img"])["psnr"].mean()
    sigma = float(cfg.get("init_noise", 0.0))
    for epoch, best in r["best"].groupby("epoch"):
        start = first.loc[epoch].reindex(best["img"]).to_numpy(dtype=float)
        end = best["psnr"].astype(float).to_numpy()
        rows.append({
            "run": name, "epoch": epoch, "init_noise": sigma, "n": len(best),
            "restarts": cfg["restarts"],
            "psnr_start_mean": float("inf") if sigma == 0 else start.mean(),
            "psnr_end_mean": end.mean(),
            "psnr_end_ci95": ci95(end),
            "psnr_gain_mean": float("nan") if sigma == 0 else (end - start).mean(),
            "lpips_end_mean": best["lpips"].astype(float).mean(),
            "final_loss_mean": best["final_loss"].astype(float).mean(),
            "gt_loss_mean": best["gt_loss"].astype(float).mean(),
        })
    return rows

def paired_rows(runs, group: str, a: str, b: str):
    ra, rb = runs.get(f"{group}/{a}"), runs.get(f"{group}/{b}")
    if ra is None or rb is None:
        return []
    rows = []
    for epoch in sorted(set(ra["best"]["epoch"]) & set(rb["best"]["epoch"])):
        da = ra["best"][ra["best"]["epoch"] == epoch].set_index("img")
        db = rb["best"][rb["best"]["epoch"] == epoch].set_index("img")
        common = da.index.intersection(db.index)
        if len(common) < 3:
            continue
        for m in ("psnr", "ssim", "lpips", "final_loss"):
            x, y = da.loc[common, m].astype(float), db.loc[common, m].astype(float)
            rows.append({
                "group": group, "a": a, "b": b, "epoch": epoch, "metric": m, "n": len(common),
                "a_mean": x.mean(), "b_mean": y.mean(), "diff_mean": (y - x).mean(),
                "diff_ci95": ci95((y - x).to_numpy()),
                "p_paired_t": stats.ttest_rel(x, y).pvalue,
                "p_wilcoxon": stats.wilcoxon(x, y).pvalue if (x != y).any() else 1.0,
            })
    return rows

def plot_curves(runs, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, r in runs.items():
        if r["curves"] is None or r["curves"].empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for epoch, g in r["curves"].groupby("epoch"):
            agg = g.groupby("iter")["cos"].quantile([0.25, 0.5, 0.75]).unstack()
            ax.plot(agg.index, agg[0.5], label=f"epoch {epoch}")
            ax.fill_between(agg.index, agg[0.25], agg[0.75], alpha=0.2)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("cosine loss (median, IQR)")
        ax.set_title(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"curve_{name.replace('/', '_')}.png", dpi=150)
        plt.close(fig)

def save_grids(runs, x_gt_by_img: dict, out_dir: Path, k: int = 8):
    for name, r in runs.items():
        for epoch, best in r["best"].groupby("epoch"):
            best = best.sort_values("img").head(k)
            recon = load_recons(best)
            gt = torch.stack([x_gt_by_img[int(i)] for i in best["img"]])
            save_image(torch.cat([gt, recon]), out_dir / f"grid_{name.replace('/', '_')}_epoch{epoch:03d}.png",
                       nrow=len(best), padding=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/attack")
    ap.add_argument("--group", default="main")
    ap.add_argument("--out", default="results")
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    root = Path(args.runs)
    tables, figures = Path(args.out) / "tables", Path(args.out) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    runs = {k: v for k, v in load_runs(root).items() if k.startswith(args.group + "/")}
    if not runs:
        raise SystemExit(f"no finished runs under {root}/{args.group}")

    all_imgs = sorted({int(i) for r in runs.values() for i in r["best"]["img"]})
    x_gt_all = load_gt(all_imgs, args.data_root)
    x_gt_by_img = {i: x_gt_all[j] for j, i in enumerate(all_imgs)}

    summary, mismatch, paired, sweep = [], [], [], []
    for name, r in runs.items():
        cfg = r["cfg"]
        for epoch, best in r["best"].groupby("epoch"):
            best = best.sort_values("img")
            row = {"run": name, "epoch": epoch, "param": cfg["param"], "obs_mode": cfg["obs_mode"],
                   "bn": cfg["bn"], "init": cfg["init"], "sigma0": cfg["noise"]["sigma0"],
                   "restarts": cfg["restarts"], "iters": cfg["iters"],
                   **summarize(best, METRICS + ["gt_loss"])}
            summary.append(row)
            x_gt = torch.stack([x_gt_by_img[int(i)] for i in best["img"]])
            mismatch.append(mismatch_rows(name, epoch, best, x_gt))
        sweep += init_sweep_rows(name, r)
    for a, b in PAIRS:
        paired += paired_rows(runs, args.group, a, b)

    controls = controls_table(x_gt_all, args.data_root)
    pd.DataFrame(summary).to_csv(tables / f"summary_{args.group}.csv", index=False)
    pd.DataFrame(mismatch).to_csv(tables / f"mismatch_{args.group}.csv", index=False)
    pd.DataFrame(paired).to_csv(tables / f"paired_{args.group}.csv", index=False)
    if sweep:
        pd.DataFrame(sweep).sort_values(["epoch", "init_noise"]).to_csv(
            tables / f"init_sweep_{args.group}.csv", index=False)
    controls.to_csv(tables / "controls.csv", index=False)
    with open(tables / f"sources_{args.group}.txt", "w") as f:
        for name, r in runs.items():
            flag = "\tMIXED-HARDWARE" if r["mixed"] else ""
            f.write(f"{name}\t{r['dir']}\t{len(r['best'])} best rows\t{len(r['restarts'])} restarts"
                    f"\t{r['device']}{flag}\n")
    devices = {r["device"] for r in runs.values()}
    mixed = [n for n, r in runs.items() if r["mixed"]]
    if len(devices) > 1:
        print(f"WARNING: runs in group '{args.group}' come from different GPUs: {sorted(devices)}. "
              f"Numbers are not bitwise comparable across hardware.")
    if mixed:
        print(f"WARNING: runs that mix hardware within themselves: {mixed}")

    plot_curves(runs, figures)
    save_grids(runs, x_gt_by_img, figures)

    show = ["run", "epoch", "n", "psnr_mean", "psnr_ci95", "ssim_mean", "lpips_mean", "final_loss_mean", "gt_loss_mean"]
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summary)[show].round(4).to_string(index=False))
    print("\ncontrols:")
    print(controls.round(4).to_string(index=False))
    print(f"\nwritten to {tables} and {figures}")

if __name__ == "__main__":
    main()