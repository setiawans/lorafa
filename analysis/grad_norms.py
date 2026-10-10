import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.checkpoints import load_lorafa  # noqa: E402
from lorafa.data import get_dataset, load_targets  # noqa: E402
from lorafa.losses import cosine_distance  # noqa: E402
from lorafa.observe import BN_MODES, OBS_MODES, bn_mode, observe  # noqa: E402
from lorafa.utils import device  # noqa: E402

CHECKPOINTS = sorted(Path("checkpoints/finetune").glob("*/epoch*.pt"))
IMAGES = list(range(100))
OUT = Path("runs/grad_norms.csv")

def checkpoint_rows(ckpt: Path, dev) -> list:
    model, meta, norm = load_lorafa(ckpt, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    x_all, y_all = load_targets(get_dataset("cifar10", "test", "data", norm), IMAGES)
    rows = []
    for i in IMAGES:
        x, y = x_all[i:i + 1].to(dev), y_all[i:i + 1].to(dev)
        with torch.no_grad(), bn_mode(model, "eval"):
            logits = model(x)
        for obs_mode in OBS_MODES:
            for bn in BN_MODES:
                g = observe(model, x, y, obs_mode, bn).detach()
                rows.append({
                    "checkpoint": str(ckpt), "arch": meta["arch"], "rank": meta["rank"],
                    "adapt_first_conv": meta["adapt_first_conv"], "backbone": meta["backbone"],
                    "epoch": int(meta["epoch"]), "obs_mode": obs_mode, "bn": bn, "img": i, "label": int(y),
                    "ce_loss": float(F.cross_entropy(logits, y)),
                    "p_true": float(F.softmax(logits, 1)[0, y]),
                    "correct": bool(logits.argmax(1) == y),
                    "m": g.numel(),
                    "grad_norm": float(g.double().norm()),
                    "grad_max_abs": float(g.abs().max()),
                    "n_zero": int((g == 0).sum()),
                    "cos_self_eps": float(cosine_distance(g, g)),
                })
    return rows

def main():
    dev = device()
    rows = [r for ckpt in CHECKPOINTS for r in checkpoint_rows(ckpt, dev)]
    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

if __name__ == "__main__":
    main()