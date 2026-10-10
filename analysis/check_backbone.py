import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.checkpoints import load_checkpoint  # noqa: E402

ROOT = Path("checkpoints/finetune")

def diff_rows(run_dir: Path) -> list:
    state, meta = load_checkpoint(run_dir / "epoch000.pt")
    if meta["backbone"] == "random":
        return []
    backbone = load_checkpoint(meta["backbone"])[0]
    rows = []
    for key, value in state.items():
        backbone_key = key.replace(".weight0", ".weight").replace(".bias0", ".bias")
        if backbone_key in backbone and backbone[backbone_key].shape == value.shape:
            rows.append({
                "dir": run_dir.name, "key": key, "numel": value.numel(),
                "max_abs_diff": float((value.double() - backbone[backbone_key].double()).abs().max()),
            })
    return rows

def main():
    run_dirs = sorted(p for p in ROOT.iterdir() if (p / "epoch000.pt").exists())
    table = pd.DataFrame([r for d in run_dirs for r in diff_rows(d)])
    summary = table.groupby("dir").agg(keys=("key", "size"), max_abs_diff=("max_abs_diff", "max")).reset_index()
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()