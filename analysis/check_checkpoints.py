import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.checkpoints import load_checkpoint  # noqa: E402

ROOT = Path("checkpoints/finetune")

def diff_rows(run_dir: Path) -> list:
    files = sorted(run_dir.glob("epoch*.pt"))
    base = load_checkpoint(files[0])[0]
    rows = []
    for f in files[1:]:
        state, meta = load_checkpoint(f)
        for key, value in state.items():
            rows.append({
                "dir": run_dir.name, "file": f.name, "epoch": meta["epoch"], "test_acc": meta["test_acc"],
                "key": key, "numel": value.numel(),
                "max_abs_diff": float((value.double() - base[key].double()).abs().max()),
            })
    return rows

def main():
    run_dirs = sorted(p for p in ROOT.iterdir() if (p / "epoch000.pt").exists())
    table = pd.DataFrame([r for d in run_dirs for r in diff_rows(d)])
    changed = table[table.max_abs_diff > 0]
    summary = changed.groupby(["dir", "file", "epoch", "test_acc"]).key.agg(
        B=lambda keys: int(keys.str.endswith(".B").sum()),
        other=lambda keys: ", ".join(keys[~keys.str.endswith(".B")])).reset_index()
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()