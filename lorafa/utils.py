import csv
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

def set_seed(seed: int, allow_tf32: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.use_deterministic_algorithms(True, warn_only=True)

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("extends", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg

def save_config(path, cfg: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def run_name(config_path, root: str = "configs") -> str:
    rel = Path(config_path).resolve().relative_to(Path(root).resolve())
    return str(rel.with_suffix("")).replace("\\", "/")

class CsvLogger:
    def __init__(self, path, fields: Iterable[str], append: bool = False):
        self.path = Path(path)
        self.fields = list(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not (append and self.path.exists() and self.path.stat().st_size > 0)
        self._f = open(self.path, "a" if append else "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fields)
        if write_header:
            self._w.writeheader()
            self._f.flush()

    def log(self, **row) -> None:
        self._w.writerow({k: row.get(k, "") for k in self.fields})
        self._f.flush()

    def close(self) -> None:
        self._f.close()

def fmt_seconds(s: float) -> str:
    s = int(s)
    return f"{s // 3600:d}:{(s % 3600) // 60:02d}:{s % 60:02d}"