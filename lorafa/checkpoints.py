import json
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from .data import Normalizer
from .models import attach_lora, build_model

def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def read_json(path) -> dict:
    with open(path) as f:
        return json.load(f)

def save_checkpoint(path, model: nn.Module, meta: dict, sidecar: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**meta, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    torch.save({"model": model.state_dict(), "meta": meta}, path)
    if sidecar:
        write_json(path.with_suffix(".json"), meta)

def load_checkpoint(path, device="cpu") -> Tuple[dict, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    return ckpt["model"], ckpt["meta"]

def load_backbone(path, device="cpu") -> Tuple[nn.Module, dict]:
    state, meta = load_checkpoint(path, device)
    model = build_model(meta["arch"], meta["num_classes"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), meta

def load_lorafa(path, device="cpu") -> Tuple[nn.Module, dict, Normalizer]:
    state, meta = load_checkpoint(path, device)
    model = build_model(meta["arch"], meta["num_classes"])
    attach_lora(model, meta["rank"], meta["alpha"], meta["adapt_first_conv"], meta["lora_seed"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), meta, Normalizer.from_meta(meta)