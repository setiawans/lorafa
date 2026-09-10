import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.checkpoints import load_backbone, save_checkpoint, write_json  # noqa: E402
from lorafa.data import Normalizer, get_dataset, get_loader, num_classes  # noqa: E402
from lorafa.models import attach_lora, build_model, count_params, lora_B_params, lora_layers, replace_head  # noqa: E402
from lorafa.train import evaluate, run_epoch  # noqa: E402
from lorafa.utils import CsvLogger, device, fmt_seconds, load_config, set_seed  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], allow_tf32=True)
    dev = device()
    amp = cfg["amp"] and dev.type == "cuda"
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg["backbone"] == "random":
        backbone = build_model(cfg["arch"], 1)
        norm = Normalizer.for_dataset(cfg["norm_dataset"])
        bmeta = {"arch": cfg["arch"], "norm_mean": list(norm.mean), "norm_std": list(norm.std),
                 "pretrain_dataset": None, "final_test_acc": None}
    else:
        backbone, bmeta = load_backbone(cfg["backbone"], dev)
        norm = Normalizer.from_meta(bmeta)
    n_cls = num_classes(cfg["dataset"])

    model = replace_head(backbone, n_cls)
    attach_lora(model, cfg["rank"], cfg["alpha"], cfg["adapt_first_conv"], cfg["lora_seed"])
    model.to(dev)

    train_set = get_dataset(cfg["dataset"], "train", cfg["data_root"], norm, augment=True)
    test_set = get_dataset(cfg["dataset"], "test", cfg["data_root"], norm)
    train_loader = get_loader(train_set, cfg["batch_size"], True, cfg["workers"], cfg["seed"])
    test_loader = get_loader(test_set, 256, False, cfg["workers"])

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=amp)

    meta = {
        "arch": bmeta["arch"],
        "num_classes": n_cls,
        "norm_mean": bmeta["norm_mean"],
        "norm_std": bmeta["norm_std"],
        "rank": cfg["rank"],
        "alpha": cfg["alpha"],
        "adapt_first_conv": cfg["adapt_first_conv"],
        "lora_seed": cfg["lora_seed"],
        "num_lora_layers": len(lora_layers(model)),
        "num_B_params": count_params(lora_B_params(model)),
        "num_trainable_params": count_params(trainable),
        "backbone": cfg["backbone"],
        "backbone_test_acc": bmeta.get("final_test_acc"),
        "pretrain_dataset": bmeta["pretrain_dataset"],
        "finetune_dataset": cfg["dataset"],
        "optimizer": "adam",
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch_size": cfg["batch_size"],
        "epochs": cfg["epochs"],
        "bn_affine_trainable": False,
        "bn_stats_frozen": cfg["freeze_bn_stats"],
        "amp": amp,
        "seed": cfg["seed"],
        "config": args.config,
    }
    print(f"{meta['arch']} r={cfg['rank']} L={meta['num_lora_layers']} "
          f"m={meta['num_B_params']:,} trainable={meta['num_trainable_params']:,} device={dev}")

    save_epochs = set(cfg["save_epochs"])
    log = CsvLogger(out_dir / "train_log.csv",
                    ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "seconds", "saved"])
    saved = {}

    def save(epoch, te_loss, te_acc):
        path = out_dir / f"epoch{epoch:03d}.pt"
        save_checkpoint(path, model, {**meta, "epoch": epoch,
                                      "test_loss": round(te_loss, 4), "test_acc": round(te_acc, 2)},
                        sidecar=False)
        saved[epoch] = {"path": str(path), "test_loss": round(te_loss, 4), "test_acc": round(te_acc, 2)}

    te_loss, te_acc = evaluate(model, test_loader, dev)
    print(f"epoch   0 test {te_loss:.4f}/{te_acc:.2f}%")
    log.log(epoch=0, test_loss=f"{te_loss:.4f}", test_acc=f"{te_acc:.2f}", saved=0 in save_epochs)
    if 0 in save_epochs:
        save(0, te_loss, te_acc)

    t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, dev, optimizer, scaler, amp, cfg["freeze_bn_stats"])
        te_loss, te_acc = evaluate(model, test_loader, dev)
        secs = time.time() - t0
        log.log(epoch=epoch, train_loss=f"{tr_loss:.4f}", train_acc=f"{tr_acc:.2f}",
                test_loss=f"{te_loss:.4f}", test_acc=f"{te_acc:.2f}", seconds=f"{secs:.1f}",
                saved=epoch in save_epochs)
        print(f"epoch {epoch:3d}/{cfg['epochs']} train {tr_loss:.4f}/{tr_acc:.2f}% "
              f"test {te_loss:.4f}/{te_acc:.2f}% ({secs:.0f}s, total {fmt_seconds(time.time() - t_start)})")
        if epoch in save_epochs:
            save(epoch, te_loss, te_acc)
    log.close()

    write_json(out_dir / "meta.json", {**meta, "checkpoints": saved})
    print(f"saved {len(saved)} checkpoints to {out_dir}")

if __name__ == "__main__":
    main()