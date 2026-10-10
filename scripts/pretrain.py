import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.checkpoints import save_checkpoint  # noqa: E402
from lorafa.data import Normalizer, get_dataset, get_loader, num_classes  # noqa: E402
from lorafa.models import build_model, count_params  # noqa: E402
from lorafa.train import run_epoch  # noqa: E402
from lorafa.utils import CsvLogger, device, fmt_seconds, load_config, set_seed  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"], allow_tf32=True)
    dev = device()
    amp = cfg["amp"] and dev.type == "cuda"

    out = Path(cfg["out"])
    last = out.with_suffix(".last.pt")
    log_path = out.with_suffix(".log.csv")

    norm = Normalizer.for_dataset(cfg["dataset"])
    train_set = get_dataset(cfg["dataset"], "train", cfg["data_root"], norm, augment=True)
    test_set = get_dataset(cfg["dataset"], "test", cfg["data_root"], norm)
    train_loader = get_loader(train_set, cfg["batch_size"], True, cfg["workers"], cfg["seed"])
    test_loader = get_loader(test_set, 256, False, cfg["workers"])

    model = build_model(cfg["arch"], num_classes(cfg["dataset"])).to(dev)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"], nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(enabled=amp)

    start, best = 1, 0.0
    if args.resume and last.exists():
        state = torch.load(last, map_location=dev, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best = state["epoch"] + 1, state["best"]
        print(f"resumed at epoch {start}")

    print(f"{cfg['arch']} on {cfg['dataset']} | params={count_params(model.parameters()):,} "
          f"| device={dev} amp={amp}")
    log = CsvLogger(log_path, ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc", "seconds"],
                    append=args.resume)

    t_start = time.time()
    for epoch in range(start, cfg["epochs"] + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, train_loader, dev, optimizer, scaler, amp)
        te_loss, te_acc = run_epoch(model, test_loader, dev)
        scheduler.step()
        best = max(best, te_acc)
        secs = time.time() - t0
        log.log(epoch=epoch, lr=f"{lr:.6f}", train_loss=f"{tr_loss:.4f}", train_acc=f"{tr_acc:.2f}",
                test_loss=f"{te_loss:.4f}", test_acc=f"{te_acc:.2f}", seconds=f"{secs:.1f}")
        print(f"epoch {epoch:3d}/{cfg['epochs']} lr={lr:.4f} train {tr_loss:.4f}/{tr_acc:.2f}% "
              f"test {te_loss:.4f}/{te_acc:.2f}% ({secs:.0f}s, total {fmt_seconds(time.time() - t_start)})")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best}, last)
    log.close()

    te_loss, te_acc = run_epoch(model, test_loader, dev)
    meta = {
        "arch": cfg["arch"],
        "num_classes": num_classes(cfg["dataset"]),
        "pretrain_dataset": cfg["dataset"],
        "norm_mean": list(norm.mean),
        "norm_std": list(norm.std),
        "params": count_params(model.parameters()),
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "optimizer": "sgd_nesterov",
        "lr": cfg["lr"],
        "momentum": cfg["momentum"],
        "weight_decay": cfg["weight_decay"],
        "schedule": "cosine",
        "amp": amp,
        "seed": cfg["seed"],
        "final_test_loss": round(te_loss, 4),
        "final_test_acc": round(te_acc, 2),
        "best_test_acc": round(best, 2),
        "config": args.config,
    }
    save_checkpoint(out, model, meta)
    last.unlink(missing_ok=True)
    print(f"saved {out}  test_acc={te_acc:.2f}%  best={best:.2f}%")

if __name__ == "__main__":
    main()