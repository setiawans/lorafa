import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.checkpoints import load_lorafa, read_json, write_json  # noqa: E402
from lorafa.data import get_dataset, load_targets, target_indices  # noqa: E402
from lorafa.losses import attack_loss  # noqa: E402
from lorafa.metrics import all_metrics, psnr  # noqa: E402
from lorafa.observe import observe  # noqa: E402
from lorafa.params import linear_decay, make_param  # noqa: E402
from lorafa.utils import CsvLogger, device, fmt_seconds, load_config, run_name, save_config, set_seed  # noqa: E402

RESTART_FIELDS = ["epoch", "img", "label", "restart", "seed", "final_loss", "final_cos", "final_tv",
                  "psnr", "ssim", "lpips", "seconds", "recon"]
BEST_FIELDS = ["epoch", "img", "label", "gt_loss", "best_restart", "final_loss", "final_cos",
               "psnr", "ssim", "lpips", "recon", "image"]
CURVE_FIELDS = ["epoch", "img", "restart", "iter", "loss", "cos", "tv", "psnr"]

def restart_seed(base: int, img: int, restart: int) -> int:
    return base * 1_000_000 + img * 1_000 + restart

def read_rows(path, key_fields):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {tuple(row[k] for k in key_fields): row for row in csv.DictReader(f)}

def final_eval(model, recon_raw, x_gt_raw, y, g_star, norm, cfg):
    g_hat = observe(model, norm.normalize(recon_raw), y, cfg["obs_mode"], cfg["bn"]).detach()
    _, parts = attack_loss(g_hat, g_star, recon_raw, cfg["tv_weight"])
    if not all(math.isfinite(v) for v in parts.values()) or not torch.isfinite(recon_raw).all():
        print("  WARNING: non-finite reconstruction or loss; restart recorded with loss=inf")
        parts = {k: float("inf") for k in parts}
        recon_raw = torch.nan_to_num(recon_raw, nan=0.5, posinf=1.0, neginf=0.0)
    m = all_metrics(recon_raw, x_gt_raw)
    return parts, {k: float(v[0]) for k, v in m.items()}, recon_raw

def run_restart(model, cfg, norm, x_gt_norm, x_gt_raw, y, g_star, seed, dev, curve, epoch, img, restart):
    set_seed(seed)
    x0 = None
    if cfg["init"] == "gt":
        x0 = x_gt_norm + cfg.get("init_noise", 0.0) * torch.randn_like(x_gt_norm)
    param = make_param(cfg, norm, x_gt_norm.shape, dev, x0_norm=x0)
    T, sigma0, log_every = cfg["iters"], cfg["noise"]["sigma0"], cfg["log_every"]
    t0 = time.time()

    for t in range(T):
        x_norm, x_raw = param.image()
        g_hat = observe(model, x_norm, y, cfg["obs_mode"], cfg["bn"], create_graph=True)
        loss, parts = attack_loss(g_hat, g_star, x_raw, cfg["tv_weight"])
        if t % log_every == 0 or t == T - 1:
            with torch.no_grad():
                p = float(psnr(x_raw.detach().clamp(0, 1), x_gt_raw)[0])
            curve.log(epoch=epoch, img=img, restart=restart, iter=t,
                      loss=f"{parts['loss']:.6f}", cos=f"{parts['cos']:.6f}", tv=f"{parts['tv']:.6f}",
                      psnr=f"{p:.3f}")
        param.step(loss)
        if sigma0 > 0:
            param.inject_noise(linear_decay(sigma0, t, T))

    recon = param.final_raw()
    return recon, time.time() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.get("label", "known") != "known":
        raise NotImplementedError("only the known-label setting is implemented")
    set_seed(cfg["seed"])
    dev = device()
    run_dir = Path("runs") / run_name(args.config)
    existing = [f for f in ("best.csv", "restarts.csv", "curves.csv", "config.yaml") if (run_dir / f).exists()]
    if existing and not args.resume:
        raise FileExistsError(f"{run_dir} already contains {existing}; pass --resume to continue it")
    if args.resume and (run_dir / "config.yaml").exists():
        saved = load_config(run_dir / "config.yaml")
        if saved != cfg:
            diff = {k: (saved.get(k), cfg.get(k)) for k in set(saved) | set(cfg) if saved.get(k) != cfg.get(k)}
            raise ValueError(f"config differs from the saved run config; refusing to resume. diff={diff}")
    for sub in ("recon", "images"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    save_config(run_dir / "config.yaml", cfg)
    env = {
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": sys.version.split()[0],
    }
    env_path = run_dir / "environment.json"
    if env_path.exists():
        saved_env = read_json(env_path)
        if saved_env != env:
            if not args.allow_env_mismatch:
                raise RuntimeError(f"environment differs from the one this run was started on; results would "
                                   f"mix hardware. saved={saved_env} current={env}. "
                                   f"Pass --allow-env-mismatch to continue anyway.")
            mixed_path = run_dir / "environment_mixed.json"
            history = read_json(mixed_path) if mixed_path.exists() else []
            history.append({**env, "resumed_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            write_json(mixed_path, history)
            print(f"WARNING: continuing on a different environment; this run now mixes hardware. "
                  f"original={saved_env['device']} current={env['device']}. Recorded in {mixed_path}.")
    else:
        write_json(env_path, env)

    done_best = read_rows(run_dir / "best.csv", ["epoch", "img"])
    done_restarts = read_rows(run_dir / "restarts.csv", ["epoch", "img", "restart"])
    restarts_log = CsvLogger(run_dir / "restarts.csv", RESTART_FIELDS, append=True)
    best_log = CsvLogger(run_dir / "best.csv", BEST_FIELDS, append=True)
    curve_log = CsvLogger(run_dir / "curves.csv", CURVE_FIELDS, append=True)

    indices = target_indices(cfg["targets"]["n"], cfg["targets"]["offset"])
    all_metrics(torch.rand(1, 3, 32, 32, device=dev), torch.rand(1, 3, 32, 32, device=dev))
    print(f"run={run_name(args.config)} device={dev} param={cfg['param']} obs={cfg['obs_mode']} "
          f"bn={cfg['bn']} init={cfg['init']}+{cfg.get('init_noise', 0.0)} sigma0={cfg['noise']['sigma0']} "
          f"targets={len(indices)} restarts={cfg['restarts']} iters={cfg['iters']}")

    t_start = time.time()
    for ckpt in cfg["checkpoints"]:
        model, meta, norm = load_lorafa(ckpt, dev)
        for p in model.parameters():
            p.requires_grad_(False)
        epoch = int(meta["epoch"])
        test_set = get_dataset(cfg["dataset"], "test", cfg["data_root"], norm)
        x_all, y_all = load_targets(test_set, indices)
        print(f"checkpoint {ckpt} (epoch {epoch}, test_acc {meta.get('test_acc')})")

        for i, img in enumerate(indices):
            if (str(epoch), str(img)) in done_best:
                continue
            x_gt_norm = x_all[i:i + 1].to(dev)
            y = y_all[i:i + 1].to(dev)
            x_gt_raw = norm.denormalize(x_gt_norm).clamp(0, 1)
            g_star = observe(model, x_gt_norm, y, cfg["obs_mode"], cfg["bn"]).detach()
            _, gt_parts = attack_loss(g_star, g_star, x_gt_raw, cfg["tv_weight"])

            best = None
            for r in range(cfg["restarts"]):
                key = (str(epoch), str(img), str(r))
                recon_path = run_dir / "recon" / f"epoch{epoch:03d}_img{img:05d}_r{r:02d}.pt"
                if key in done_restarts:
                    saved = torch.load(done_restarts[key]["recon"], weights_only=True)
                    result = {k: float(saved[k]) for k in ("final_loss", "final_cos", "psnr", "ssim", "lpips")}
                    result.update(recon=done_restarts[key]["recon"], restart=r)
                else:
                    seed = restart_seed(cfg["seed"], img, r)
                    recon, secs = run_restart(model, cfg, norm, x_gt_norm, x_gt_raw, y, g_star, seed, dev,
                                              curve_log, epoch, img, r)
                    parts, m, recon = final_eval(model, recon, x_gt_raw, y, g_star, norm, cfg)
                    torch.save({"recon": recon.cpu(), "epoch": epoch, "img": img, "restart": r,
                                "seed": seed, "final_loss": parts["loss"], "final_cos": parts["cos"], **m},
                               recon_path)
                    result = {"final_loss": parts["loss"], "final_cos": parts["cos"], **m,
                              "recon": str(recon_path), "restart": r}
                    restarts_log.log(epoch=epoch, img=img, label=int(y), restart=r, seed=seed,
                                     final_loss=f"{parts['loss']:.6f}", final_cos=f"{parts['cos']:.6f}",
                                     final_tv=f"{parts['tv']:.6f}", psnr=f"{m['psnr']:.3f}",
                                     ssim=f"{m['ssim']:.4f}", lpips=f"{m['lpips']:.4f}",
                                     seconds=f"{secs:.1f}", recon=str(recon_path))
                    print(f"  epoch {epoch:3d} img {img:5d} r{r} loss={parts['loss']:.5f} "
                          f"psnr={m['psnr']:.2f} ssim={m['ssim']:.3f} lpips={m['lpips']:.3f} "
                          f"({secs:.0f}s, total {fmt_seconds(time.time() - t_start)})")
                if best is None or result["final_loss"] < best["final_loss"]:
                    best = result

            best_recon = torch.load(best["recon"], weights_only=True)["recon"]
            image_path = run_dir / "images" / f"epoch{epoch:03d}_img{img:05d}.png"
            save_image(torch.cat([x_gt_raw.cpu(), best_recon]), image_path, nrow=2)
            best_log.log(epoch=epoch, img=img, label=int(y), gt_loss=f"{gt_parts['loss']:.6f}",
                         best_restart=best["restart"], final_loss=f"{best['final_loss']:.6f}",
                         final_cos=f"{best['final_cos']:.6f}", psnr=f"{best['psnr']:.3f}",
                         ssim=f"{best['ssim']:.4f}", lpips=f"{best['lpips']:.4f}",
                         recon=best["recon"], image=str(image_path))
            print(f"BEST  epoch {epoch:3d} img {img:5d} r{best['restart']} loss={best['final_loss']:.5f} "
                  f"psnr={best['psnr']:.2f} lpips={best['lpips']:.3f}")

        del model
        torch.cuda.empty_cache()

    for log in (restarts_log, best_log, curve_log):
        log.close()
    print(f"done in {fmt_seconds(time.time() - t_start)} -> {run_dir}")

if __name__ == "__main__":
    main()