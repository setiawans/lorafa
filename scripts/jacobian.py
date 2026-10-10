import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.autograd.forward_ad as fwAD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lorafa.checkpoints import load_lorafa  # noqa: E402
from lorafa.data import get_dataset, load_targets, target_indices  # noqa: E402
from lorafa.metrics import psnr  # noqa: E402
from lorafa.models import BasicBlock  # noqa: E402
from lorafa.observe import bn_mode, observe  # noqa: E402
from lorafa.utils import CsvLogger, device, fmt_seconds, load_config, save_config, set_seed  # noqa: E402

SCALING_FIELDS = ["epoch", "img", "sigma", "probes", "loss_mean", "loss_std", "loss_min", "loss_max",
                  "dg_norm_mean", "flips_mean", "flips_frac_mean", "psnr_mean", "pred_quadratic"]
EXPONENT_FIELDS = ["epoch", "img", "slope_loss", "intercept_loss", "r2_loss",
                   "slope_dg", "r2_dg", "slope_flips", "r2_flips",
                   "sigma_lo", "sigma_hi", "quad_ratio_at_probe"]
SPECTRUM_FIELDS = ["epoch", "img", "label", "n", "m", "sigma_max", "sigma_min", "fro_sq", "nuclear",
                   "eff_rank", "part_ratio", "r50", "r90", "r95", "r99", "grad_norm", "seconds"]
THRESHOLD_FIELDS = ["epoch", "img", "threshold", "visible", "near_null", "near_null_pct"]
PROJECTION_FIELDS = ["epoch", "img", "run", "psnr", "band", "sigma_lo", "sigma_hi", "dims",
                     "energy_frac", "energy_frac_random"]
FD_FIELDS = ["epoch", "img", "step", "dtype", "probes", "rel_err", "rel_fro", "ratio_sigma_max", "corr_log_sigma",
             "flips_mean"]

def normalized_grad(model, x, y, obs_mode, bn, create_graph=True):
    g = observe(model, x, y, obs_mode, bn, create_graph=create_graph)
    return g / g.norm()

def cosine_jacobian_jvp(model, x, y, obs_mode, bn) -> torch.Tensor:
    n = x.numel()
    cols = []
    base = x.detach()
    for j in range(n):
        tangent = torch.zeros(n, device=x.device, dtype=x.dtype)
        tangent[j] = 1.0
        with fwAD.dual_level():
            dual = fwAD.make_dual(base, tangent.view_as(base))
            gn = normalized_grad(model, dual, y, obs_mode, bn, create_graph=False)
            col = fwAD.unpack_dual(gn).tangent
            cols.append(col.detach().clone())
    return torch.stack(cols, dim=1)

def cosine_jacobian_vjp(model, x, y, obs_mode, bn, chunk: int = 512) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    gn = normalized_grad(model, x, y, obs_mode, bn)
    m = gn.numel()
    rows = []
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        for i in range(start, stop):
            v = torch.zeros_like(gn)
            v[i] = 1.0
            (row,) = torch.autograd.grad(gn, x, grad_outputs=v, retain_graph=True)
            rows.append(row.reshape(-1))
    return torch.stack(rows)

class PreactivationProbe:
    def __init__(self, model):
        self.signs = []
        self.handles = [m.register_forward_hook(self._hook) for name, m in model.named_modules()
                        if name.endswith("bn1") or isinstance(m, BasicBlock)]

    def _hook(self, module, inp, out):
        self.signs.append((out.detach() > 0).flatten())

    def capture(self, model, x, bn):
        self.signs = []
        with torch.no_grad(), bn_mode(model, bn):
            model(x)
        return torch.cat(self.signs)

    def close(self):
        for h in self.handles:
            h.remove()

def cosine_loss_at(model, x, y, g_star, obs_mode, bn) -> float:
    g = observe(model, x, y, obs_mode, bn).detach()
    return float(1.0 - torch.dot(g, g_star) / (g.norm() * g_star.norm()))

def scaling_sweep(model, x_star, y, g_star, obs_mode, bn, norm, sigmas, probes, seed, probe):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gbar_star = g_star.double() / g_star.double().norm()
    x_raw = norm.denormalize(x_star)
    signs_star = probe.capture(model, x_star, bn) if probe else None
    n_units = signs_star.numel() if signs_star is not None else 0
    rows = []
    for sigma in sigmas:
        losses, devs, flips, psnrs = [], [], [], []
        for k in range(probes):
            d = torch.randn(x_star.shape, generator=gen).to(x_star.device, x_star.dtype) * sigma
            xp = x_star + d
            g = observe(model, xp, y, obs_mode, bn).double()
            dist = float((g / g.norm() - gbar_star).norm())
            losses.append(0.5 * dist ** 2)
            devs.append(dist)
            psnrs.append(float(psnr(norm.denormalize(xp), x_raw)[0]))
            if probe is not None:
                signs = probe.capture(model, xp, bn)
                flips.append(float((signs != signs_star).sum()))
        rows.append({"sigma": sigma, "probes": probes,
                     "loss_mean": float(np.mean(losses)), "loss_std": float(np.std(losses, ddof=1)),
                     "loss_min": float(np.min(losses)), "loss_max": float(np.max(losses)),
                     "dg_norm_mean": float(np.mean(devs)),
                     "flips_mean": float(np.mean(flips)) if flips else float("nan"),
                     "flips_frac_mean": float(np.mean(flips)) / n_units if flips else float("nan"),
                     "psnr_mean": float(np.mean(psnrs))})
    return rows

def loglog_fit(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = (xs > 0) & (ys > 0)
    if ok.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    lx, ly = np.log(xs[ok]), np.log(ys[ok])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = ((ly - pred) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    return float(slope), float(intercept), float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def effective_rank(s):
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))

def participation_ratio(s):
    e = s ** 2
    return float(e.sum() ** 2 / (e ** 2).sum())

def energy_ranks(s, fractions=(0.5, 0.9, 0.95, 0.99)):
    cum = (s ** 2).cumsum(0) / (s ** 2).sum()
    out = {}
    for f in fractions:
        idx = int(torch.searchsorted(cum, torch.tensor(f, dtype=cum.dtype)).item())
        out[f"r{int(f * 100)}"] = min(idx + 1, len(s))
    return out

def log_bands(s, n_bands):
    ls = torch.log10(s.clamp_min(1e-7 * float(s[0])))
    edges = torch.linspace(float(ls[0]), float(ls[-1]), n_bands + 1, dtype=ls.dtype)
    band = torch.bucketize(-ls, -edges[1:-1], right=True)
    return [(b, (band == b).nonzero().flatten(), float(10 ** edges[b + 1]), float(10 ** edges[b]))
            for b in range(n_bands) if (band == b).any()]

def fd_directional(model, x, y, obs_mode, bn, dirs, step, probe=None):
    cols, flips = [], []
    for d in dirs:
        dv = d.view_as(x) * step
        gp = observe(model, x + dv, y, obs_mode, bn)
        gm = observe(model, x - dv, y, obs_mode, bn)
        cols.append(((gp / gp.norm()) - (gm / gm.norm())) / (2 * step))
        if probe is not None:
            flips.append(float((probe.capture(model, x + dv, bn) != probe.capture(model, x - dv, bn)).sum()))
    return torch.stack(cols, dim=1), (float(np.mean(flips)) if flips else float("nan"))

def load_recon(run_dir: Path, epoch: int, img: int):
    best = run_dir / "best.csv"
    if not best.exists():
        return None
    df = pd.read_csv(best)
    row = df[(df["epoch"] == epoch) & (df["img"] == img)]
    if row.empty:
        return None
    row = row.iloc[0]
    if not Path(row["recon"]).exists():
        return None
    return torch.load(row["recon"], weights_only=True)["recon"], float(row["psnr"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    dev = device()
    out_dir = Path(cfg["out_dir"])
    existing = [f for f in ("spectrum.csv", "scaling.csv", "config.yaml") if (out_dir / f).exists()]
    if existing and not args.resume:
        raise FileExistsError(f"{out_dir} already contains {existing}; pass --resume to continue it")
    if args.resume and (out_dir / "config.yaml").exists():
        saved = load_config(out_dir / "config.yaml")
        if saved != cfg:
            diff = {k: (saved.get(k), cfg.get(k)) for k in set(saved) | set(cfg) if saved.get(k) != cfg.get(k)}
            raise ValueError(f"config differs from the saved run config; refusing to resume. diff={diff}")
    (out_dir / "spectra").mkdir(parents=True, exist_ok=True)
    save_config(out_dir / "config.yaml", cfg)

    done = set()
    if args.resume and (out_dir / "exponent.csv").exists():
        d = pd.read_csv(out_dir / "exponent.csv")
        done = {(int(a), int(b)) for a, b in zip(d["epoch"], d["img"])}

    scal_log = CsvLogger(out_dir / "scaling.csv", SCALING_FIELDS, append=args.resume)
    exp_log = CsvLogger(out_dir / "exponent.csv", EXPONENT_FIELDS, append=args.resume)
    spec_log = CsvLogger(out_dir / "spectrum.csv", SPECTRUM_FIELDS, append=args.resume)
    thr_log = CsvLogger(out_dir / "thresholds.csv", THRESHOLD_FIELDS, append=args.resume)
    proj_log = CsvLogger(out_dir / "projection.csv", PROJECTION_FIELDS, append=args.resume)
    fd_log = CsvLogger(out_dir / "finite_difference.csv", FD_FIELDS, append=args.resume)

    indices = target_indices(cfg["targets"]["n"], cfg["targets"]["offset"])
    sigmas = [float(s) for s in cfg["sigmas"]]
    probe_sigma = float(cfg["probe_sigma"])
    fit_lo, fit_hi = (float(v) for v in cfg["fit_range"])
    jac_images = set(indices[:cfg["jacobian_images"]])
    fd_images = set(indices[:cfg["fd_images"]])
    if cfg["fd_images"] > cfg["jacobian_images"]:
        raise ValueError("fd_images must not exceed jacobian_images")
    missing_runs = set()
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
            if (epoch, img) in done:
                continue
            t0 = time.time()
            x = x_all[i:i + 1].to(dev)
            y = y_all[i:i + 1].to(dev)
            xd = x.double()
            model.double()
            try:
                g_star = observe(model, xd, y, cfg["obs_mode"], cfg["bn"]).detach()

                probe = PreactivationProbe(model) if cfg["count_flips"] else None
                rows = scaling_sweep(model, xd, y, g_star, cfg["obs_mode"], cfg["bn"], norm,
                                     sigmas, cfg["probes"], cfg["seed"] + img, probe)
                if probe:
                    probe.close()

                fro_sq = float("nan")
                s = None
                if img in jac_images:
                    method = cfg.get("jacobian_method", "jvp")
                    fd_rows = []
                    J = (cosine_jacobian_jvp(model, xd, y, cfg["obs_mode"], cfg["bn"]) if method == "jvp"
                         else cosine_jacobian_vjp(model, xd, y, cfg["obs_mode"], cfg["bn"], cfg["chunk"]))
                    s = torch.linalg.svdvals(J).cpu()
                    fro_sq = float((s ** 2).sum())
                    _, _, Vh = torch.linalg.svd(J, full_matrices=False)
                    Vh = Vh.cpu()
                    torch.save({"sigma": s.float(), "Vh": Vh.float() if cfg["save_basis"] else None,
                                "epoch": epoch, "img": img},
                               out_dir / "spectra" / f"epoch{epoch:03d}_img{img:05d}.pt")
                    t_jac = time.time()

                    if img in fd_images:
                        gen = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 7919 + img)
                        dirs = torch.randn(cfg["fd_probes"], x.numel(), generator=gen, dtype=torch.float64).to(dev)
                        dirs = dirs / dirs.norm(dim=1, keepdim=True)
                        Jp = J @ dirs.T
                        s_ad = torch.linalg.svdvals(Jp).cpu()
                        fd_probe = PreactivationProbe(model) if cfg["count_flips"] else None
                        for step in cfg["fd_steps"]:
                            Jfd, fd_flips = fd_directional(model, xd, y, cfg["obs_mode"], cfg["bn"], dirs,
                                                           float(step), fd_probe)
                            s_fd = torch.linalg.svdvals(Jfd).cpu()
                            k = min(len(s_fd), len(s_ad))
                            rel = float((s_fd[:k] - s_ad[:k]).norm() / s_ad[:k].norm())
                            corr = float(np.corrcoef(np.log(s_fd[:k].clamp_min(1e-30).numpy()),
                                                     np.log(s_ad[:k].clamp_min(1e-30).numpy()))[0, 1])
                            fd_rows.append(dict(epoch=epoch, img=img, step=step, dtype=str(Jfd.dtype).split(".")[-1],
                                                probes=cfg["fd_probes"], rel_err=f"{rel:.4e}",
                                                rel_fro=f"{float((Jfd - Jp).norm() / Jp.norm()):.4e}",
                                                ratio_sigma_max=f"{float(s_fd[0] / s_ad[0]):.4e}",
                                                corr_log_sigma=f"{corr:.4f}", flips_mean=f"{fd_flips:.3f}"))
                        if fd_probe is not None:
                            fd_probe.close()

                    ranks = energy_ranks(s)
                    spec_log.log(epoch=epoch, img=img, label=int(y), n=x.numel(), m=J.shape[0],
                                 sigma_max=f"{float(s[0]):.6e}", sigma_min=f"{float(s[-1]):.6e}",
                                 fro_sq=f"{fro_sq:.6e}", nuclear=f"{float(s.sum()):.6e}",
                                 eff_rank=f"{effective_rank(s):.3f}",
                                 part_ratio=f"{participation_ratio(s):.3f}", **ranks,
                                 grad_norm=f"{float(g_star.norm()):.6e}",
                                 seconds=f"{t_jac - t0:.1f}")
                    for thr in cfg["thresholds"]:
                        vis = int((s / s[0] > float(thr)).sum())
                        thr_log.log(epoch=epoch, img=img, threshold=thr, visible=vis,
                                    near_null=len(s) - vis,
                                    near_null_pct=f"{100.0 * (len(s) - vis) / len(s):.3f}")

                    gen = torch.Generator(device="cpu").manual_seed(cfg["seed"] + img)
                    bands = log_bands(s, cfg["bands"])
                    for run in cfg.get("projection_runs", []):
                        got = load_recon(Path("runs/attack") / run, epoch, img)
                        if got is None:
                            missing_runs.add(run)
                            continue
                        recon, psnr_val = got
                        delta = (norm.normalize(recon.to(dev)) - x).reshape(-1).cpu().double()
                        rnd = torch.randn(delta.numel(), generator=gen).double()
                        c = (Vh @ delta) ** 2
                        cr = (Vh @ (rnd / rnd.norm() * delta.norm())) ** 2
                        for b, idx, lo, hi in bands:
                            proj_log.log(epoch=epoch, img=img, run=run, psnr=f"{psnr_val:.3f}", band=b,
                                         sigma_lo=f"{lo:.3e}", sigma_hi=f"{hi:.3e}", dims=idx.numel(),
                                         energy_frac=f"{float(c[idx].sum() / c.sum()):.6f}",
                                         energy_frac_random=f"{float(cr[idx].sum() / cr.sum()):.6f}")
                    for row in fd_rows:
                        fd_log.log(**row)
                    del J, Vh
                    torch.cuda.empty_cache()
            finally:
                model.float()

            for r in rows:
                pred = 0.5 * r["sigma"] ** 2 * fro_sq
                scal_log.log(epoch=epoch, img=img, **{k: (f"{v:.6e}" if isinstance(v, float) else v)
                                                      for k, v in r.items()},
                             pred_quadratic=f"{pred:.6e}" if np.isfinite(pred) else "")
            fit = [r for r in rows if fit_lo <= r["sigma"] <= fit_hi]
            fit_sigmas = [r["sigma"] for r in fit]
            sl, ic, r2 = loglog_fit(fit_sigmas, [r["loss_mean"] for r in fit])
            sd, _, r2d = loglog_fit(fit_sigmas, [r["dg_norm_mean"] for r in fit])
            sf, _, r2f = loglog_fit(fit_sigmas, [r["flips_mean"] for r in fit])
            at_probe = next((r for r in rows if abs(r["sigma"] - probe_sigma) < 1e-12), None)
            ratio = (0.5 * probe_sigma ** 2 * fro_sq / at_probe["loss_mean"]
                     if at_probe and np.isfinite(fro_sq) and at_probe["loss_mean"] > 0 else float("nan"))
            exp_log.log(epoch=epoch, img=img, slope_loss=f"{sl:.4f}", intercept_loss=f"{ic:.4f}",
                        r2_loss=f"{r2:.4f}", slope_dg=f"{sd:.4f}", r2_dg=f"{r2d:.4f}",
                        slope_flips=f"{sf:.4f}", r2_flips=f"{r2f:.4f}",
                        sigma_lo=min(fit_sigmas, default=float("nan")),
                        sigma_hi=max(fit_sigmas, default=float("nan")),
                        quad_ratio_at_probe=f"{ratio:.4e}")

            print(f"  epoch {epoch:3d} img {img:5d}  loss~sigma^{sl:.2f} (R2={r2:.3f}) "
                  f"dg~sigma^{sd:.2f} flips~sigma^{sf:.2f}"
                  + (f"  sigma_max={float(s[0]):.2e} eff_rank={effective_rank(s):.1f} "
                     f"quad/meas={ratio:.2e}" if s is not None else "")
                  + f"  ({time.time() - t0:.0f}s, total {fmt_seconds(time.time() - t_start)})")

        del model
        torch.cuda.empty_cache()

    if missing_runs:
        print(f"WARNING: no reconstructions found for {sorted(missing_runs)}; projection rows skipped")
    for log in (scal_log, exp_log, spec_log, thr_log, proj_log, fd_log):
        log.close()
    print(f"done in {fmt_seconds(time.time() - t_start)} -> {out_dir}")

if __name__ == "__main__":
    main()