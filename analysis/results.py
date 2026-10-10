from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

RUNS = Path("runs/attack/main")
BASELINES = Path("runs/baselines.csv")
CALIBRATION = Path("runs/calibration.csv")
GRAD_NORMS = Path("runs/grad_norms.csv")
JACOBIAN = Path("runs/jacobian/resnet20_4_r16")
EPOCHS = [0, 1, 5, 10, 30]
LADDER = ["lorafa", "lorafa_fc", "lora", "full"]
SATURATION_NORM = 1e-5
RANK_IMAGES = list(range(50))
BASE_METRICS = ("lpips", "lpips_null_frac_better", "rank1")
MODE_LABEL = {"lorafa": "grad B", "lorafa_fc": "grad B + head", "lora": "grad A + grad B + head", "full": "full gradient"}
CONCAT_RUNS = {"lorafa": "pixel", "lorafa_fc": "obs_lorafa_fc", "lora": "obs_lora", "full": "obs_full"}
BLOCK_RUNS = {"lorafa": "cos_pixel", "lorafa_fc": "blocks_lorafa_fc", "lora": "blocks_lora", "full": "blocks_full"}
CONTROL_LABEL = {"gray": "constant gray", "mean_color": "per-image mean color",
                 "dataset_mean": "dataset mean image", "same_label_image": "other test image, same label"}
RANK_RUNS = {"rank_r4": 4, "rank_r8": 8, "pixel": 16, "rank_r32": 32}
GT_INIT_RUNS = ["init_gt_lr_1e-4", "init_gt_lr_1e-3", "init_gt_noise_000"]
ABLATION_LABEL = {"pixel": "pixel, grad B", "dip": "DIP, grad B", "geiping": "sign-Adam, TV 1e-4",
                  "firstconv": "first conv adapted", "bn_train": "BatchNorm in train mode",
                  "backbone_random": "random backbone", "cos_pixel": "pixel, eps-free cosine",
                  "cos_dip": "DIP, eps-free cosine", "resnet18_pixel": "ResNet-18, pixel",
                  "resnet18_dip": "ResNet-18, DIP", "resnet18_obs_full": "ResNet-18, full gradient"}

def show(table: pd.DataFrame, name: str) -> None:
    print(f"\n== {name}")
    print(table.to_string(index=False))

def checkpoint_accuracy(path: str) -> dict:
    meta = torch.load(path, map_location="cpu", weights_only=True)["meta"]
    return {"epoch": int(meta["epoch"]), "test_acc": meta["test_acc"]}

def load_run(run_dir: Path) -> tuple:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    restarts = pd.read_csv(run_dir / "restarts.csv").drop_duplicates(["epoch", "img", "restart"], keep="last")
    best = pd.read_csv(run_dir / "best.csv").drop_duplicates(["epoch", "img"], keep="last")
    accuracy = pd.DataFrame([checkpoint_accuracy(c) for c in config["checkpoints"]])
    restarts = restarts.merge(best[["epoch", "img", "gt_loss"]], on=["epoch", "img"]).merge(accuracy, on="epoch")
    identification = pd.read_csv(run_dir / "identification.csv")
    return restarts.assign(run=run_dir.name, lr=config["lr"]), identification.assign(run=run_dir.name)

def load_best() -> pd.DataFrame:
    restarts, identification = zip(*[load_run(d) for d in sorted(RUNS.iterdir()) if (d / "restarts.csv").exists()])
    restarts = pd.concat(restarts, ignore_index=True)
    selected = restarts.loc[restarts.groupby(["run", "epoch", "img"])["final_loss"].idxmin()]
    selected = selected[np.isfinite(selected.final_cos)]
    best = selected.merge(pd.concat(identification), on=["run", "epoch", "img"])
    best = best.merge(pd.read_csv(BASELINES), on=["img", "label"])
    best["rank1"] = (best.lpips_null_frac_better == 0).astype(float)
    best["best_trivial_psnr"] = best[["gray_psnr", "mean_color_psnr", "dataset_mean_psnr"]].max(axis=1)
    return best

def load_grad_norms() -> pd.DataFrame:
    grads = pd.read_csv(GRAD_NORMS)
    return grads[(grads.bn == "eval") & grads.checkpoint.str.contains("resnet20_4_r16/")]

def saturated_images(grads: pd.DataFrame) -> dict:
    below = grads[(grads.obs_mode == "lorafa") & (grads.grad_norm < SATURATION_NORM)]
    return {epoch: sorted(below[below.epoch == epoch].img) for epoch in EPOCHS}

def block_norm_ratios(grads: pd.DataFrame) -> pd.DataFrame:
    norms = grads.pivot_table(index=["epoch", "img"], columns="obs_mode", values="grad_norm")
    head = np.sqrt((norms.lorafa_fc ** 2 - norms.lorafa ** 2).clip(lower=0))
    a = np.sqrt((norms.lora ** 2 - norms.lorafa_fc ** 2).clip(lower=0))
    return pd.DataFrame({"head": head / norms.lorafa, "A": a / norms.lorafa}).groupby("epoch").median()

def a_gradient_check(grads: pd.DataFrame) -> pd.DataFrame:
    lora = grads[grads.obs_mode == "lora"].set_index(["epoch", "img"])
    lorafa_fc = grads[grads.obs_mode == "lorafa_fc"].set_index(["epoch", "img"]).reindex(lora.index)
    a_is_zero = (lora.n_zero - lorafa_fc.n_zero) == (lora.m - lorafa_fc.m)
    norm_gap = (lora.grad_norm - lorafa_fc.grad_norm).abs() / lorafa_fc.grad_norm
    return pd.DataFrame({"zero": a_is_zero, "gap": norm_gap}).groupby("epoch").agg(zero=("zero", "sum"), gap=("gap", "max"))

def cell(best: pd.DataFrame, run: str, epoch: int, images=None) -> pd.DataFrame:
    rows = best[(best.run == run) & (best.epoch == epoch)]
    return rows if images is None else rows[rows.img.isin(images)]

def summarize(rows: pd.DataFrame) -> dict:
    return {"n": len(rows), "cos": rows.final_cos.mean(), "PSNR": rows.psnr.mean(), "LPIPS": rows.lpips.mean(),
            "fb": rows.lpips_null_frac_better.mean(), "rank1": rows.rank1.mean()}

def ladder_run(cosine: str, mode: str, epoch: int) -> str:
    if epoch == 0 and mode == "lora":
        mode = "lorafa_fc"
    if cosine == "blocks" and mode == "lorafa" and epoch == 0:
        return "pixel"
    return (BLOCK_RUNS if cosine == "blocks" else CONCAT_RUNS)[mode]

def splice_cells(param: str) -> list:
    return [(param, 0), (f"splice_fc1_b0_{param}", 0), (f"splice_fc0_b1_{param}", 1), (param, 1)]

def wilcoxon_p(a: pd.Series, b: pd.Series) -> float:
    difference = (b - a).to_numpy()
    return stats.wilcoxon(difference).pvalue if np.any(difference != 0) else 1.0

def mcnemar_p(a: pd.Series, b: pd.Series) -> float:
    gained = int(((a == 0) & (b == 1)).sum())
    lost = int(((a == 1) & (b == 0)).sum())
    return stats.binomtest(min(gained, lost), gained + lost, 0.5).pvalue if gained + lost else 1.0

def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    adjusted = (p.loc[order] * np.arange(len(p), 0, -1)).cummax().clip(upper=1.0)
    return adjusted.reindex(p.index)

def comparison_family(best: pd.DataFrame) -> list:
    head_images = cell(best, "blocks_lorafa_fc", 30).img.tolist()
    family = [("R2", "pixel vs DIP", ("pixel", e), ("dip", e), None, ("final_cos",) + BASE_METRICS) for e in EPOCHS]
    family += [("K2 LoRA", "LoRA-FA vs LoRA, concatenated", ("obs_lorafa_fc", 30), ("obs_lora", 30), None, BASE_METRICS),
               ("K2 LoRA", "LoRA-FA vs LoRA, per-block", ("blocks_lorafa_fc", 30), ("blocks_lora", 30), None, BASE_METRICS),
               ("K2 head", "adding head, concatenated", ("pixel", 30), ("obs_lorafa_fc", 30), head_images, BASE_METRICS),
               ("K2 head", "concatenated to per-block", ("obs_lorafa_fc", 30), ("blocks_lorafa_fc", 30), head_images,
                BASE_METRICS)]
    ranks = [("rank_r4", "rank_r8"), ("rank_r8", "pixel"), ("pixel", "rank_r32"), ("rank_r4", "rank_r32")]
    family += [("K2 rank", f"r{RANK_RUNS[a]} vs r{RANK_RUNS[b]}", (a, e), (b, e), RANK_IMAGES, BASE_METRICS)
               for e in (0, 30) for a, b in ranks]
    for param in ("dip", "pixel"):
        cells = splice_cells(param)
        family += [("R3", f"splice {param}", cells[i], cells[j], None, BASE_METRICS) for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]]
    return family

def run_comparisons(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, label, a, b, images, metrics in comparison_family(best):
        rows_a = cell(best, *a, images).set_index("img")
        rows_b = cell(best, *b, images).set_index("img")
        shared = rows_a.index.intersection(rows_b.index)
        for metric in metrics:
            x, y = rows_a.loc[shared, metric], rows_b.loc[shared, metric]
            rows.append({"family": family, "comparison": label, "a": f"{a[0]} ep{a[1]}", "b": f"{b[0]} ep{b[1]}",
                         "n": len(shared), "metric": metric, "a mean": x.mean(), "b mean": y.mean(),
                         "p": mcnemar_p(x, y) if metric == "rank1" else wilcoxon_p(x, y)})
    table = pd.DataFrame(rows)
    table["p Holm"] = table.groupby("family").p.transform(holm)
    return table

def table_protocol(best: pd.DataFrame, grads: pd.DataFrame) -> None:
    norms = grads.groupby(["obs_mode", "epoch"]).grad_norm.median().unstack()
    rows = [{"observation": MODE_LABEL[mode], "m": int(grads[grads.obs_mode == mode].m.iloc[0]),
             **{f"ep {e}": norms.loc[mode, e] for e in EPOCHS}} for mode in LADDER]
    ratios = block_norm_ratios(grads)
    rows.append({"observation": "median |grad head| / |grad B|", **{f"ep {e}": ratios.loc[e, "head"] for e in EPOCHS}})
    rows.append({"observation": "median |grad A| / |grad B|", **{f"ep {e}": ratios.loc[e, "A"] for e in EPOCHS}})
    check = a_gradient_check(grads)
    rows.append({"observation": "images with grad A == 0 (of 100)", **{f"ep {e}": check.loc[e, "zero"] for e in EPOCHS}})
    rows.append({"observation": "max relative norm gap, LoRA vs grad B + head",
                 **{f"ep {e}": check.loc[e, "gap"] for e in EPOCHS}})
    accuracy = best[best.run == "pixel"].drop_duplicates("epoch").set_index("epoch").test_acc
    rows.append({"observation": "test accuracy (%)", **{f"ep {e}": accuracy[e] for e in EPOCHS}})
    show(pd.DataFrame(rows), "protocol")

def table_calibration() -> None:
    calibration = pd.read_csv(CALIBRATION)
    calibration = calibration.merge(pd.read_csv(BASELINES)[["img", "pixel_var"]], on="img")
    rows = []
    for control, label in CONTROL_LABEL.items():
        chunk = calibration[calibration.control == control]
        rows.append({"control": label, "PSNR": chunk.psnr.mean(), "LPIPS": chunk.lpips.mean(),
                     "rho(var, PSNR)": stats.spearmanr(chunk.pixel_var, chunk.psnr).statistic,
                     "fb": chunk.lpips_null_frac_better.mean(),
                     "rank1": (chunk.lpips_null_frac_better == 0).mean()})
    show(pd.DataFrame(rows), "calibration")

def table_main(best: pd.DataFrame) -> None:
    rows = []
    for param in ("pixel", "dip"):
        for epoch in EPOCHS:
            chunk = cell(best, param, epoch)
            rows.append({"attack": param, "epoch": epoch, **summarize(chunk),
                         "rho(var, PSNR)": stats.spearmanr(chunk.pixel_var, chunk.psnr).statistic,
                         "beats trivial PSNR": (chunk.psnr > chunk.best_trivial_psnr).mean()})
    show(pd.DataFrame(rows), "main")

def table_ladder(best: pd.DataFrame) -> None:
    rows = []
    for epoch in (0, 30):
        for mode in LADDER:
            row = {"epoch": epoch, "observation": MODE_LABEL[mode]}
            for cosine in ("concatenated", "blocks"):
                chunk = cell(best, ladder_run(cosine, mode, epoch), epoch)
                row.update({f"{cosine} n": len(chunk), f"{cosine} fb": chunk.lpips_null_frac_better.mean(),
                            f"{cosine} rank1": chunk.rank1.mean()})
            rows.append(row)
    show(pd.DataFrame(rows), "ladder")

def table_lora_vs_lorafa(comparisons: pd.DataFrame) -> None:
    show(comparisons[comparisons.family.isin(["K2 LoRA", "K2 head"])], "lora_vs_lorafa")

def table_rank(best: pd.DataFrame) -> None:
    rows = [{"epoch": epoch, "r": r, **summarize(cell(best, run, epoch, RANK_IMAGES))}
            for epoch in (0, 30) for run, r in RANK_RUNS.items()]
    show(pd.DataFrame(rows), "rank")

def table_ground_truth_init(best: pd.DataFrame) -> None:
    curves = pd.concat([pd.read_csv(RUNS / run / "curves.csv").assign(run=run) for run in GT_INIT_RUNS])
    curves = curves.drop_duplicates(["run", "epoch", "img", "restart", "iter"], keep="last")
    early = curves[curves.iter == 100].set_index(["run", "epoch", "img"]).cos
    late = curves[curves.iter == curves.iter.max()].set_index(["run", "epoch", "img"]).cos
    drifted = (late > early.reindex(late.index)).groupby(["run", "epoch"]).mean()
    rows = []
    for epoch in (0, 30):
        previous_rms = np.nan
        for run in GT_INIT_RUNS:
            chunk = cell(best, run, epoch)
            rms = (10 ** (-chunk.psnr / 20)).mean()
            rows.append({"epoch": epoch, "lr": chunk.lr.iloc[0], "n": len(chunk),
                         "cos": chunk.final_cos.mean(), "TV": chunk.final_tv.mean(), "PSNR": chunk.psnr.mean(),
                         "RMS": rms, "RMS ratio": rms / previous_rms, "rank1": chunk.rank1.mean(),
                         "cos rose": drifted[(run, epoch)]})
            previous_rms = rms
    show(pd.DataFrame(rows), "gt_init")

def table_objective(best: pd.DataFrame, saturated: dict) -> None:
    rows = []
    for epoch in (0, 30):
        random_init = cell(best, "pixel", epoch).set_index("img").final_loss
        for run in GT_INIT_RUNS:
            gt_init = cell(best, run, epoch).set_index("img")
            ratio = random_init / gt_init.final_loss.reindex(random_init.index)
            rows.append({"epoch": epoch, "lr": gt_init.lr.iloc[0], "random init median": random_init.median(),
                         "target init median": gt_init.final_loss.median(), "median ratio": ratio.median(),
                         "target init lower": (ratio > 1).mean()})
    show(pd.DataFrame(rows), "objective")
    below = []
    for epoch in (0, 30):
        for param in ("pixel", "dip"):
            chunk = cell(best, param, epoch)
            kept = chunk[~chunk.img.isin(saturated[epoch])]
            below.append({"epoch": epoch, "attack": param, "n": len(chunk),
                          "below target": int((chunk.final_loss < chunk.gt_loss).sum()),
                          "n unsaturated": len(kept), "below target, unsaturated": int((kept.final_loss < kept.gt_loss).sum())})
    show(pd.DataFrame(below), "below_target")

def table_scaling() -> None:
    scaling = pd.read_csv(JACOBIAN / "scaling.csv")
    scaling["ratio"] = scaling.loss_mean / scaling.pred_quadratic
    table = scaling.groupby(["epoch", "sigma"]).agg(loss=("loss_mean", "mean"), quadratic=("pred_quadratic", "mean"),
                                                    ratio_mean=("ratio", "mean"), ratio_median=("ratio", "median"),
                                                    loss_min=("loss_min", "min"), dg=("dg_norm_mean", "mean"),
                                                    flips=("flips_mean", "mean"),
                                                    n_quadratic=("pred_quadratic", "count")).reset_index()
    show(table, "scaling")
    exponent = pd.read_csv(JACOBIAN / "exponent.csv").groupby("epoch").agg(
        loss_slope=("slope_loss", "mean"), loss_slope_min=("slope_loss", "min"), loss_slope_max=("slope_loss", "max"),
        r2_min=("r2_loss", "min"), dg_slope=("slope_dg", "mean"), flips_slope=("slope_flips", "mean")).reset_index()
    show(exponent, "exponents")

def table_quadratic_check() -> None:
    scaling = pd.read_csv(JACOBIAN / "scaling.csv")
    scaling["flips_per_sigma"] = scaling.flips_mean / scaling.sigma
    exact = scaling[scaling.pred_quadratic.notna()].copy()
    exact["ratio_mean_probe"] = exact.loss_mean / exact.pred_quadratic
    exact["ratio_lowest_probe"] = exact.loss_min / exact.pred_quadratic
    table = exact.groupby(["epoch", "sigma"]).agg(n=("img", "size"), ratio_mean_probe=("ratio_mean_probe", "median"),
                                                  ratio_lowest_probe=("ratio_lowest_probe", "median"),
                                                  flips=("flips_mean", "mean")).reset_index()
    rate = scaling.groupby(["epoch", "sigma"]).flips_per_sigma.median().rename("flips_per_sigma_all_images").reset_index()
    show(table.merge(rate, on=["epoch", "sigma"]), "quadratic_check")

def table_spectrum() -> None:
    spectrum = pd.read_csv(JACOBIAN / "spectrum.csv")
    spectrum["condition"] = spectrum.sigma_max / spectrum.sigma_min
    table = spectrum.groupby("epoch").agg(n=("img", "size"), condition=("condition", "median"),
                                          condition_min=("condition", "min"), condition_max=("condition", "max"),
                                          eff_rank=("eff_rank", "mean"), eff_rank_min=("eff_rank", "min"),
                                          eff_rank_max=("eff_rank", "max"), part_ratio_min=("part_ratio", "min"),
                                          part_ratio_max=("part_ratio", "max"),
                                          r50_at_most_5=("r50", lambda v: int((v <= 5).sum()))).reset_index()
    show(table, "spectrum")
    thresholds = pd.read_csv(JACOBIAN / "thresholds.csv").groupby(["epoch", "threshold"]).near_null_pct.agg(["mean", "max"])
    show(thresholds.reset_index(), "thresholds")

def table_finite_difference() -> None:
    fd = pd.read_csv(JACOBIAN / "finite_difference.csv")
    fd["clean"] = fd.flips_mean == 0
    table = fd.groupby(["epoch", "step"]).apply(lambda g: pd.Series({
        "n": len(g), "no flip": int(g.clean.sum()), "median error": g.rel_err.median(),
        "median error, no flip": g[g.clean].rel_err.median(), "max error, no flip": g[g.clean].rel_err.max(),
        "min error, flip": g[~g.clean].rel_err.min(), "max error": g.rel_err.max()}),
        include_groups=False).reset_index()
    show(table, "finite_difference")

def table_projection() -> None:
    projection = pd.read_csv(JACOBIAN / "projection.csv")
    projection["vs_random"] = projection.energy_frac / projection.energy_frac_random
    profile = projection.pivot_table(index=["epoch", "band"], columns="run", values="vs_random", aggfunc="median")
    profile.columns = [f"{c.split('/')[-1]} vs random" for c in profile.columns]
    shares = projection.pivot_table(index=["epoch", "img", "band"], columns="run", values="energy_frac")
    shares["ratio"] = shares["main/dip"] / shares["main/pixel"]
    shares = shares[(shares.ratio > 0) & np.isfinite(shares.ratio)]
    rows = []
    for (epoch, band), chunk in shares.groupby(level=["epoch", "band"]):
        rows.append({"epoch": epoch, "band": band, "n": len(chunk), "median DIP/pixel share": chunk.ratio.median(),
                     "DIP lower": int((chunk.ratio < 1).sum()), "p": stats.wilcoxon(np.log(chunk.ratio)).pvalue})
    tests = pd.DataFrame(rows)
    tests["p Holm"] = holm(tests.p)
    show(profile.reset_index().merge(tests, on=["epoch", "band"]), "projection")

def table_splice(best: pd.DataFrame) -> None:
    names = ["random head, B=0", "trained head, B=0", "random head, trained B", "trained head, trained B"]
    rows = [{"attack": param, "checkpoint": name, "test acc": cell(best, *c).test_acc.iloc[0], **summarize(cell(best, *c))}
            for param in ("dip", "pixel") for name, c in zip(names, splice_cells(param))]
    show(pd.DataFrame(rows), "splice")

def table_ablations(best: pd.DataFrame) -> None:
    rows = [{"setting": label, "epoch": epoch, **summarize(cell(best, run, epoch)),
             "beats trivial PSNR": (cell(best, run, epoch).psnr > cell(best, run, epoch).best_trivial_psnr).mean()}
            for run, label in ABLATION_LABEL.items() for epoch in sorted(best[best.run == run].epoch.unique())]
    show(pd.DataFrame(rows), "ablations")

def table_gradient_norm(best: pd.DataFrame, grads: pd.DataFrame) -> None:
    norms = grads[grads.obs_mode == "lorafa"][["epoch", "img", "grad_norm"]]
    joined = best[best.run.isin(["pixel", "dip"])].merge(norms, on=["epoch", "img"])
    rows = []
    for (run, epoch), chunk in joined.groupby(["run", "epoch"]):
        rho = stats.spearmanr(chunk.grad_norm, chunk.lpips)
        rows.append({"attack": run, "epoch": epoch, "median grad B norm": chunk.grad_norm.median(),
                     "rho(norm, LPIPS)": rho.statistic, "p": rho.pvalue})
    show(pd.DataFrame(rows), "gradient_norm")
    late = joined[joined.epoch == 30].copy()
    late["bin"] = pd.cut(late.grad_norm, [0, 1e-4, 1e-2, np.inf], labels=["<1e-4", "1e-4 to 1e-2", ">1e-2"])
    bins = late.groupby(["run", "bin"], observed=True).agg(n=("img", "size"), rank1=("rank1", "mean"),
                                                           fb=("lpips_null_frac_better", "mean")).reset_index()
    show(bins, "gradient_bins")

def table_sensitivity(best: pd.DataFrame, saturated: dict) -> None:
    cases = [(run, epoch) for epoch in (5, 10, 30) for run in ("pixel", "dip")]
    cases += [(run, 30) for run in ("obs_lorafa_fc", "obs_lora", "obs_full", "cos_pixel", "cos_dip",
                                    "blocks_lorafa_fc", "blocks_lora", "blocks_full")]
    rows = []
    for run, epoch in cases:
        chunk = cell(best, run, epoch)
        kept = chunk[~chunk.img.isin(saturated[epoch])]
        rows.append({"run": run, "epoch": epoch, "n": len(chunk), "fb": chunk.lpips_null_frac_better.mean(),
                     "rank1": chunk.rank1.mean(), "n kept": len(kept),
                     "fb kept": kept.lpips_null_frac_better.mean(), "rank1 kept": kept.rank1.mean()})
    show(pd.DataFrame(rows), "sensitivity")

def main():
    best = load_best()
    grads = load_grad_norms()
    saturated = saturated_images(grads)
    comparisons = run_comparisons(best)
    table_protocol(best, grads)
    table_calibration()
    table_main(best)
    table_ladder(best)
    table_lora_vs_lorafa(comparisons)
    table_rank(best)
    table_ground_truth_init(best)
    table_objective(best, saturated)
    table_scaling()
    table_quadratic_check()
    table_spectrum()
    table_finite_difference()
    table_projection()
    table_splice(best)
    table_ablations(best)
    show(comparisons, "significance")
    table_gradient_norm(best, grads)
    table_sensitivity(best, saturated)

if __name__ == "__main__":
    main()
