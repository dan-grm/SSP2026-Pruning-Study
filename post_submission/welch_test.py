#!/usr/bin/env python3
"""Per-ratio Welch tests of pruned+fine-tuned vs. the ``tuned_0`` control, and the headline X.

For every pruning method, model and ratio, the 10 pruned+fine-tuned runs from
the paper's CSV are compared with the 10 ``tuned_0`` control runs, using a
one-sided Welch t-test (H1: the pruned model is worse, i.e. has a higher RMSE
or S-Score), alpha = 0.05, with no multiple-comparison correction.

X (per model, structured pruning only) is the largest share of MACs removed
at which neither RMSE nor S-Score is significantly worse. If that holds for
RMSE only, the RMSE-only X is reported and the slide says "no RMSE loss".

Fallback (no control CSV): fixed tolerance against the raw checkpoint,
mean ΔRMSE <= 0.5 and mean ΔS <= 10 %.

Both "largest ratio that passes" and "largest ratio such that every smaller
ratio passes too" are reported. The plan uses the first; a warning is printed
when they differ.

Outputs in results/post_submission/:
    welch_tests.csv        one row per method x model x ratio x metric
    headline.json          X per model, both criteria, plus the 90 % numbers
    headline_macros.tex    \\newcommand macros for the slides

Usage (from the repo root):
    python post_submission/welch_test.py
    python post_submission/welch_test.py --control results/post_submission/tuned_control.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "post_submission"
PAPER_CSV = {
    "structured": REPO / "results" / "experiment_logs" / "structured_pruning_data_3.csv",
    "unstructured": REPO / "results" / "experiment_logs" / "unstructured_pruning_data_3.csv",
}
MODELS = ["big", "small"]
METRICS = ["rmse", "score"]
RATIOS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
NUM_RUNS = 10

TOL_RMSE = 0.5     # absolute, RUL cycles
TOL_SCORE = 0.10   # relative


def load(path: Path) -> dict[str, dict]:
    with path.open() as f:
        return {r["metric"]: r for r in csv.DictReader(f)}


def runs(table: dict, name: str) -> np.ndarray:
    row = table[name]
    v = np.array([float(row[f"run_{i + 1}"]) if row.get(f"run_{i + 1}") else math.nan
                  for i in range(NUM_RUNS)])
    return v[~np.isnan(v)]


def avg(table: dict, name: str) -> float:
    return float(table[name]["avg"])


def pick_x(entries: list[dict], key: str) -> tuple[dict | None, dict | None]:
    """(largest passing ratio, largest ratio with all smaller ones passing)."""
    passing = [e for e in entries if e[key]]
    largest = max(passing, key=lambda e: e["ratio"]) if passing else None
    contiguous = None
    for e in sorted(entries, key=lambda e: e["ratio"]):
        if not e[key]:
            break
        contiguous = e
    return largest, contiguous


def fmt_pct(x: float | None) -> str:
    return "--" if x is None else f"{x:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--control", type=Path, default=RESULTS_DIR / "tuned_control.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--outdir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    paper = {method: load(path) for method, path in PAPER_CSV.items()}
    control = load(args.control) if args.control.exists() else None
    if control is None:
        print(f"[welch] no control CSV at {args.control}: only the fixed-tolerance fallback is computed")
    else:
        n = {s: len(runs(control, f"{s}_rmse_tuned_0")) for s in MODELS}
        print(f"[welch] control runs: {n}")
        if min(n.values()) < 2:
            print("[welch] fewer than 2 control runs for a model; Welch tests are not possible")
            control = None

    rows = []
    per_model: dict[tuple[str, str], list[dict]] = {}
    for method, table in paper.items():
        for size in MODELS:
            macs0 = avg(table, f"{size}_MACs_pruned_0")
            raw = {m: avg(table, f"{size}_{m}_pruned_0") for m in METRICS}
            entries = []
            for ratio in RATIOS:
                macs_removed = 100 * (1 - avg(table, f"{size}_MACs_tuned_{ratio}") / macs0)
                entry = {"ratio": ratio, "macs_removed_pct": macs_removed}
                for metric in METRICS:
                    tuned = runs(table, f"{size}_{metric}_tuned_{ratio}")
                    row = {
                        "method": method, "model": size, "ratio": ratio,
                        "macs_removed_pct": round(macs_removed, 3), "metric": metric,
                        "n_tuned": len(tuned), "tuned_mean": tuned.mean(), "tuned_std": tuned.std(ddof=1),
                        "raw_checkpoint": raw[metric],
                        "delta_vs_raw": tuned.mean() - raw[metric],
                        "delta_vs_raw_pct": 100 * (tuned.mean() - raw[metric]) / raw[metric],
                    }
                    tol_ok = (row["delta_vs_raw"] <= TOL_RMSE) if metric == "rmse" \
                        else (row["delta_vs_raw_pct"] <= 100 * TOL_SCORE)
                    row["fallback_ok"] = bool(tol_ok)
                    if control is not None:
                        ctrl = runs(control, f"{size}_{metric}_tuned_0")
                        t, p = stats.ttest_ind(tuned, ctrl, equal_var=False, alternative="greater")
                        row.update(n_control=len(ctrl), control_mean=ctrl.mean(), control_std=ctrl.std(ddof=1),
                                   delta_vs_control=tuned.mean() - ctrl.mean(),
                                   delta_vs_control_pct=100 * (tuned.mean() - ctrl.mean()) / ctrl.mean(),
                                   welch_t=float(t), welch_p=float(p),
                                   significantly_worse=bool(p < args.alpha))
                    rows.append(row)
                    entry[f"{metric}_fallback_ok"] = row["fallback_ok"]
                    if control is not None:
                        entry[f"{metric}_welch_ok"] = not row["significantly_worse"]
                    entry[f"{metric}_row"] = row
                entry["fallback_both"] = entry["rmse_fallback_ok"] and entry["score_fallback_ok"]
                entry["fallback_rmse"] = entry["rmse_fallback_ok"]
                if control is not None:
                    entry["welch_both"] = entry["rmse_welch_ok"] and entry["score_welch_ok"]
                    entry["welch_rmse"] = entry["rmse_welch_ok"]
                entries.append(entry)
            per_model[(method, size)] = entries

    # ---------------------------------------------------------------- outputs
    args.outdir.mkdir(parents=True, exist_ok=True)
    columns = ["method", "model", "ratio", "macs_removed_pct", "metric", "n_tuned", "tuned_mean", "tuned_std",
               "raw_checkpoint", "delta_vs_raw", "delta_vs_raw_pct", "fallback_ok",
               "n_control", "control_mean", "control_std", "delta_vs_control", "delta_vs_control_pct",
               "welch_t", "welch_p", "significantly_worse"]
    with (args.outdir / "welch_tests.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    criterion = "welch" if control is not None else "fallback"
    headline: dict = {"criterion": criterion, "alpha": args.alpha,
                      "tolerance": {"rmse_abs": TOL_RMSE, "score_rel": TOL_SCORE}, "models": {}}
    for size in MODELS:
        entries = per_model[("structured", size)]
        model_out: dict = {}
        for crit in (["welch"] if control is not None else []) + ["fallback"]:
            for scope in ("both", "rmse"):
                largest, contiguous = pick_x(entries, f"{crit}_{scope}")
                model_out[f"{crit}_{scope}"] = {
                    "x_macs_removed_pct": largest and largest["macs_removed_pct"],
                    "x_ratio": largest and largest["ratio"],
                    "contiguous_x_macs_removed_pct": contiguous and contiguous["macs_removed_pct"],
                    "contiguous_x_ratio": contiguous and contiguous["ratio"],
                }
                if (largest or {}).get("ratio") != (contiguous or {}).get("ratio"):
                    print(f"[welch] NOTE {size} {crit}/{scope}: largest passing ratio "
                          f"{largest and largest['ratio']} != contiguous {contiguous and contiguous['ratio']}")
        both = model_out[f"{criterion}_both"]["x_macs_removed_pct"]
        rmse_only = model_out[f"{criterion}_rmse"]["x_macs_removed_pct"]
        if both is not None:
            claim, x = "no measurable loss", both
        elif rmse_only is not None:
            claim, x = "no RMSE loss", rmse_only
        else:
            claim, x = None, None
        e90 = next(e for e in entries if e["ratio"] == 90)
        model_out["headline"] = {"x_macs_removed_pct": x, "claim": claim}
        model_out["at_90"] = {
            "macs_removed_pct": e90["macs_removed_pct"],
            "delta_rmse_vs_raw": e90["rmse_row"]["delta_vs_raw"],
            "delta_score_vs_raw_pct": e90["score_row"]["delta_vs_raw_pct"],
            "delta_rmse_vs_control": e90["rmse_row"].get("delta_vs_control"),
            "delta_score_vs_control_pct": e90["score_row"].get("delta_vs_control_pct"),
        }
        if control is not None:
            # Fine-tuning penalty of the unpruned model (backup slide): control vs raw checkpoint.
            penalty = {}
            for metric in METRICS:
                ctrl = runs(control, f"{size}_{metric}_tuned_0")
                raw = avg(paper["structured"], f"{size}_{metric}_pruned_0")
                t, p = stats.ttest_1samp(ctrl, raw)
                penalty[metric] = {"raw": raw, "control_mean": float(ctrl.mean()),
                                   "delta": float(ctrl.mean() - raw),
                                   "delta_pct": float(100 * (ctrl.mean() - raw) / raw),
                                   "p_two_sided": float(p)}
            model_out["fine_tuning_penalty"] = penalty
        headline["models"][size] = model_out

    (args.outdir / "headline.json").write_text(json.dumps(headline, indent=2))

    macros = [f"% generated by post_submission/welch_test.py (criterion: {criterion})"]
    for size, name in (("big", "Big"), ("small", "Small")):
        h = headline["models"][size]
        macros.append(f"\\newcommand{{\\X{name}}}{{{fmt_pct(h['headline']['x_macs_removed_pct'])}}}")
        macros.append(f"\\newcommand{{\\X{name}Claim}}{{{h['headline']['claim'] or 'none'}}}")
        a = h["at_90"]
        macros.append(f"\\newcommand{{\\{name}NinetyMACs}}{{{a['macs_removed_pct']:.1f}}}")
        macros.append(f"\\newcommand{{\\{name}NinetyDeltaRMSE}}{{{a['delta_rmse_vs_raw']:+.2f}}}")
        macros.append(f"\\newcommand{{\\{name}NinetyDeltaS}}{{{a['delta_score_vs_raw_pct']:+.1f}}}")
    macros.append(f"\\newcommand{{\\XCriterion}}{{{criterion}}}")
    (args.outdir / "headline_macros.tex").write_text("\n".join(macros) + "\n")

    # ---------------------------------------------------------------- report
    for size in MODELS:
        print(f"\n[welch] structured, {size} model")
        head = f"{'ratio':>5s} {'MACs-':>6s} | {'ΔRMSE raw':>9s} {'ΔS% raw':>8s} {'tol':>4s}"
        if control is not None:
            head += f" | {'ΔRMSE ctl':>9s} {'p':>7s} {'ΔS% ctl':>8s} {'p':>7s} {'ok':>3s}"
        print(head)
        for e in per_model[("structured", size)]:
            r, s = e["rmse_row"], e["score_row"]
            line = (f"{e['ratio']:5d} {e['macs_removed_pct']:5.1f}% | {r['delta_vs_raw']:+9.2f} "
                    f"{s['delta_vs_raw_pct']:+8.1f} {'yes' if e['fallback_both'] else 'no':>4s}")
            if control is not None:
                line += (f" | {r['delta_vs_control']:+9.2f} {r['welch_p']:7.4f} "
                         f"{s['delta_vs_control_pct']:+8.1f} {s['welch_p']:7.4f} "
                         f"{'yes' if e['welch_both'] else 'no':>3s}")
            print(line)
        h = headline["models"][size]["headline"]
        print(f"[welch] {size}: X = {fmt_pct(h['x_macs_removed_pct'])} % MACs removed "
              f"({h['claim'] or 'no ratio passes'}; criterion {criterion})")
        if control is not None:
            for metric, pen in headline["models"][size]["fine_tuning_penalty"].items():
                print(f"[welch] {size} fine-tuning penalty {metric}: {pen['raw']:.2f} -> {pen['control_mean']:.2f} "
                      f"({pen['delta_pct']:+.1f} %, p = {pen['p_two_sided']:.4f})")
    print(f"\n[welch] wrote {args.outdir / 'welch_tests.csv'}, headline.json, headline_macros.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
