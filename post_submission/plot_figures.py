#!/usr/bin/env python3
"""All figures for the SSP talk, in the slide style, written to presentation_figures/.

Each figure is drawn only if its inputs exist, so this can run before or after
the post-submission experiments:

From the paper's CSVs (always):
    accuracy_structured           RMSE and S-Score vs. pruning ratio, both models (slide 5)
    unstructured_vs_structured    latency and profiler memory vs. ratio (slide 4)
    relative_resources_{big,small}  remaining MACs/latency/memory/params/size in % (slide 7)
With results/post_submission/tuned_control.csv:
    accuracy_structured_control   as above, plus the tuned_0 control band
With results/post_submission/batch_sweep.csv:
    batch_sweep_{big,small}       memory and GPU/CPU latency vs. batch size (slide 8)
    memory_kinds_{big,small}      profiler sum vs. peak memory, STM32N6 line (slide 9)
    factor_chain_{big,small}      reduction factors at 90 %, batch 512 vs. 1 (slides 7-8)
    + results/post_submission/factor_chains.csv (also includes the paper's batch-512 chain)

PNG previews go to presentation_figures/png/.

Usage (from the repo root):
    python post_submission/plot_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator

import plot_style as st

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results" / "post_submission"
FIGURES_DIR = REPO / "presentation_figures"
PAPER_CSV = {
    "structured": REPO / "results" / "experiment_logs" / "structured_pruning_data_3.csv",
    "unstructured": REPO / "results" / "experiment_logs" / "unstructured_pruning_data_3.csv",
}
MODELS = ["big", "small"]
MODEL_LABEL = {"big": "Big model (FD001, $N_{tw}$ = 30)", "small": "Small model (FD004, $N_{tw}$ = 15)"}
RATIOS = list(range(0, 100, 10))
STM32N6_RAM_MB = 4.2
CHAIN_RATIO = 90
SWEEP_RATIOS = {0: st.NAVY, 50: st.CORNFLOWER, 90: st.CORAL}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_paper(method: str) -> pd.DataFrame:
    df = pd.read_csv(PAPER_CSV[method])
    parts = df["metric"].str.extract(r"^(big|small)_(.+)_(pruned|tuned)_(\d+)$")
    df["model"], df["kind"], df["stage"], df["ratio"] = parts[0], parts[1], parts[2], parts[3].astype(int)
    return df


def series(df: pd.DataFrame, model: str, kind: str, stage: str):
    sub = df[(df.model == model) & (df.kind == kind) & (df.stage == stage)].sort_values("ratio")
    return sub["ratio"].to_numpy(), sub["avg"].to_numpy(float), sub["std"].to_numpy(float)


def macs_removed(df: pd.DataFrame, model: str) -> dict[int, float]:
    r, macs, _ = series(df, model, "MACs", "pruned")
    return {int(k): 100 * (1 - v / macs[0]) for k, v in zip(r, macs)}


def load_control(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "tuned_control.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path).set_index("metric")
    return df if df["avg"].notna().all() else None


def load_sweep(results_dir: Path) -> pd.DataFrame | None:
    path = results_dir / "batch_sweep.csv"
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def ratio_axis(ax, mr: dict[int, float], ticks=(0, 30, 60, 90), xlabel=True):
    """x = conv channels pruned, with the MACs removed printed underneath."""
    ax.set_xlim(-5, 95)
    ax.set_xticks(list(ticks))
    ax.set_xticklabels([f"{t}\n{mr[t]:.0f}" for t in ticks])
    if xlabel:
        ax.set_xlabel("conv channels pruned [%]\n(below: MACs removed [%])", fontsize=8, color=st.INK_MUTED)


def batch_axis(ax, batches):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(batches))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("inference batch size")


def plain_log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def end_label(ax, x, y, text, color, dx=2, **kw):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center",
                fontsize=8, color=st.INK, **kw)


# --------------------------------------------------------------------------
# Paper figures
# --------------------------------------------------------------------------

def fig_accuracy(structured: pd.DataFrame, control: pd.DataFrame | None, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(st.FULL_WIDTH, 3.1), sharex="col")
    for col, model in enumerate(MODELS):
        mr = macs_removed(structured, model)
        for row, (kind, label) in enumerate([("rmse", "RMSE [cycles]"), ("score", "S-Score")]):
            ax = axes[row, col]
            raw = series(structured, model, kind, "pruned")[1][0]
            if control is not None:
                c = control.loc[f"{model}_{kind}_tuned_0"]
                ax.axhspan(c["avg"] - c["std"], c["avg"] + c["std"], color=st.SAND, lw=0, zorder=0)
                ax.axhline(c["avg"], color=st.GREY, lw=0.8, ls=":", zorder=1)
            ax.axhline(raw, color=st.GREY, lw=0.9, ls="--", zorder=1)
            x, y, s = series(structured, model, kind, "pruned")
            ax.errorbar(x, y, yerr=s, color=st.CORNFLOWER, marker="o", mfc="white", lw=1.2,
                        capsize=1.5, elinewidth=0.6, zorder=2)
            x, y, s = series(structured, model, kind, "tuned")
            ax.errorbar(x, y, yerr=s, color=st.NAVY, marker="o", lw=1.6, capsize=1.5,
                        elinewidth=0.6, zorder=3)
            if kind == "score":
                plain_log_y(ax)
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(MODEL_LABEL[model])
            ratio_axis(ax, mr, xlabel=(row == 1))
    handles = [
        Line2D([], [], color=st.CORNFLOWER, marker="o", mfc="white", lw=1.2, label="pruned"),
        Line2D([], [], color=st.NAVY, marker="o", lw=1.6, label="pruned + 20 epochs fine-tuning"),
        Line2D([], [], color=st.GREY, ls="--", lw=0.9, label="unpruned baseline"),
    ]
    if control is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=st.SAND, label="baseline + same fine-tuning (mean ± std)"))
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1),
               handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(h_pad=0.4)
    st.save(fig, outdir, "accuracy_structured_control" if control is not None else "accuracy_structured")


def fig_unstructured_vs_structured(paper: dict[str, pd.DataFrame], outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(st.FULL_WIDTH, st.HEIGHT))
    colors = {"structured": st.NAVY, "unstructured": st.CORAL}
    styles = {"big": ("o", "-"), "small": ("s", "--")}
    for ax, (kind, label) in zip(axes, [("inference_ms", "GPU latency [ms]"),
                                        ("memory_mb", "memory (profiler sum) [MB]")]):
        for method, df in paper.items():
            for model in MODELS:
                x, y, _ = series(df, model, kind, "pruned")
                marker, ls = styles[model]
                ax.plot(x, y, color=colors[method], marker=marker, ls=ls, mfc="white" if model == "small" else None)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-5, 95)
        ax.set_xticks(RATIOS[::3])
        ax.set_xlabel("pruning ratio [%]")
        ax.set_title(label)
    handles = [Line2D([], [], color=colors[m], lw=1.6, label=m) for m in colors] + \
              [Line2D([], [], color=st.GREY, marker=styles[m][0], ls=styles[m][1],
                      mfc="white" if m == "small" else None, label=f"{m} model") for m in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    st.save(fig, outdir, "unstructured_vs_structured")


def fig_relative_resources(structured: pd.DataFrame, model: str, outdir: Path):
    fig, ax = plt.subplots(figsize=(st.FULL_WIDTH * 0.8, st.HEIGHT))
    mr = macs_removed(structured, model)
    entries = [
        ("MACs", "MACs", st.NAVY, "-"),
        ("inference_ms", "GPU latency", st.CORNFLOWER, "-"),
        ("memory_mb", "memory (profiler)", st.CORAL, "-"),
        # Parameters and model size coincide within 0.1 %, so they share one line.
        ("params", "parameters ≈ model size", st.GREY, "-"),
    ]
    ends = []
    for kind, label, color, ls in entries:
        x, y, _ = series(structured, model, kind, "pruned")
        rel = 100 * y / y[0]
        ax.plot(x, rel, color=color, ls=ls, marker="o", ms=3.5)
        ends.append([rel[-1], f"{label} {rel[-1]:.0f} %"])
    # Keep end labels at least 9 percentage points apart.
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 9)
    for y_label, text in ends:
        ax.annotate(text, (90, y_label), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=st.INK, annotation_clip=False)
    ax.set_ylim(0, 108)
    ax.set_ylabel("remaining [% of unpruned]")
    ratio_axis(ax, mr)
    ax.set_title(f"{MODEL_LABEL[model]}, structured, batch 512, no fine-tuning")
    fig.tight_layout()
    st.save(fig, outdir, f"relative_resources_{model}")


# --------------------------------------------------------------------------
# Batch-sweep figures
# --------------------------------------------------------------------------

def fig_batch_sweep(sweep: pd.DataFrame, model: str, outdir: Path):
    sub = sweep[(sweep.model == model) & (sweep.method == "structured")]
    batches = sorted(sub.batch_size.unique())
    panels = [("profiler_mem_mb", "memory (profiler sum) [MB]"),
              ("gpu_latency_ms", "GPU latency [ms]"),
              ("cpu_latency_ms", "CPU latency, 1 thread [ms]")]
    panels = [p for p in panels if sub[p[0]].notna().any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(st.FULL_WIDTH, st.HEIGHT))
    axes = np.atleast_1d(axes)
    for ax, (col, title) in zip(axes, panels):
        for ratio, color in SWEEP_RATIOS.items():
            d = sub[sub.ratio == ratio].sort_values("batch_size")
            ax.plot(d.batch_size, d[col], color=color, marker="o", label=f"{ratio} % pruned")
        batch_axis(ax, batches)
        plain_log_y(ax)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"{l} (structured)" if i == 0 else l for i, l in enumerate(labels)],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    st.save(fig, outdir, f"batch_sweep_{model}")


def fig_memory_kinds(sweep: pd.DataFrame, structured: pd.DataFrame, model: str, outdir: Path):
    sub = sweep[(sweep.model == model) & (sweep.method == "structured")]
    if sub.peak_mem_mb.isna().all():
        return
    mr = macs_removed(structured, model)
    batches = [b for b in (512, 1) if b in set(sub.batch_size)]
    fig, axes = plt.subplots(1, len(batches), figsize=(st.FULL_WIDTH, st.HEIGHT), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, batch in zip(axes, batches):
        d = sub[sub.batch_size == batch].sort_values("ratio")
        ax.plot(d.ratio, d.profiler_mem_mb, color=st.NAVY, marker="o", label="profiler sum (paper's metric)")
        ax.plot(d.ratio, d.peak_mem_mb, color=st.CORAL, marker="s", label="peak allocated")
        ax.axhline(STM32N6_RAM_MB, color=st.GREY, ls="--", lw=0.9)
        ax.annotate(f"STM32N6 RAM ({STM32N6_RAM_MB} MB)", (-3, STM32N6_RAM_MB), xytext=(0, -3),
                    textcoords="offset points", ha="left", va="top", fontsize=7.5, color=st.INK_MUTED)
        plain_log_y(ax)
        ratio_axis(ax, mr)
        ax.set_title(f"batch {batch}")
    axes[0].set_ylabel("GPU memory [MB]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    st.save(fig, outdir, f"memory_kinds_{model}")


CHAIN = [  # column in sweep, label, column in paper CSV (None: not in paper)
    ("macs", "MACs", "MACs"),
    ("gpu_latency_ms", "GPU latency", "inference_ms"),
    ("cpu_latency_ms", "CPU latency (1 thread)", None),
    ("profiler_mem_mb", "memory (profiler sum)", "memory_mb"),
    ("peak_mem_mb", "peak memory", None),
    ("params", "parameters", "params"),
    ("size_mb", "model size", "size_mb"),
]


def factor_chains(sweep: pd.DataFrame | None, structured: pd.DataFrame) -> pd.DataFrame:
    """Reduction factor unpruned / 90 % structured, per model and batch size."""
    rows = []
    for model in MODELS:
        for col, label, paper_kind in CHAIN:
            if paper_kind is not None:
                _, y, _ = series(structured, model, paper_kind, "pruned")
                rows.append({"source": "paper", "model": model, "batch_size": 512, "metric": label,
                             "unpruned": y[0], "pruned": y[-1], "factor": y[0] / y[-1]})
            if sweep is None:
                continue
            sub = sweep[(sweep.model == model) & (sweep.method == "structured")]
            for batch in sorted(sub.batch_size.unique()):
                d = sub[sub.batch_size == batch].set_index("ratio")
                if 0 not in d.index or CHAIN_RATIO not in d.index or pd.isna(d.loc[0, col]):
                    continue
                a, b = d.loc[0, col], d.loc[CHAIN_RATIO, col]
                rows.append({"source": "batch_sweep", "model": model, "batch_size": int(batch), "metric": label,
                             "unpruned": a, "pruned": b, "factor": a / b})
    return pd.DataFrame(rows)


def fig_factor_chain(chains: pd.DataFrame, model: str, outdir: Path):
    sub = chains[(chains.source == "batch_sweep") & (chains.model == model)]
    batches = [b for b in (512, 1) if b in set(sub.batch_size)]
    if not batches:
        return
    labels = [label for _, label, _ in CHAIN if label in set(sub.metric)]
    fig, ax = plt.subplots(figsize=(st.FULL_WIDTH * 0.8, st.HEIGHT))
    height = 0.8 / len(batches)
    colors = {512: st.NAVY, 1: st.CORNFLOWER}
    y = np.arange(len(labels))
    for i, batch in enumerate(batches):
        d = sub[sub.batch_size == batch].set_index("metric").reindex(labels)
        pos = y + (i - (len(batches) - 1) / 2) * height
        ax.barh(pos, d.factor, height=height * 0.9, color=colors[batch], label=f"batch {batch}")
        for p, f in zip(pos, d.factor):
            if pd.notna(f):
                ax.annotate(f"{f:.1f}×", (f, p), xytext=(3, 0), textcoords="offset points",
                            va="center", fontsize=7.5, color=st.INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlim(0.8, sub.factor.max() * 2.2)
    ax.axvline(1, color=st.INK_MUTED, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"reduction factor, unpruned → {CHAIN_RATIO} % channels pruned (structured)")
    ax.set_title(MODEL_LABEL[model])
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.save(fig, outdir, f"factor_chain_{model}")


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outdir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="where tuned_control.csv and batch_sweep.csv are read and factor_chains.csv is written")
    args = parser.parse_args()

    st.apply()
    paper = {m: load_paper(m) for m in PAPER_CSV}
    structured = paper["structured"]
    control = load_control(args.results_dir)
    sweep = load_sweep(args.results_dir)

    fig_accuracy(structured, None, args.outdir)
    if control is not None:
        fig_accuracy(structured, control, args.outdir)
    else:
        print("[plot] no complete tuned_control.csv: skipping accuracy_structured_control")
    fig_unstructured_vs_structured(paper, args.outdir)
    for model in MODELS:
        fig_relative_resources(structured, model, args.outdir)

    chains = factor_chains(sweep, structured)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    chains.to_csv(args.results_dir / "factor_chains.csv", index=False)
    print(f"[plot] wrote {args.results_dir / 'factor_chains.csv'}")
    for (source, model, batch), d in chains.groupby(["source", "model", "batch_size"], sort=False):
        chain = " → ".join(f"{f:.1f}× {m}" for m, f in zip(d.metric, d.factor))
        print(f"[chain] {source:11s} {model:5s} b={batch:<3d} {chain}")

    if sweep is None:
        print("[plot] no batch_sweep.csv: skipping batch-sweep figures")
        return 0
    for model in MODELS:
        fig_batch_sweep(sweep, model, args.outdir)
        fig_memory_kinds(sweep, structured, model, args.outdir)
        fig_factor_chain(chains, model, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
