#!/usr/bin/env python3
"""``tuned_0`` control: fine-tune the *unpruned* baselines exactly like the pruned ones.

The paper compares pruned+fine-tuned models with the raw checkpoint. That mixes
two effects: pruning, and 20 more epochs at lr 1e-3 (the baseline ended its
training at lr 1e-4). This control isolates the second effect: the unpruned
baseline gets the same 20 epochs (Adam, lr 1e-3, training batch 512), over
10 runs seeded ``42 + run``, for both models.

Output: ``results/post_submission/tuned_control.csv`` in the paper's CSV format
(rows ``{big,small}_{rmse,score}_tuned_0``, columns ``run_1..run_10, avg, std``
with the sample std, ddof=1, like the paper's CSVs). The file is rewritten
after every run, and runs that are already filled in are skipped unless
``--overwrite`` is given.

Seeding: the paper calls ``set_seed(42 + run)`` once per run and then works
through all ratios on one RNG stream, so the stream a given ratio saw depends
on everything before it. That cannot be reproduced for a ratio that was never
run; here each run does ``set_seed(42 + run)`` and then fine-tunes the big and
then the small model, as the paper does within one ratio.

Usage (from the repo root):
    python post_submission/tuned_control.py
    python post_submission/tuned_control.py --runs 1 2 3
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

import common
from common import MODELS

METRICS = ["rmse", "score"]
NUM_RUNS = 10


def row_name(size: str, metric: str) -> str:
    return f"{size}_{metric}_tuned_0"


def read_table(path: Path) -> dict[str, list[float]]:
    table = {row_name(s, m): [math.nan] * NUM_RUNS for s in MODELS for m in METRICS}
    if path.exists():
        with path.open() as f:
            for r in csv.DictReader(f):
                if r["metric"] in table:
                    table[r["metric"]] = [float(r[f"run_{i + 1}"]) if r[f"run_{i + 1}"] else math.nan
                                          for i in range(NUM_RUNS)]
    return table


def write_table(path: Path, table: dict[str, list[float]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + [f"run_{i + 1}" for i in range(NUM_RUNS)] + ["avg", "std"])
        for name, values in table.items():
            v = np.array(values, dtype=float)
            v = v[~np.isnan(v)]
            avg = float(v.mean()) if len(v) else ""
            std = float(v.std(ddof=1)) if len(v) > 1 else ""
            writer.writerow([name] + ["" if math.isnan(x) else x for x in values] + [avg, std])
    tmp.replace(path)


def paper_value(size: str, metric: str) -> float:
    path = common.PAPER_CSV["structured"]
    if not path.exists():
        return math.nan
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["metric"] == f"{size}_{metric}_pruned_0":
                return float(r["avg"])
    return math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=common.DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=common.DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=common.RESULTS_DIR / "tuned_control.csv")
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(1, NUM_RUNS + 1)),
                        help="run numbers 1..10 (seed = 42 + run)")
    parser.add_argument("--epochs", type=int, default=common.FINE_TUNE_EPOCHS)
    parser.add_argument("--device", default=None, help="default: cuda if available")
    parser.add_argument("--overwrite", action="store_true", help="re-run runs that are already filled in")
    args = parser.parse_args()

    device = common.device_from_arg(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = common.env_info() | {"device": str(device), "args": {k: str(v) for k, v in vars(args).items()}}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[control] {json.dumps(meta)}")

    loaders = common.load_loaders(args.data_dir)
    baselines = {size: common.load_baseline(size, args.model_dir) for size in MODELS}

    # Sanity check: the raw checkpoints must reproduce the paper's ratio-0 numbers.
    for size in MODELS:
        rmse, score = common.validate_model(copy.deepcopy(baselines[size]), loaders[size][1], device)
        print(f"[control] raw {size:5s}: RMSE {rmse:.4f} (paper {paper_value(size, 'rmse'):.4f}), "
              f"S {score:.2f} (paper {paper_value(size, 'score'):.2f})")

    table = read_table(args.out)
    for run in args.runs:
        filled = all(not math.isnan(table[row_name(s, m)][run - 1]) for s in MODELS for m in METRICS)
        if filled and not args.overwrite:
            print(f"[control] run {run} already in {args.out}, skipping")
            continue
        started = time.time()
        common.set_seed(42 + run)
        for size in MODELS:  # big, then small, as in the paper
            train_dl, test_dl = loaders[size]
            model = copy.deepcopy(baselines[size])
            common.fine_tune_model(model, train_dl, test_dl, device, num_epochs=args.epochs,
                                   log_prefix=f"[control] run {run:2d} {size:5s} ")
            rmse, score = common.validate_model(model, test_dl, device)
            table[row_name(size, "rmse")][run - 1] = rmse
            table[row_name(size, "score")][run - 1] = score
            print(f"[control] run {run:2d} {size:5s}: RMSE {rmse:.4f}  S {score:.2f}")
        write_table(args.out, table)
        print(f"[control] run {run} done in {time.time() - started:.0f}s -> {args.out}")

    for size in MODELS:
        for metric in METRICS:
            v = np.array(table[row_name(size, metric)], dtype=float)
            v = v[~np.isnan(v)]
            if len(v):
                print(f"[control] {size:5s} {metric:5s}: mean {v.mean():9.3f}  std {v.std(ddof=1) if len(v) > 1 else 0:8.3f}  "
                      f"n={len(v)}  (raw checkpoint {paper_value(size, metric):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
