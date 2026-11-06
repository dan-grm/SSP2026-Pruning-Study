#!/usr/bin/env python3
"""Resource metrics of the pruned models across inference batch sizes.

Grid: {big, small} x {unstructured, structured} x ratios 0..90 % x batch sizes
{1, 8, 32, 128, 512}. Models are pruned only, never fine-tuned: resource
metrics do not depend on the weights' values or the seed, and channel
selection is deterministic, so every configuration is measured once.

Per configuration (one row of the output CSV):

* ``profiler_mem_mb``: memory as in the paper,
  ``p.key_averages().total_average().device_memory_usage``. This adds up the
  memory allocated by every operator, so it is a consistent relative measure
  but not a peak.
* ``peak_mem_mb``: ``reset_peak_memory_stats`` -> one forward pass ->
  ``max_memory_allocated``. Only the model under test and its input are on the
  GPU at that point, so this is weights + input + the largest set of live
  intermediate tensors (including any cuDNN workspace drawn from the caching
  allocator). ``resident_mem_mb`` is what was allocated before the pass
  (weights + input), and ``peak_activation_mb`` the difference.
* ``gpu_latency_ms``: median of 100 timed passes after 1000 warm-ups, as in
  the paper.
* ``cpu_latency_ms``: the same on the CPU with ``torch.set_num_threads(1)``.
  This is a stand-in for a single-core device, not an MCU measurement.
* ``params``, ``params_nonzero``, ``macs``, ``flops`` (calflops, for the whole
  batch, as in the paper), ``size_mb`` (the paper's definition: parameters +
  buffers in memory) and ``state_dict_bytes`` (serialized checkpoint size).

Rows are written after every configuration, so a crash loses at most one.
Re-running skips rows that already exist unless ``--overwrite`` is given.

Usage (from the repo root):
    python post_submission/batch_sweep.py
    python post_submission/batch_sweep.py --models big --batches 1 512 --ratios 0 90
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import common
from common import MODELS, RATIOS, input_shape

BATCHES = [1, 8, 32, 128, 512]
METHODS = ["unstructured", "structured"]
MB = 1024 * 1024

KEY_COLUMNS = ["model", "method", "ratio", "batch_size"]
COLUMNS = KEY_COLUMNS + [
    "params", "params_nonzero", "macs", "flops", "size_mb", "state_dict_bytes",
    "profiler_mem_mb", "peak_mem_mb", "resident_mem_mb", "peak_activation_mb",
    "gpu_latency_ms", "gpu_latency_std_ms", "cpu_latency_ms", "cpu_latency_std_ms",
    "gpu_idle_alloc_bytes",
]


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------

def gpu_latency(model, x, runs, warmup):
    """src/pruning_experiments.py: measure_inference_time (seconds)."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return float(np.median(times)), float(np.std(times))


def cpu_latency(model, x, runs, warmup):
    """Same procedure on the CPU, single-threaded (seconds)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
            times = []
            for _ in range(runs):
                start = time.perf_counter()
                _ = model(x)
                times.append(time.perf_counter() - start)
    finally:
        torch.set_num_threads(previous)
    return float(np.median(times)), float(np.std(times))


def profiler_memory(model, x):
    """src/pruning_experiments.py: model_mem_time, memory part (MiB)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as p:
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    return p.key_averages().total_average().device_memory_usage / MB


def peak_memory(model, x):
    """(peak, resident) bytes for one forward pass."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del y
    return peak, resident


def static_metrics(model, size, batch):
    from calflops import calculate_flops

    flops, macs, params = calculate_flops(
        model=model, input_shape=input_shape(size, batch),
        print_results=False, print_detailed=False, output_as_string=False,
    )
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    # Effective non-zero weights: for unstructured pruning the mask, not the
    # stored weight_orig, decides what is zero.
    nonzero = 0
    for name, p in model.named_parameters():
        module_name, _, pname = name.rpartition(".")
        module = model.get_submodule(module_name) if module_name else model
        if pname.endswith("_orig") and hasattr(module, pname[:-5] + "_mask"):
            p = p * getattr(module, pname[:-5] + "_mask")
        nonzero += int(torch.count_nonzero(p).item())
    return {
        "params": int(params),
        "params_nonzero": nonzero,
        "macs": int(macs),
        "flops": int(flops),
        "size_mb": common.calc_model_size(model),
        "state_dict_bytes": buffer.getbuffer().nbytes,
    }


def measure(model, size, batch, args, gpu: torch.device | None):
    row = static_metrics(model, size, batch)
    nan = math.nan
    row.update(profiler_mem_mb=nan, peak_mem_mb=nan, resident_mem_mb=nan, peak_activation_mb=nan,
               gpu_latency_ms=nan, gpu_latency_std_ms=nan, gpu_idle_alloc_bytes=nan,
               cpu_latency_ms=nan, cpu_latency_std_ms=nan)

    if gpu is not None:
        torch.cuda.empty_cache()
        row["gpu_idle_alloc_bytes"] = torch.cuda.memory_allocated()
        model.to(gpu)
        x = torch.randn(*input_shape(size, batch), device=gpu)
        median, std = gpu_latency(model, x, args.gpu_runs, args.gpu_warmup)
        row["gpu_latency_ms"], row["gpu_latency_std_ms"] = median * 1000, std * 1000
        row["profiler_mem_mb"] = profiler_memory(model, x)
        peak, resident = peak_memory(model, x)
        row["peak_mem_mb"] = peak / MB
        row["resident_mem_mb"] = resident / MB
        row["peak_activation_mb"] = (peak - resident) / MB
        del x
        model.to("cpu")
        torch.cuda.empty_cache()

    if not args.skip_cpu:
        x = torch.randn(*input_shape(size, batch))
        median, std = cpu_latency(model, x, args.cpu_runs, args.cpu_warmup)
        row["cpu_latency_ms"], row["cpu_latency_std_ms"] = median * 1000, std * 1000
    return row


# --------------------------------------------------------------------------
# CSV bookkeeping
# --------------------------------------------------------------------------

def key_of(row) -> tuple:
    return (row["model"], row["method"], int(row["ratio"]), int(row["batch_size"]))


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict]) -> None:
    order = {m: i for i, m in enumerate(MODELS)} | {m: i for i, m in enumerate(METHODS)}
    rows = sorted(rows, key=lambda r: (order[r["model"]], order[r["method"]], int(r["ratio"]), int(r["batch_size"])))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def compare_with_paper(rows: list[dict]) -> None:
    """Print the batch-512 numbers next to the paper's, as a sanity check."""
    paper = {}
    for method, path in common.PAPER_CSV.items():
        if path.exists():
            with path.open() as f:
                paper[method] = {r["metric"]: r for r in csv.DictReader(f)}
    if not paper:
        return
    print("\n[sweep] batch 512 vs. paper (pruned, avg over runs)")
    print(f"{'model':6s} {'method':13s} {'ratio':>5s} | {'mem paper':>9s} {'mem now':>9s} | "
          f"{'lat paper':>9s} {'lat now':>9s} | {'MACs paper':>12s} {'MACs now':>12s}")
    for r in rows:
        if int(r["batch_size"]) != 512 or int(r["ratio"]) not in (0, 50, 90) or r["method"] not in paper:
            continue
        p = paper[r["method"]]
        tag = f"{r['model']}_{{}}_pruned_{int(r['ratio'])}"
        get = lambda m: float(p[tag.format(m)]["avg"]) if tag.format(m) in p else math.nan
        print(f"{r['model']:6s} {r['method']:13s} {int(r['ratio']):5d} | "
              f"{get('memory_mb'):9.2f} {float(r['profiler_mem_mb']):9.2f} | "
              f"{get('inference_ms'):9.3f} {float(r['gpu_latency_ms']):9.3f} | "
              f"{get('MACs'):12.0f} {float(r['macs']):12.0f}")


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-dir", type=Path, default=common.DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=common.RESULTS_DIR / "batch_sweep.csv")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--ratios", nargs="+", type=int, default=RATIOS)
    parser.add_argument("--batches", nargs="+", type=int, default=BATCHES)
    parser.add_argument("--gpu-warmup", type=int, default=1000)
    parser.add_argument("--gpu-runs", type=int, default=100)
    parser.add_argument("--cpu-warmup", type=int, default=1000)
    parser.add_argument("--cpu-runs", type=int, default=100)
    parser.add_argument("--skip-cpu", action="store_true", help="skip the single-thread CPU latency")
    parser.add_argument("--allow-no-gpu", action="store_true",
                        help="run without CUDA (GPU columns stay empty); for smoke tests")
    parser.add_argument("--overwrite", action="store_true", help="re-measure rows that already exist")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_no_gpu:
        print("[sweep] no CUDA device; pass --allow-no-gpu to measure CPU latency only")
        return 2
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    # Pruning itself runs on the CPU: channel selection is deterministic and
    # the result is identical, and it keeps the GPU empty between measurements.
    prune_device = torch.device("cpu")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.out)
    done = {key_of(r) for r in rows}
    if args.overwrite:
        wanted = {(m, me, r, b) for m in args.models for me in args.methods for r in args.ratios for b in args.batches}
        rows = [r for r in rows if key_of(r) not in wanted]
        done = {key_of(r) for r in rows}

    meta = common.env_info() | {"args": {k: str(v) for k, v in vars(args).items()}}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[sweep] {json.dumps(meta)}")

    todo = [(m, me, r, b) for m in args.models for me in args.methods for r in args.ratios
            for b in args.batches if (m, me, r, b) not in done]
    print(f"[sweep] {len(todo)} configurations to measure, {len(done)} already in {args.out}")

    baselines = {size: common.load_baseline(size, args.model_dir) for size in args.models}
    started = time.time()
    count = 0
    for size in args.models:
        for method in args.methods:
            for ratio in args.ratios:
                batches = [b for b in args.batches if (size, method, ratio, b) not in done]
                if not batches:
                    continue
                model = common.pruned_copy(baselines[size], method, ratio, size, prune_device).to("cpu")
                for batch in batches:
                    row = {"model": size, "method": method, "ratio": ratio, "batch_size": batch}
                    row.update(measure(model, size, batch, args, gpu))
                    rows.append(row)
                    write_rows(args.out, rows)
                    count += 1
                    print(f"[sweep] {count:3d}/{len(todo)} {size:5s} {method:12s} {ratio:2d}% b={batch:3d} | "
                          f"MACs {row['macs'] / 1e6:8.2f}M | prof {row['profiler_mem_mb']:8.3f} MB | "
                          f"peak {row['peak_mem_mb']:8.3f} MB | gpu {row['gpu_latency_ms']:7.3f} ms | "
                          f"cpu {row['cpu_latency_ms']:8.3f} ms | {time.time() - started:6.0f}s")
                del model

    compare_with_paper(read_rows(args.out))
    print(f"[sweep] done -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
