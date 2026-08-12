# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a pruning study on CNN-based RUL (Remaining Useful Life) prediction over the NASA C-MAPSS turbofan dataset (FD001–FD004). It compares **unstructured** (`torch.nn.utils.prune.l1_unstructured`) vs. **structured** (`torch-pruning` dependency-graph channel pruning) at ratios 0–90%, before and after fine-tuning, across 10 seeded runs, measuring RMSE, the C-MAPSS asymmetric score, model size, GPU memory, inference latency, params/MACs/FLOPs.

## The hard rule: `src/` is frozen unless the user says otherwise

The files under [src/](src/) are Google Colab notebook exports and are **not to be edited on your own initiative** — reproducing the published numbers means running the same code. Never "fix" something there opportunistically, not even an obvious bug you happen to notice: report it and let the user decide. Edit `src/` only when the user explicitly asks for that specific change, and keep it surgical.

The user has authorised fixes in [src/evaluation.py](src/evaluation.py) and [src/setup_data.py](src/setup_data.py) (see Known issues below). [src/setup_models.py](src/setup_models.py) and [src/pruning_experiments.py](src/pruning_experiments.py) are still byte-for-byte the Colab exports.

Everything needed to run them outside Colab lives outside `src/`:

- [compat/](compat/) — shims for `google.colab`, an offline `gspread` client, CPU fallbacks for the CUDA-only profiling calls, headless figure capture, and the DataFrame→CSV export that used to be a manual spreadsheet step.
- [run_stage.py](run_stage.py) — strips the `!pip install` lines (preserving line numbers), installs the shims, and `exec`s one stage as `__main__`.
- [Dockerfile](Dockerfile) / [docker-compose.yml](docker-compose.yml) — one image for both CPU and GPU.
- [REPRODUCE.md](REPRODUCE.md) — the authoritative operator documentation.

The dependency direction is one-way: nothing in `src/` imports `compat`. When something in `src/` does not work in the container, the fix belongs in `compat/`, not in `src/`.

Verify container changes by actually building and running — `docker build -t ssp2026-pruning:test .` then a stage against synthetic data. Assuming a shim works has been wrong more often than not here (`calflops` needs an undeclared `transformers`; `accelerate` probes `find_spec("google.colab")`, which needs a real `__spec__`).

Run a stage with `docker compose run --rm {cpu|gpu} <stage>`, or locally with `uv run --frozen python run_stage.py <stage>`; `--check` reports the environment and mounts without executing anything. Stages: `setup_data`, `setup_models`, `pruning_experiments`, `evaluation`. There are still no tests or lint config — do not invent commands.

Hardcoded Drive paths (`/content/drive/MyDrive/Bachelor/new coding/{CMAPSSData,NEW_DATA,NEW_MODELS}`) are deliberately left alone; the container recreates that tree and the host mounts onto it. [main.py](main.py) is an unused `uv init` stub.

Each file carries leftover exploratory cells (shape prints, sample plots, `train_data[0].head(193)`) and redefines `CNN`, `TimeWindowDataset`, `create_dataloaders`, `setup_data`, `rmse_loss`, `scoring_function` rather than sharing them. That duplication is intentional-by-freeze; do not "clean it up".

Line numbers in `src/evaluation.py` shifted by a few lines with the authorised edits — re-read before citing one.

## Pipeline (the files run in this order)

1. **[src/setup_data.py](src/setup_data.py)** — reads the raw `train_FD00{1..4}.txt` / `test_FD00{1..4}.txt` / `RUL_FD00{1..4}.txt`, drops 10 low-variance/constant columns (leaving unit id, cycle, 15 columns), MinMax-scales to [-1, 1] per subset (train scalers reused on test), computes the RUL label, slices into time windows, and saves `train_timed.pt` / `test_timed.pt` to `DATA_DIR`.
2. **[src/setup_models.py](src/setup_models.py)** — loads the `.pt` tensors, wraps them in `TimeWindowDataset`/`DataLoader`, trains one `CNN` per subset for 250 epochs, writes `model{0..3}.pth` to `MODEL_DIR`.
3. **[src/pruning_experiments.py](src/pruning_experiments.py)** — the main experiment. Loads `model0` (big, tw=30) and `model3` (small, tw=15), and for each run/ratio deep-copies the baseline, prunes, profiles, fine-tunes (20 epochs, lr 1e-3), profiles again, and pushes results into a metrics DataFrame and a Google Sheet.
4. **[src/evaluation.py](src/evaluation.py)** — reads `structured_pruning_data_3.csv` / `unstructured_pruning_data_3.csv` (exported from the sheets; not in the repo) and produces the plots, with `curve_fit` trendlines (linear by default, `exp_decay` for params/MACs/FLOPs).

## Conventions that matter when editing

- **Time windows are per-subset**: `[30, 20, 30, 15]` for FD001–FD004. `CNN.fc1` is sized `(sequence_length - 2) * feature_length`, so `sequence_length` must match the loader's window or the state dict won't load. `feature_length=14` (the 15th column is the label).
- **Labels are normalized RUL**: RUL is clipped at 125 cycles then mapped to `2*(rul/125) - 1`. Every metric is computed after de-normalizing with `(x + 1) / 2 * 125` — keep that de-norm in any new loss/metric or the numbers are not comparable.
- **`scoring_function`** is the standard asymmetric C-MAPSS score (late predictions penalized via `exp(d/10)`, early via `exp(-d/13)`); lower is better, and it is summed then averaged per batch.
- Metric rows are keyed by the string `f"{size}_{metric}_{typ}_{int(ratio*100)}"` (e.g. `big_rmse_tuned_40`), and [src/evaluation.py](src/evaluation.py) parses that exact shape back out with a regex. Changing the naming breaks the evaluation stage.
- Sheet columns B–K map to runs 1–10; `update_sheet` only fills cells that are currently empty.
- Reproducibility is `set_seed(42 + run)` at the start of each run.
- Style is 2-space indentation throughout (Colab default), not 4.

## Known issues in the current code

Pre-existing defects. The user has said the remaining ones will be fixed later — do not fix them opportunistically. `REPRODUCE.md` documents their operational effect.

Nothing blocks a stage any more — all four run to completion.

Fixed in [src/evaluation.py](src/evaluation.py) — stage 4 now exits 0 with 70 figures, where before it produced none:

- `plot_metric_inter` reused the pruned series' x-values for the tuned series, which has one fewer point because no model is fine-tuned at ratio 0 → `ValueError: x and y must be the same size`. It now takes the tuned ratios via `x2`; `plot_trend` gained an optional `x2` defaulting to `x`, so `plot_metric_intra` is unaffected.
- The driver loops rebound the name `plot_trend` from the function to a bool → `TypeError: 'bool' object is not callable`. The loop variable is now `use_trendline`.
- The two chained-indexing lines (`df[a][b][c]`, results discarded) that emitted `UserWarning: Boolean Series key will be reindexed` were removed.

Fixed in [src/setup_data.py](src/setup_data.py) — stage 1 now exits 0 and its `plot_rul` calls run:

- `preprocess()` had its `return` commented out, so unpacking its result raised `TypeError` (after `torch.save`, so the tensors were fine). The `return` is restored, the duplicated `head(193)` debug prints removed, and the driver wrapped in `if __name__ == "__main__":` — compatible with `run_stage.py`, which execs with `__name__ = "__main__"`.
- Still there by design: the hardcoded `train_timed[0][77]` debug print at [:129-130](src/setup_data.py#L129-L130), which needs ≥78 test units. Real C-MAPSS is fine; trimmed test subsets are not.

**Open question for the authors** (not a code bug): `update_sheet` writes positionally into a sheet whose metric labels in column A were filled in by hand, so if that sheet ever carried `*_tuned_0` label rows, every published value after the first is shifted by one row. Needs the original exported CSVs to settle.

Non-blocking:

- [src/pruning_experiments.py:217-226](src/pruning_experiments.py#L217-L226) adds to `train_loss` twice per batch (once normalized, once de-normalized), so reported fine-tuning train loss is inflated.
- `profile_models` uses the module-level `big_tw_test_loader` / `small_tw_test_loader` globals regardless of arguments, and takes an unused `sheet` parameter.
- `measure_inference_time` calls `torch.cuda.synchronize()` unconditionally — handled by [compat/cuda.py](compat/cuda.py) on CPU.
- The two `gc.open_by_url("-")` calls at [src/pruning_experiments.py:346-347](src/pruning_experiments.py#L346-L347) are stripped placeholders — handled by [compat/sheets.py](compat/sheets.py), which distinguishes the sheets by open order.

**Not** a bug, despite appearances: `optim.Adam(model.named_parameters(), ...)` at [src/setup_models.py:145](src/setup_models.py#L145) is valid from PyTorch 2.6 on — optimizers accept named parameters and record them as `param_names`. Training works.

## Dependencies

[pyproject.toml](pyproject.toml) holds constraints, `uv.lock` the exact pins; the image installs with `uv sync --frozen`. Edit and `uv lock`, then commit both. Two non-obvious pins: `transformers` (imported by `calflops` at import time but undeclared by it) and `pandas<3` (`read_csv(delim_whitespace=True)` was removed in 3.0). `torch<2.9` keeps the wheels on the CUDA 12.x runtime — 2.9+ needs an NVIDIA driver ≥ 580 on the host.
