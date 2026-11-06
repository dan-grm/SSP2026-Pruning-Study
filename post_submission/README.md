# Post-submission experiments (SSP 2026 talk)

Additional measurements for the talk. These are not part of the paper. `src/` stays frozen. These scripts copy the few functions they need into `common.py`, because the `src/` files run the whole experiment on import.

| Script | Needs | Writes |
| --- | --- | --- |
| `batch_sweep.py` | GPU, `model0.pth`, `model3.pth` | `results/post_submission/batch_sweep.{csv,json}` |
| `tuned_control.py` | GPU, checkpoints, `train_timed.pt`, `test_timed.pt` | `results/post_submission/tuned_control.{csv,json}` |
| `welch_test.py` | paper CSVs (+ `tuned_control.csv`) | `results/post_submission/welch_tests.csv`, `headline.json`, `headline_macros.tex` |
| `plot_figures.py` | paper CSVs (+ the two CSVs above) | `presentation_figures/*.pdf` (+ `png/` previews), `results/post_submission/factor_chains.csv` |

## What is measured

**`batch_sweep.py`**: big and small model × {unstructured, structured} × ratios 0–90 % × inference batch sizes {1, 8, 32, 128, 512}. The models are pruned only, without fine-tuning. Resource metrics don't depend on the weight values, so each configuration is measured once. Per row:

- `profiler_mem_mb`: the paper's memory metric, `total_average().device_memory_usage`. This is a sum over operators, not a peak.
- `peak_mem_mb`: `reset_peak_memory_stats()`, then one forward pass, then `max_memory_allocated()`, with only the model under test and its input on the GPU. `resident_mem_mb` is weights + input before the pass. `peak_activation_mb` is the difference.
- `gpu_latency_ms`: median of 100 passes after 1000 warm-ups, as in the paper.
- `cpu_latency_ms`: the same procedure with `torch.set_num_threads(1)`. This is a stand-in for a single-core device, not an MCU measurement.
- `params`, `params_nonzero`, `macs`, `flops`: calflops, for the whole batch, as in the paper.
- `size_mb`: the paper's definition, parameters + buffers.
- `state_dict_bytes`: serialized checkpoint size.

At batch 512, the script prints its numbers next to the paper's as a sanity check. MACs should match exactly. Memory and latency should be close.

**`tuned_control.py`**: the unpruned baselines get the same 20 epochs of fine-tuning as the pruned models (Adam, lr 1e-3, training batch 512), over 10 runs seeded `42 + run`. The output CSV has the same format as the paper's CSVs, with rows `{big,small}_{rmse,score}_tuned_0`. Before training, the script checks that the raw checkpoints reproduce the paper's ratio-0 RMSE and S-Score. If those don't match, the checkpoints or data are not the paper's.

**`welch_test.py`**: for every ratio, a one-sided Welch t-test of pruned+fine-tuned (paper CSV) vs. the control, for RMSE and S-Score, at α = 0.05 with no multiple-comparison correction. X is the largest share of MACs removed (structured pruning) at which neither metric is significantly worse. Without a control CSV, it falls back to a fixed tolerance against the raw checkpoint: ΔRMSE ≤ 0.5 and ΔS ≤ 10 %.

## Running on the server

```sh
git fetch && git checkout post-submission && git pull
docker compose build            # only if pyproject/uv.lock changed (they did not)
# set the MIG UUID in the `post` service of docker-compose.yml (same as `gpu`)

docker compose run --rm post post_submission/batch_sweep.py     # ~1 h, see below
docker compose run --rm post post_submission/tuned_control.py   # 10 runs x 2 models x 20 epochs
docker compose run --rm post post_submission/welch_test.py      # seconds, no GPU needed
docker compose run --rm post post_submission/plot_figures.py    # seconds, no GPU needed
```

Without Docker, from the repo root with the uv environment, use `uv run --frozen python post_submission/<script>.py`. Pass `--model-dir` and `--data-dir` if the data isn't under the Colab paths or `./data/`.

Both GPU scripts **resume**: rows and runs that are already in the CSV are skipped, so an interrupted run can simply be restarted. Use `--overwrite` to re-measure. To check the setup first, run a small subset:

```sh
docker compose run --rm post post_submission/batch_sweep.py --models big --methods structured \
    --ratios 0 90 --batches 1 512 --out results/post_submission/sweep_check.csv
docker compose run --rm post post_submission/tuned_control.py --runs 1 --out results/post_submission/control_check.csv
```

**Runtime.** On an Apple M-series CPU, the single-thread CPU pass takes about 60 ms for the big model at batch 512 and about 0.2 ms at batch 1. With 1100 passes per configuration, the CPU part of the sweep is roughly 30–60 minutes on the EPYC, and the GPU part is under 30 minutes. Use `--skip-cpu` to drop the CPU part, or `--cpu-warmup 100` to shorten it.

## What to copy back

To `ssp2026-presentation/`:

- `presentation_figures/*.pdf`
- `results/post_submission/*.csv`, `*.json`, `headline_macros.tex`

## Notes

- **Seeding of the control.** The paper seeds once per run and then works through all ratios on one RNG stream. The control seeds `42 + run` and then fine-tunes big, then small. Its RNG stream therefore isn't identical to any pruned ratio's. That doesn't matter for a comparison of means across 10 seeds.
- **Pruning device.** The sweep prunes on the CPU so that the GPU holds nothing but the model under test during the peak-memory measurement. Channel selection is deterministic (L1 norm), so this gives the same architecture.
- **Profiler noise.** In the paper's CSV, `memory_mb` differs between the `pruned` and `tuned` rows at the same ratio (big, 10 %: 150.9 vs. 160.7 MB), although the architecture is identical. The profiler sum isn't a stable absolute number, which is one more reason to report the peak.
