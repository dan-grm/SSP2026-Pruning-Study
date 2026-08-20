# Structured vs. unstructured pruning for CNN-based RUL prediction

Code and reproduction package for the study on **structured vs. unstructured
pruning of CNN-based Remaining Useful Life (RUL) prediction models** on the NASA
C-MAPSS turbofan degradation dataset (FD001–FD004).

The experiment compares magnitude-based unstructured pruning
(`torch.nn.utils.prune.l1_unstructured`) against dependency-graph channel
pruning (`torch-pruning`) at ratios from 0 % to 90 %, before and after
fine-tuning, over 10 seeded runs, measuring prediction quality (RMSE, the
asymmetric C-MAPSS score) against deployment cost (model size, GPU memory,
inference latency, parameters, MACs, FLOPs).

Everything needed to repeat the experiments on your own infrastructure is in
this repository: a pinned Docker image, a launcher, and the four pipeline
stages. Only the dataset itself has to be downloaded separately.

## The study at a glance

| | |
| --- | --- |
| Model | Re-implementation of the DCNN of Li et al. (2018): 5 conv layers (4 × (10,1) with 10 channels, 1 × (3,1) with 1 channel), dropout 0.5, FC(100), FC(1) |
| Data | C-MAPSS FD001–FD004; 7 constant sensors and the 3 operational settings dropped, min-max scaled to [-1, 1]; RUL clipped at 125 cycles and normalised to [-1, 1] |
| Time windows | 30 / 20 / 30 / 15 for FD001–FD004 |
| Models evaluated | **big** = FD001, N_tw = 30 · **small** = FD004, N_tw = 15 |
| Baseline training | 250 epochs, Adam, lr 1e-3 (×0.1 after epoch 200), Xavier init |
| Pruning | L1 magnitude; unstructured on conv weights, structured on conv output channels; FC layers untouched |
| Ratios | 0 %, 10 %, … 90 % |
| Fine-tuning | 20 epochs, Adam, lr 1e-3 |
| Repetitions | 10 runs, seeded `42 + run` |
| Profiling | Batch of 512; memory from the PyTorch profiler, latency as median of 100 timed passes after 1000 warm-ups |

## Repository layout

```
src/                  The experiment code — four stages, run in order.
                      Colab notebook exports; deliberately frozen (see below).
compat/               Colab-to-container shims: google.colab, an offline
                      gspread client writing CSVs, CPU fallbacks for the
                      CUDA-only profiling calls, headless figure capture.
run_stage.py          Launcher: checks mounts, installs the shims, execs one
                      stage. `--check` reports the environment and exits.
Dockerfile            Pinned image; one build serves CPU and GPU.
docker-compose.yml    Two services (`cpu`, `gpu`) over that one image.
pyproject.toml        Dependency constraints.
uv.lock               Exact pins — what the image actually installs.
scripts/              Post-hoc figure scripts for the paper (not the pipeline).
results/figures/      Figures as they appear in the paper.
results/experiment_logs/   Outputs of our reproduction run: per-run CSVs,
                      aggregated CSVs with avg/std, and all evaluation plots.
```

`data/` is created by you and is git-ignored (see below). `main.py` is an
unused `uv init` stub.

## Reproducing the experiments

**Requirements:** Docker with Compose v2. For GPU runs additionally the NVIDIA
Container Toolkit and a host driver `>= 525`. Build for `linux/amd64` to get the
CUDA-enabled torch wheels; an `arm64` build resolves to CPU-only torch.

### 1. Get the data

The C-MAPSS dataset is not redistributed here. Download *CMAPSS Jet Engine
Simulated Data* from the [NASA Prognostics Data
Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) and
lay it out as:

```sh
mkdir -p data/CMAPSSData data/NEW_DATA data/NEW_MODELS results
# data/CMAPSSData/  <- train_FD001..4.txt, test_FD001..4.txt, RUL_FD001..4.txt
# data/NEW_DATA/    <- empty; stage 1 writes the preprocessed tensors here
# data/NEW_MODELS/  <- empty; stage 2 writes the trained baselines here
```

These three folders are mounted onto the original Google Drive paths inside the
container, which is why no path in `src/` has to change.

### 2. Build and check

```sh
docker compose build
docker compose run --rm cpu --check     # verifies mounts, torch, CUDA
```

### 3. Run the four stages

```sh
docker compose run --rm cpu setup_data           # 1: preprocess -> *_timed.pt
docker compose run --rm gpu setup_models         # 2: train baselines -> model{0..3}.pth
docker compose run --rm gpu pruning_experiments  # 3: prune, fine-tune, profile
docker compose run --rm cpu evaluation           # 4: plots
```

Substitute `cpu` for `gpu` anywhere if no GPU is available — but: without a GPU, `memory_mb` is always 0 and
`inference_ms` measures CPU latency. `rmse`, `score`, `size_mb`, `params`, `MACs` and `FLOPs` are expected to be device-independent.

## Our results

| Path | Contents |
| --- | --- |
| `results/experiment_logs/sheet_{1,2}_*.csv` | Raw result sink, one column per run |
| `results/experiment_logs/{un,}structured_pruning_data_3.csv` | The same data with `avg` and `std` across runs — the input to stage 4 |
| `results/experiment_logs/plots/` | 70 evaluation figures (metric × model × method) |
| `results/experiment_logs/downloads/` | Figures the notebooks exported via `files.download` |
| `results/figures` | Figures shown in the paper |


### Hardware we used

The results in this repository were produced by two AMD EPYC 7763 64-core processors at 2.45 GHz, 256 GB RAM, and two NVIDIA
A100 80 GB GPUs partitioned into MIG instances, under Ubuntu 22.04.5 LTS. We only used one 10GB MIG slice. The
experiments were originally run in Google Colab and repeated on this server for
reproducibility.

## Dependencies

`pyproject.toml` holds the constraints, `uv.lock` the exact pins; the image
installs with `uv sync --frozen`, which fails the build if the two disagree. To
change a dependency, edit `pyproject.toml`, run `uv lock`, and commit both.
