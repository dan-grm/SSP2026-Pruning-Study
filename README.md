# ssp2026-pruning

Code and reproduction package for the study on **structured vs. unstructured
pruning of CNN-based Remaining Useful Life (RUL) prediction models** on the NASA
C-MAPSS turbofan degradation dataset (FD001–FD004).

The experiment compares magnitude-based unstructured pruning
(`torch.nn.utils.prune`) against dependency-graph channel pruning
(`torch-pruning`) at ratios from 0 % to 90 %, before and after fine-tuning,
over 10 seeded runs, measuring prediction quality (RMSE, the asymmetric C-MAPSS
score) against cost (model size, GPU memory, inference latency, parameters,
MACs, FLOPs).

## Layout

| Path | Contents |
| --- | --- |
| `src/` | The experiment code. |
| `compat/` | Colab-to-container shims. |
| `run_stage.py` | Launcher that runs one pipeline stage outside Colab. |
| `Dockerfile`, `docker-compose.yml` | Reproducibility image (CPU and GPU). |

## Quick start

```sh
mkdir -p data/CMAPSSData data/NEW_DATA data/NEW_MODELS results
# place the C-MAPSS txt files in data/CMAPSSData
docker compose build
docker compose run --rm cpu --check
```

## Data

The C-MAPSS dataset is not redistributed in this repository. Obtain *CMAPSS Jet Engine Simulated Data* from the [NASA Prognostics Data
Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
