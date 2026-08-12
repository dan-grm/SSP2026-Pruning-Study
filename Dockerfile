# Reproducibility image for the C-MAPSS pruning study.
#
# One image serves both targets: the default PyPI torch wheels for
# linux/amd64 bundle the CUDA 12.8 runtime and fall back to CPU when no GPU is
# passed in, so `docker run` and `docker run --gpus all` use the same build.
#
#   docker build -t ssp2026-pruning .
#   docker run --rm ssp2026-pruning --check
#
# Build for linux/amd64 if you want the CUDA-enabled wheels; an arm64 build
# resolves to CPU-only torch (the lockfile covers both platforms).

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# libgomp1: OpenMP runtime for the CPU kernels in torch/scikit-learn.
# ca-certificates: TLS roots, so a run can still reach the network if needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependencies first: this layer only rebuilds when the lockfile changes.
# --frozen fails the build if uv.lock and pyproject.toml disagree, which is
# what makes the image reproducible rather than "whatever resolved today".
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY compat ./compat
COPY src ./src
COPY run_stage.py ./

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    SSP_RESULTS_DIR=/work/results

# The notebook code addresses Google Drive by absolute path. Recreating that
# tree in the image means the host can mount its data onto it and not a single
# path in src/ has to change.
RUN mkdir -p "/content/drive/MyDrive/Bachelor/new coding/CMAPSSData" \
             "/content/drive/MyDrive/Bachelor/new coding/NEW_DATA" \
             "/content/drive/MyDrive/Bachelor/new coding/NEW_MODELS" \
             /work/results

WORKDIR /work

ENTRYPOINT ["python", "/app/run_stage.py"]
CMD ["--check"]
