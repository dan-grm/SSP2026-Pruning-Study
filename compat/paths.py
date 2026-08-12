"""Directory layout for a containerised run.

The notebook code addresses Google Drive with absolute paths::

    DRIVE_DIR = '/content/drive/MyDrive/Bachelor/new coding/CMAPSSData/'
    DATA_DIR  = '/content/drive/MyDrive/Bachelor/new coding/NEW_DATA/'
    MODEL_DIR = '/content/drive/MyDrive/Bachelor/new coding/NEW_MODELS/'

Those constants are left exactly as they are. The container recreates that tree
as real directories and the host mounts its own folders onto them, so no path
in ``src/`` has to change. :func:`drive_dirs` exists so ``run_stage.py`` can
check the mounts up front and fail with a useful message instead of a
``FileNotFoundError`` deep inside preprocessing.

Outputs (sheets, plots, "downloads") go to ``$SSP_RESULTS_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

COLAB_ROOT = Path("/content/drive/MyDrive/Bachelor/new coding")

#: Drive folders referenced by ``src/``, keyed by the constant that names them.
DRIVE_DIRS = {
    "DRIVE_DIR": COLAB_ROOT / "CMAPSSData",
    "DATA_DIR": COLAB_ROOT / "NEW_DATA",
    "MODEL_DIR": COLAB_ROOT / "NEW_MODELS",
}

#: Which of those each stage reads from / writes to.
STAGE_REQUIREMENTS = {
    "setup_data": ("DRIVE_DIR", "DATA_DIR"),
    "setup_models": ("DATA_DIR", "MODEL_DIR"),
    "pruning_experiments": ("DATA_DIR", "MODEL_DIR"),
    "evaluation": (),
}


def drive_dirs() -> dict[str, Path]:
    return dict(DRIVE_DIRS)


def results_dir() -> Path:
    return Path(os.environ.get("SSP_RESULTS_DIR", Path.cwd() / "results"))


def download_dir() -> Path:
    configured = os.environ.get("SSP_DOWNLOAD_DIR")
    if configured:
        return Path(configured)
    return results_dir() / "downloads"
