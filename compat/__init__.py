"""Colab-to-container compatibility layer.

The scripts in ``src/`` are verbatim exports of the Google Colab notebooks used
for the bachelor thesis. They are deliberately **not** modified: reproducing the
published numbers means running the same code. Everything that Colab used to
provide -- the ``google.colab`` module, an authenticated Google Sheets client,
an always-present CUDA device -- is emulated here instead, and installed by
``run_stage.py`` before the notebook code is executed.

Nothing in this package is imported by ``src/``; the direction is one-way.
"""

from compat.colab import install_colab_shim
from compat.cuda import install_cuda_fallback
from compat.plots import install_plot_capture
from compat.results import dump_result_frames
from compat.sheets import install_sheets_shim

__all__ = [
    "install_colab_shim",
    "install_cuda_fallback",
    "install_plot_capture",
    "install_sheets_shim",
    "dump_result_frames",
]
