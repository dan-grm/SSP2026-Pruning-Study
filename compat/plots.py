"""Persist figures that the notebook code only ever displayed.

``src/evaluation.py`` draws its plots with ``download=False``, which ends in
``plt.show()``. In a notebook that renders the figure inline; under the
headless Agg backend it is a no-op, so a containerised run would compute every
plot and then throw all of them away.

This shim makes ``plt.show()`` write the current figure to
``$SSP_RESULTS_DIR/plots`` before closing it. Nothing about how the figures are
computed changes -- only that they end up on disk. Figures saved explicitly by
the notebook code (the ``download=True`` path) are unaffected and still go
through :mod:`compat.colab`.
"""

from __future__ import annotations

import re

from compat.paths import results_dir


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug[:120]


def install_plot_capture() -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg", force=True)
    target = results_dir() / "plots"
    counter = {"n": 0}

    def _show(*args, **kwargs) -> None:
        figure = plt.gcf()
        if not figure.get_axes():
            plt.close(figure)
            return
        counter["n"] += 1

        # Prefer the axes title so filenames match the plots in the paper.
        title = ""
        for axes in figure.get_axes():
            if axes.get_title():
                title = axes.get_title()
                break
        name = f"{counter['n']:03d}_{_slugify(title) or 'figure'}.png"

        target.mkdir(parents=True, exist_ok=True)
        path = target / name
        figure.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(figure)
        print(f"[compat] plt.show() -> {path}")

    plt.show = _show
    print(f"[compat] plot capture installed (figures -> {target})")
