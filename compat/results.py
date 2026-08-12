"""Turn the experiment's in-memory result tables into analysis-ready CSVs.

In Colab the hand-off between stage 3 and stage 4 was manual: the metrics went
into a Google Sheet, ``avg`` and ``std`` columns were added there with
spreadsheet formulas, and the sheet was exported as
``structured_pruning_data_3.csv`` / ``unstructured_pruning_data_3.csv`` -- the
files ``src/evaluation.py`` reads.

That manual step is the one thing a container cannot inherit, so it is
reproduced here. After ``src/pruning_experiments.py`` finishes, ``run_stage.py``
hands its module globals to :func:`dump_result_frames`, which writes the same
two CSVs, with ``avg``/``std`` computed across the ``run_*`` columns.

This does not change how any metric is computed -- it only aggregates and
serialises the DataFrames the experiment code already built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from compat.paths import results_dir

#: DataFrame global in src/pruning_experiments.py -> filename src/evaluation.py reads.
FRAME_EXPORTS = {
    "unstructured_df": "unstructured_pruning_data_3.csv",
    "structured_df": "structured_pruning_data_3.csv",
}


def dump_result_frames(namespace: Mapping[str, Any]) -> list[Path]:
    """Write the known result DataFrames from ``namespace`` to CSV."""
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a hard dependency
        return []

    target_dir = results_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for global_name, filename in FRAME_EXPORTS.items():
        frame = namespace.get(global_name)
        if not isinstance(frame, pd.DataFrame):
            continue

        export = frame.copy()
        run_columns = [column for column in export.columns if column.startswith("run_")]
        if run_columns:
            numeric = export[run_columns].apply(pd.to_numeric, errors="coerce")
            # ddof=1: sample standard deviation across runs, matching the
            # spreadsheet's STDEV() that produced the published error bars.
            export["avg"] = numeric.mean(axis=1)
            export["std"] = numeric.std(axis=1, ddof=1)

        path = target_dir / filename
        export.to_csv(path, index=False)
        written.append(path)
        filled = int(numeric.notna().sum().sum()) if run_columns else 0
        print(f"[compat] wrote {path} ({len(export)} rows, {filled} run values)")

    if not written:
        print("[compat] no result DataFrames found to export")
    return written
