"""Offline replacement for the Google Sheets result sink.

``src/pruning_experiments.py`` streams its per-run metrics into two Google
Sheets via ``gspread``. That requires an interactive OAuth flow, and the two
spreadsheet URLs were stripped from the published code anyway (both are the
placeholder ``"-"``).

This module swaps ``gspread.authorize`` for a client whose spreadsheets are
CSV files on disk. The experiment code is untouched and still "writes to a
sheet"; the sheet just lives in ``$SSP_RESULTS_DIR``.

The semantics that matter are preserved: ``range()`` hands back cells carrying
the current value, and ``update_sheet`` in the experiment code only fills cells
whose value is falsy -- so re-running a run number does not overwrite results
that are already there, exactly as with the real spreadsheet.

These CSVs are an audit trail of what was written. The analysis-ready tables
are produced by :mod:`compat.results` from the in-memory DataFrames.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from compat.paths import results_dir

_A1_CELL = re.compile(r"^([A-Za-z]+)([0-9]+)$")


def _col_to_index(letters: str) -> int:
    """'A' -> 1, 'B' -> 2, ... 'AA' -> 27."""
    index = 0
    for char in letters.upper():
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index


def _parse_a1(a1: str) -> tuple[int, int, int, int]:
    """Parse ``'B2:K181'`` into ``(first_row, first_col, last_row, last_col)``."""
    if ":" not in a1:
        a1 = f"{a1}:{a1}"
    start, end = a1.split(":", 1)
    start_match, end_match = _A1_CELL.match(start.strip()), _A1_CELL.match(end.strip())
    if not start_match or not end_match:
        raise ValueError(f"[compat] unsupported A1 range: {a1!r}")
    return (
        int(start_match.group(2)),
        _col_to_index(start_match.group(1)),
        int(end_match.group(2)),
        _col_to_index(end_match.group(1)),
    )


class LocalWorksheet:
    """A worksheet backed by a CSV file."""

    def __init__(self, path: Path, title: str = "Sheet1") -> None:
        self.path = path
        self.title = title
        self._cells: dict[tuple[int, int], object] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(newline="", encoding="utf-8") as handle:
            for row_index, row in enumerate(csv.reader(handle), start=1):
                for col_index, value in enumerate(row, start=1):
                    if value != "":
                        self._cells[(row_index, col_index)] = value

    def _flush(self) -> None:
        if not self._cells:
            return
        max_row = max(row for row, _ in self._cells)
        max_col = max(col for _, col in self._cells)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for row in range(1, max_row + 1):
                writer.writerow(
                    [self._cells.get((row, col), "") for col in range(1, max_col + 1)]
                )

    def range(self, a1: str) -> list:
        from gspread.cell import Cell

        first_row, first_col, last_row, last_col = _parse_a1(a1)
        return [
            Cell(row, col, self._cells.get((row, col)))
            for row in range(first_row, last_row + 1)
            for col in range(first_col, last_col + 1)
        ]

    def update_cells(self, cell_list, *args, **kwargs) -> dict:
        written = 0
        for cell in cell_list:
            if cell.value is None or cell.value == "":
                continue
            self._cells[(cell.row, cell.col)] = cell.value
            written += 1
        self._flush()
        print(f"[compat] sheet '{self.title}': wrote {written} cells -> {self.path}")
        return {"updatedCells": written}

    # Convenience accessors used by neither stage today, but cheap to provide
    # so a future notebook cell does not hit an AttributeError.
    def get_all_values(self) -> list[list[str]]:
        if not self._cells:
            return []
        max_row = max(row for row, _ in self._cells)
        max_col = max(col for _, col in self._cells)
        return [
            [str(self._cells.get((row, col), "")) for col in range(1, max_col + 1)]
            for row in range(1, max_row + 1)
        ]

    def update_cell(self, row: int, col: int, value) -> None:
        self._cells[(row, col)] = value
        self._flush()


class LocalSpreadsheet:
    def __init__(self, path: Path, title: str) -> None:
        self.title = title
        self._worksheet = LocalWorksheet(path, title=title)

    @property
    def sheet1(self) -> LocalWorksheet:
        return self._worksheet

    def get_worksheet(self, index: int) -> LocalWorksheet:
        if index != 0:
            raise IndexError("[compat] offline spreadsheets have a single worksheet")
        return self._worksheet

    def worksheet(self, title: str) -> LocalWorksheet:
        return self._worksheet


#: The published code opens both spreadsheets with the same placeholder URL
#: ("-"), so the URL cannot tell them apart -- the open order does. In
#: src/pruning_experiments.py the unstructured sheet is opened first.
OPEN_ORDER_NAMES = ("unstructured", "structured")


class LocalClient:
    """Drop-in for ``gspread.Client`` that never touches the network."""

    def __init__(self) -> None:
        self._order = 0

    def _open(self, key: str) -> LocalSpreadsheet:
        # Deliberately not cached by URL: two open() calls are two distinct
        # spreadsheets, as they were in Colab. Caching on the placeholder URL
        # would merge the structured and unstructured results into one grid.
        self._order += 1
        if self._order <= len(OPEN_ORDER_NAMES):
            name = OPEN_ORDER_NAMES[self._order - 1]
        else:
            name = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "sheet"
        title = f"sheet_{self._order}_{name}"
        path = results_dir() / f"{title}.csv"
        spreadsheet = LocalSpreadsheet(path, title=title)
        print(f"[compat] gspread.open_by_url({key!r}) -> {path}")
        return spreadsheet

    def open_by_url(self, url: str) -> LocalSpreadsheet:
        return self._open(url)

    def open_by_key(self, key: str) -> LocalSpreadsheet:
        return self._open(key)

    def open(self, title: str) -> LocalSpreadsheet:
        return self._open(title)


def install_sheets_shim() -> None:
    """Point ``gspread.authorize`` at the offline client."""
    import gspread

    def _offline_authorize(*args, **kwargs) -> LocalClient:
        return LocalClient()

    gspread.authorize = _offline_authorize
    print(f"[compat] gspread shim installed (sheets -> {results_dir()})")
