#!/usr/bin/env python3
"""Run one stage of the pruning study outside of Google Colab.

The scripts in ``src/`` are unmodified Colab notebook exports. This launcher
makes them executable in a plain Python process:

1. it installs the :mod:`compat` shims (``google.colab``, an offline Google
   Sheets client, CPU fallbacks for the CUDA-only profiling calls);
2. it strips the IPython ``!pip install`` lines, which are not valid Python --
   the packages they install are pinned in ``pyproject.toml``/``uv.lock``
   instead. Line numbers are preserved so tracebacks still point at the right
   line of the original file;
3. it executes the file as ``__main__`` with its own path as ``__file__``;
4. after the experiment stage it exports the result DataFrames to the CSVs the
   evaluation stage expects.

The files under ``src/`` are never rewritten on disk.

Usage:
    python run_stage.py setup_data
    python run_stage.py pruning_experiments
    python run_stage.py --check          # verify mounts and environment only
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compat.paths import STAGE_REQUIREMENTS, drive_dirs, results_dir  # noqa: E402

STAGES = {
    "setup_data": REPO_ROOT / "src" / "setup_data.py",
    "setup_models": REPO_ROOT / "src" / "setup_models.py",
    "pruning_experiments": REPO_ROOT / "src" / "pruning_experiments.py",
    "evaluation": REPO_ROOT / "src" / "evaluation.py",
}

_MAGIC_LINE = re.compile(r"^(\s*)([!%].*)$")


def strip_ipython_magics(source: str) -> tuple[str, list[tuple[int, str]]]:
    """Replace IPython magic/shell lines with ``pass``, keeping line numbers.

    Only applied when the file does not compile as-is, so a ``!`` inside a
    string literal is never touched.
    """
    try:
        compile(source, "<probe>", "exec")
        return source, []
    except SyntaxError:
        pass

    lines = source.splitlines(keepends=True)
    stripped: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _MAGIC_LINE.match(line.rstrip("\r\n"))
        if not match:
            continue
        indent, magic = match.groups()
        stripped.append((index + 1, magic.strip()))
        newline = "\n" if line.endswith("\n") else ""
        lines[index] = f"{indent}pass  # [compat] stripped: {magic.strip()}{newline}"
    return "".join(lines), stripped


def preflight(stage: str) -> list[str]:
    """Return human-readable problems that would make the stage fail."""
    problems: list[str] = []
    directories = drive_dirs()
    for key in STAGE_REQUIREMENTS.get(stage, ()):
        path = directories[key]
        if not path.is_dir():
            problems.append(f"missing {key}: {path} is not a directory (mount it)")
    if stage == "setup_data":
        raw = directories["DRIVE_DIR"]
        if raw.is_dir() and not any(raw.glob("train_FD00*.txt")):
            problems.append(
                f"{raw} contains no train_FD00*.txt; mount the C-MAPSS dataset there"
            )
    return problems


def describe_environment() -> None:
    import torch

    print(f"[run] python           {sys.version.split()[0]}")
    print(f"[run] torch            {torch.__version__}")
    print(f"[run] cuda available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[run] cuda device      {torch.cuda.get_device_name(0)}")
    print(f"[run] results dir      {results_dir()}")
    for key, path in drive_dirs().items():
        marker = "ok     " if path.is_dir() else "MISSING"
        print(f"[run] {key:<16} {marker} {path}")


def run_stage(stage: str) -> int:
    from compat import (
        dump_result_frames,
        install_colab_shim,
        install_cuda_fallback,
        install_plot_capture,
        install_sheets_shim,
    )

    path = STAGES[stage]
    source = path.read_text(encoding="utf-8")
    source, stripped = strip_ipython_magics(source)
    for line_number, magic in stripped:
        print(f"[compat] {path.name}:{line_number} stripped `{magic}` (pinned in uv.lock)")

    install_colab_shim()
    install_sheets_shim()
    install_cuda_fallback()
    install_plot_capture()

    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(path),
        "__builtins__": __builtins__,
    }

    print(f"[run] executing {path.relative_to(REPO_ROOT)}")
    code = compile(source, str(path), "exec")
    try:
        exec(code, namespace)
    except Exception:
        traceback.print_exc()
        if stage == "pruning_experiments":
            # Salvage whatever runs completed before the failure.
            dump_result_frames(namespace)
        return 1

    if stage == "pruning_experiments":
        dump_result_frames(namespace)
    print(f"[run] stage '{stage}' finished")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "stage",
        nargs="?",
        choices=sorted(STAGES),
        help="pipeline stage to execute (in order: setup_data, setup_models, "
             "pruning_experiments, evaluation)",
    )
    parser.add_argument(
        "--results-dir",
        default=os.environ.get("SSP_RESULTS_DIR"),
        help="where sheets, CSVs and plots are written (default: $SSP_RESULTS_DIR "
             "or ./results)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report environment and mounts, then exit without running a stage",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even if the preflight check reports missing inputs",
    )
    args = parser.parse_args(argv)

    # Resolve the results directory once and export it, so the compat modules
    # agree on it after the chdir below.
    target = Path(args.results_dir) if args.results_dir else Path.cwd() / "results"
    target.mkdir(parents=True, exist_ok=True)
    os.environ["SSP_RESULTS_DIR"] = str(target.resolve())

    # src/evaluation.py reads its input CSVs from the working directory and
    # writes plots there, so every stage runs with the results directory as cwd.
    os.chdir(target)

    # Headless container: no interactive matplotlib backend available.
    os.environ.setdefault("MPLBACKEND", "Agg")

    if args.check or not args.stage:
        describe_environment()
        if not args.stage:
            print("\n[run] no stage given; nothing executed")
            return 0
        problems = preflight(args.stage)
        for problem in problems:
            print(f"[run] PROBLEM: {problem}")
        return 1 if problems else 0

    problems = preflight(args.stage)
    if problems:
        for problem in problems:
            print(f"[run] PROBLEM: {problem}")
        if not args.force:
            print("[run] aborting; pass --force to run anyway")
            return 2

    describe_environment()
    return run_stage(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
