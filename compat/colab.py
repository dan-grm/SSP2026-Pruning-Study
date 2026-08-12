"""Stand-in for the ``google.colab`` module.

``src/setup_data.py`` and ``src/evaluation.py`` do ``from google.colab import
files`` and call ``files.download(name)`` to pull a rendered plot out of the
notebook. ``src/pruning_experiments.py`` additionally calls
``auth.authenticate_user()`` and ``google.auth.default()`` to obtain Sheets
credentials.

In a container there is no browser to download to and no interactive OAuth
flow, so:

* ``files.download(path)`` copies the file into the downloads directory
  (``$SSP_DOWNLOAD_DIR``, default ``<cwd>/downloads``);
* ``auth.authenticate_user()`` is a no-op;
* ``google.auth.default()`` returns a placeholder credential that is only ever
  handed to the offline Sheets shim in :mod:`compat.sheets`.
"""

from __future__ import annotations

import importlib.machinery
import shutil
import sys
import types
from pathlib import Path

from compat.paths import download_dir


class _PlaceholderCredentials:
    """Marker object standing in for Google application-default credentials."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<offline placeholder credentials>"


def _new_module(name: str, is_package: bool = False) -> types.ModuleType:
    """Create a module that survives ``importlib.util.find_spec`` probes.

    A bare ``types.ModuleType`` has ``__spec__ = None``, which makes
    ``find_spec`` raise ``ValueError``. Third-party packages probe for
    ``google.colab`` exactly that way (``accelerate`` does, and it is pulled in
    by ``calflops``), so the shim modules need a real spec.
    """
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
    module.__loader__ = None
    if is_package:
        module.__path__ = []
    return module


def _make_files_module() -> types.ModuleType:
    module = _new_module("google.colab.files")

    def download(filename: str) -> None:
        source = Path(filename)
        if not source.exists():
            print(f"[compat] files.download: {source} does not exist, skipping")
            return
        target_dir = download_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        print(f"[compat] files.download -> {target}")

    def upload() -> dict[str, bytes]:
        raise RuntimeError(
            "google.colab.files.upload() is interactive and unavailable in the "
            "container; mount the input files instead."
        )

    module.download = download
    module.upload = upload
    return module


def _make_auth_module() -> types.ModuleType:
    module = _new_module("google.colab.auth")

    def authenticate_user(*args, **kwargs) -> None:
        print("[compat] auth.authenticate_user(): no-op (offline run)")

    module.authenticate_user = authenticate_user
    return module


def install_colab_shim() -> None:
    """Register the fake ``google.colab`` package and patch ``google.auth``."""
    if "google.colab" in sys.modules:
        return

    import google  # provided by google-auth as a namespace package

    colab = _new_module("google.colab", is_package=True)
    files_module = _make_files_module()
    auth_module = _make_auth_module()
    colab.files = files_module
    colab.auth = auth_module

    sys.modules["google.colab"] = colab
    sys.modules["google.colab.files"] = files_module
    sys.modules["google.colab.auth"] = auth_module
    google.colab = colab

    # `creds, _ = default()` in src/pruning_experiments.py would otherwise fail
    # with DefaultCredentialsError; the credentials are never actually used
    # because the Sheets client is replaced too.
    import google.auth

    def _offline_default(*args, **kwargs):
        return _PlaceholderCredentials(), None

    google.auth.default = _offline_default
    print("[compat] google.colab shim installed")
