from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np


def install_numpy_core_aliases() -> None:
    """Allow NumPy 1.x to load object arrays pickled by NumPy 2.x."""
    try:
        numpy_core = importlib.import_module("numpy.core")
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for name in (
        "multiarray",
        "numeric",
        "fromnumeric",
        "numerictypes",
        "umath",
        "_multiarray_umath",
    ):
        try:
            module = importlib.import_module(f"numpy.core.{name}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", module)


def load_pickle_npy(path: str | Path) -> Any:
    install_numpy_core_aliases()
    return np.load(path, allow_pickle=True)
