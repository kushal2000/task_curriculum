from __future__ import annotations

import ctypes
import importlib
from importlib.metadata import version
from pathlib import Path


SUPPORTED_FLEXIVRDK = {"1.9.1"}


def _flexivrdk_extension(flexivrdk) -> Path:
    package_dir = Path(flexivrdk.__file__).resolve().parent
    candidates = sorted(package_dir.glob("flexivrdk*.so"))
    if not candidates:
        raise RuntimeError(f"Could not find flexivrdk extension in {package_dir}")
    return candidates[0]


def load_rt():
    flexivrdk = importlib.import_module("flexivrdk")
    installed = version("flexivrdk")
    if installed not in SUPPORTED_FLEXIVRDK:
        supported = ", ".join(sorted(SUPPORTED_FLEXIVRDK))
        raise RuntimeError(
            f"aioflexiv supports flexivrdk versions [{supported}], but found {installed}"
        )

    ctypes.CDLL(str(_flexivrdk_extension(flexivrdk)), mode=ctypes.RTLD_GLOBAL)
    return importlib.import_module("aioflexiv._aioflexiv_rt")
