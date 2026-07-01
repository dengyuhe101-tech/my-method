"""Runtime helpers for rotated NMS rotated NMS."""

from __future__ import annotations

import sys
import types
from pathlib import Path


NMS_DIR = Path(__file__).resolve().parents[2]
SKAO_DIR = NMS_DIR.parent
ROOT_DIR = SKAO_DIR.parent
CIANNA_DIR = ROOT_DIR / "src"
CANDIDATE_DIR = SKAO_DIR / "candidates"
ANGLE_DIR = SKAO_DIR / "angle"
GEOMETRY_DIR = SKAO_DIR / "geometry"
DEFAULT_SRC_RUN_DIR = CANDIDATE_DIR / "outputs" / "train_report"
DEFAULT_OUT_DIR = NMS_DIR / "outputs" / "eval"


def configure_paths() -> None:
    sys.path.insert(0, str(NMS_DIR / "src"))
    sys.path.insert(0, str(CANDIDATE_DIR / "src"))
    sys.path.insert(0, str(ANGLE_DIR / "src"))
    sys.path.insert(0, str(GEOMETRY_DIR / "src"))
    sys.path.insert(0, str(SKAO_DIR))

    preferred = CIANNA_DIR / "build" / "lib.cianna4090-cuda" / "CIANNA.so"
    if preferred.is_file():
        sys.path.insert(0, str(preferred.parent))
        return

    build_libs = sorted(
        (CIANNA_DIR / "build").glob("lib.*/CIANNA.so"),
        key=lambda path: path.stat().st_mtime,
    )
    if build_libs:
        sys.path.insert(0, str(build_libs[-1].parent))


def install_numba_fallback_if_needed() -> None:
    try:
        import numba  # noqa: F401
        return
    except ImportError:
        pass

    def jit(*jit_args, **jit_kwargs):
        if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
            return jit_args[0]

        def decorator(func):
            return func

        return decorator

    stub = types.ModuleType("numba")
    stub.jit = jit
    sys.modules["numba"] = stub
