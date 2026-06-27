"""Runtime helpers for V4d slim OBB + physical shape training."""

from __future__ import annotations

import sys
import types
import os
from pathlib import Path


V4D_DIR = Path(__file__).resolve().parents[2]
SKAO_DIR = V4D_DIR.parent
ROOT_DIR = SKAO_DIR.parent
CIANNA_DIR = ROOT_DIR / "src"
V25_DIR = SKAO_DIR / "candidates"
V2_DIR = SKAO_DIR / "angle"
V1_DIR = SKAO_DIR / "geometry"
V1A_DIR = SKAO_DIR / "target_source"
DEFAULT_TARGET_SOURCE = os.environ.get("CIANNA_V4D_TARGET_SOURCE", "v1a").strip().lower() or "v1a"
DEFAULT_V1_RUN_DIR = V4D_DIR / "outputs" / "train_geometry"
DEFAULT_V1A_RUN_DIR = V4D_DIR / "outputs" / "train_target_source"


def normalize_target_source(source: str | None) -> str:
    value = (source or DEFAULT_TARGET_SOURCE).strip().lower()
    if value not in ("v1", "v1a"):
        raise ValueError("target source must be 'v1' or 'v1a'.")
    return value


def target_table_path(source: str | None = None) -> Path:
    source = normalize_target_source(source)
    if source == "v1a":
        return V1A_DIR / "rotated_training_source_table.csv"
    return V1_DIR / "rotated_training_source_table.csv"


def default_run_dir_for_target_source(source: str | None = None, slim_mode: str | None = None) -> Path:
    from .decode import normalize_slim_mode

    source = normalize_target_source(source)
    mode = normalize_slim_mode(slim_mode)
    if source == "v1a":
        return V4D_DIR / "outputs" / ("train_target_source_%s" % mode)
    return V4D_DIR / "outputs" / ("train_geometry_%s" % mode)


DEFAULT_RUN_DIR = default_run_dir_for_target_source(DEFAULT_TARGET_SOURCE)


def configure_paths() -> None:
    for path in (
        V4D_DIR / "src",
        V25_DIR / "src",
        V2_DIR / "src",
        V1A_DIR / "src",
        V1_DIR / "src",
        SKAO_DIR,
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

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
