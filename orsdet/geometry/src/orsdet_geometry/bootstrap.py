"""Path helpers for running geometry as a small project."""

from __future__ import annotations

from pathlib import Path


geometry_OBB_ROOT = Path(__file__).resolve().parents[2]
SKAO_SDC1_DIR = geometry_OBB_ROOT.parent
DEFAULT_OUTPUT_DIR = geometry_OBB_ROOT / "outputs"
