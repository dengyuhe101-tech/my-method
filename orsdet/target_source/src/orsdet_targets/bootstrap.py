"""Path helpers for running target_source as a small project."""

from __future__ import annotations

from pathlib import Path


V1A_OBB_ROOT = Path(__file__).resolve().parents[2]
SKAO_SDC1_DIR = V1A_OBB_ROOT.parent
CIANNA_NEW_ROOT = V1A_OBB_ROOT.parents[2]
DEFAULT_OUTPUT_DIR = V1A_OBB_ROOT / "outputs"
