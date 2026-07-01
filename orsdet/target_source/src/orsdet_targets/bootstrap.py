"""Path helpers for running target_source as a small project."""

from __future__ import annotations

from pathlib import Path


TARGET_SOURCE_ROOT = Path(__file__).resolve().parents[2]
SKAO_SDC1_DIR = TARGET_SOURCE_ROOT.parent
CIANNA_NEW_ROOT = TARGET_SOURCE_ROOT.parents[2]
DEFAULT_OUTPUT_DIR = TARGET_SOURCE_ROOT / "outputs"
