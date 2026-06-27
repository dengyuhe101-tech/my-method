"""V3 rotated-NMS helpers."""

from .nms import (
    FIRST_IOU_THRESHOLDS,
    FIRST_OBJ_THRESHOLDS,
    SECOND_IOU_THRESHOLD,
    decode_rows_obb,
    local_nms,
    merge_nms,
)
from .runtime import DEFAULT_OUT_DIR, DEFAULT_SRC_RUN_DIR, V3_DIR, configure_paths, install_numba_fallback_if_needed

__all__ = [
    "DEFAULT_OUT_DIR",
    "DEFAULT_SRC_RUN_DIR",
    "FIRST_IOU_THRESHOLDS",
    "FIRST_OBJ_THRESHOLDS",
    "SECOND_IOU_THRESHOLD",
    "V3_DIR",
    "configure_paths",
    "decode_rows_obb",
    "install_numba_fallback_if_needed",
    "local_nms",
    "merge_nms",
]
