"""candidate OBB helpers."""

from .data import CandidateDataBuilder
from .decode import (
    CANDIDATE_NB_ANGLE,
    CANDIDATE_NB_PARAM,
    CANDIDATE_TARGET_STRIDE,
    arcsec_to_pix,
    decode_obb_boxes,
    decode_target_box,
    denorm_log_value,
    obb_catalog_arrays,
    pix_to_arcsec,
    candidate_target_dim,
)
from .runtime import DEFAULT_RUN_DIR, CANDIDATE_DIR, configure_paths, install_numba_fallback_if_needed

__all__ = [
    "DEFAULT_RUN_DIR",
    "CandidateDataBuilder",
    "CANDIDATE_DIR",
    "CANDIDATE_NB_ANGLE",
    "CANDIDATE_NB_PARAM",
    "CANDIDATE_TARGET_STRIDE",
    "arcsec_to_pix",
    "configure_paths",
    "decode_obb_boxes",
    "decode_target_box",
    "denorm_log_value",
    "install_numba_fallback_if_needed",
    "obb_catalog_arrays",
    "pix_to_arcsec",
    "candidate_target_dim",
]
