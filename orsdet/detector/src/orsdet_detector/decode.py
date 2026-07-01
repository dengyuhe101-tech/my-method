"""Decoding helpers for detector OBB layouts.

The OBB width and height used for RotIoU, NMS and catalog diagnostics are the
native YOLO size branch (`x2-x1`, `y2-y1`). The public layouts are:

- size:  parameters = flux, phys_bmaj, phys_bmin; angles = OBB theta + PA.
- shared_angle: parameters = flux, phys_bmaj, phys_bmin; angles = shared theta/PA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orsdet_geometry.geometry import canonicalize_le90, hbb_from_obb, xywhtheta_to_corners
from orsdet_angle.angle_codec import decode_angle_vector
from orsdet_candidates.decode import denorm_log_value, pix_to_arcsec


DETECTOR_MODE_SIZE = "size"
DETECTOR_MODE_SHARED_ANGLE = "shared_angle"
DETECTOR_MODE_FLUX_REFINE = "flux_refine"
DETECTOR_MODE_NATIVE_FLUX_HEAD = "native_flux_head"
DETECTOR_MODE_FLUX_CALIB_GATE = "flux_calib_gate"
DETECTOR_MODES = (
    DETECTOR_MODE_SIZE,
    DETECTOR_MODE_SHARED_ANGLE,
    DETECTOR_MODE_FLUX_REFINE,
    DETECTOR_MODE_NATIVE_FLUX_HEAD,
    DETECTOR_MODE_FLUX_CALIB_GATE,
)
DETECTOR_DEFAULT_MODE = DETECTOR_MODE_SIZE

PARAM_FLUX = 0
PARAM_PHYS_BMAJ = 1
PARAM_PHYS_BMIN = 2
PARAM_FLUX_REFINE_DELTA = 3
PARAM_FLUX_REFINE_GATE = 4
BASE_NB_PARAM = 3
FLUX_REFINE_DEFAULT_DELTA_NORM_SCALE = 0.25

ANGLE_OBB_START = 0
ANGLE_PHYS_PA_START_SIZE = 2
ANGLE_PHYS_PA_START_SIZE_ANGLE = 0

ROW_PARAM_START = 7
FLUX_DECODE_BASE = "base"
FLUX_DECODE_FINAL_GATE1 = "final_gate1"
FLUX_DECODE_FINAL_GATE1_NOCLIP = "final_gate1_noclip"
FLUX_DECODE_FINAL_BRIGHT_GATE = "final_bright_gate"
FLUX_DECODE_FINAL_LEARNED_GATE = "final_learned_gate"
FLUX_DECODE_MODES = (
    FLUX_DECODE_BASE,
    FLUX_DECODE_FINAL_GATE1,
    FLUX_DECODE_FINAL_GATE1_NOCLIP,
    FLUX_DECODE_FINAL_BRIGHT_GATE,
    FLUX_DECODE_FINAL_LEARNED_GATE,
)


@dataclass(frozen=True)
class DetectorLayout:
    mode: str
    nb_param: int
    nb_angle: int
    target_stride: int
    total_aux: int
    row_angle_start: int
    angle_phys_pa_start: int

    @property
    def output_stride(self) -> int:
        return 8 + self.total_aux


def normalize_slim_mode(mode: str | None) -> str:
    text = (mode or DETECTOR_DEFAULT_MODE).strip().lower().replace("-", "_")
    aliases = {
        "s": DETECTOR_MODE_SIZE,
        "size": DETECTOR_MODE_SIZE,
        "size_only": DETECTOR_MODE_SIZE,
        "sa": DETECTOR_MODE_SHARED_ANGLE,
        "size_angle": DETECTOR_MODE_SHARED_ANGLE,
        "shared_angle": DETECTOR_MODE_SHARED_ANGLE,
        "flux_refine": DETECTOR_MODE_FLUX_REFINE,
        "shared_angle_flux_refine": DETECTOR_MODE_FLUX_REFINE,
        "native_flux_head": DETECTOR_MODE_NATIVE_FLUX_HEAD,
        "flux_calib_gate": DETECTOR_MODE_FLUX_CALIB_GATE,
        "shared_angle_flux_calib_gate": DETECTOR_MODE_FLUX_CALIB_GATE,
    }
    if text in aliases:
        return aliases[text]
    raise ValueError("Unsupported detector layout mode: %s" % mode)


def detector_layout(mode: str | None = None) -> DetectorLayout:
    slim_mode = normalize_slim_mode(mode)
    if slim_mode == DETECTOR_MODE_FLUX_CALIB_GATE:
        nb_param = 5
    elif slim_mode in (DETECTOR_MODE_FLUX_REFINE, DETECTOR_MODE_NATIVE_FLUX_HEAD):
        nb_param = 4
    else:
        nb_param = 3
    nb_angle = 4 if slim_mode == DETECTOR_MODE_SIZE else 2
    target_stride = 7 + nb_param + nb_angle + 1
    return DetectorLayout(
        mode=slim_mode,
        nb_param=nb_param,
        nb_angle=nb_angle,
        target_stride=target_stride,
        total_aux=nb_param + nb_angle,
        row_angle_start=ROW_PARAM_START + nb_param,
        angle_phys_pa_start=(
            ANGLE_PHYS_PA_START_SIZE if slim_mode == DETECTOR_MODE_SIZE else ANGLE_PHYS_PA_START_SIZE_ANGLE
        ),
    )


# Constants for the default detector size branch.
DETECTOR_NB_PARAM = detector_layout().nb_param
DETECTOR_NB_ANGLE = detector_layout().nb_angle
DETECTOR_TARGET_STRIDE = detector_layout().target_stride
DETECTOR_TOTAL_AUX = detector_layout().total_aux


def detector_target_dim(max_nb_obj_per_image: int, mode: str | None = None) -> int:
    return 1 + int(max_nb_obj_per_image) * detector_layout(mode).target_stride


def normalize_flux_decode_mode(mode: str | None) -> str:
    text = (mode or FLUX_DECODE_BASE).strip().lower().replace("-", "_")
    aliases = {
        "base": FLUX_DECODE_BASE,
        "flux_base": FLUX_DECODE_BASE,
        "final": FLUX_DECODE_FINAL_GATE1,
        "final_gate1": FLUX_DECODE_FINAL_GATE1,
        "gate1": FLUX_DECODE_FINAL_GATE1,
        "final_noclip": FLUX_DECODE_FINAL_GATE1_NOCLIP,
        "final_gate1_noclip": FLUX_DECODE_FINAL_GATE1_NOCLIP,
        "gate1_noclip": FLUX_DECODE_FINAL_GATE1_NOCLIP,
        "native_delta": FLUX_DECODE_FINAL_GATE1_NOCLIP,
        "bright_gate": FLUX_DECODE_FINAL_BRIGHT_GATE,
        "final_bright_gate": FLUX_DECODE_FINAL_BRIGHT_GATE,
        "bright_protected": FLUX_DECODE_FINAL_BRIGHT_GATE,
        "learned_gate": FLUX_DECODE_FINAL_LEARNED_GATE,
        "final_learned_gate": FLUX_DECODE_FINAL_LEARNED_GATE,
        "calib_gate": FLUX_DECODE_FINAL_LEARNED_GATE,
    }
    if text in aliases:
        return aliases[text]
    raise ValueError("Unsupported flux decode mode: %s" % mode)


def _flux_refine_norm(flat_boxes, lims, flux_decode_mode: str, delta_norm_scale: float):
    base_norm = np.asarray(flat_boxes[:, ROW_PARAM_START + PARAM_FLUX], dtype=np.float64)
    if flat_boxes.shape[1] <= ROW_PARAM_START + PARAM_FLUX_REFINE_DELTA:
        return base_norm

    mode = normalize_flux_decode_mode(flux_decode_mode)
    if mode == FLUX_DECODE_BASE:
        return base_norm

    delta_norm = (
        np.asarray(flat_boxes[:, ROW_PARAM_START + PARAM_FLUX_REFINE_DELTA], dtype=np.float64)
        * float(delta_norm_scale)
    )
    if mode == FLUX_DECODE_FINAL_LEARNED_GATE:
        if flat_boxes.shape[1] <= ROW_PARAM_START + PARAM_FLUX_REFINE_GATE:
            return base_norm
        gate = np.clip(
            np.asarray(flat_boxes[:, ROW_PARAM_START + PARAM_FLUX_REFINE_GATE], dtype=np.float64),
            0.0,
            1.0,
        )
        delta_norm = delta_norm * gate
    if mode == FLUX_DECODE_FINAL_BRIGHT_GATE:
        lims_row = np.asarray(lims[0], dtype=np.float64)
        flux_range = float(lims_row[0] - lims_row[1])
        base_flux = denorm_log_value(base_norm, lims_row)
        delta_log_flux = delta_norm * flux_range
        weights = np.ones_like(delta_norm)
        bright = base_flux >= 1.0e-5
        weights[bright & (delta_log_flux < -0.10)] = 0.75
        weights[bright & (delta_log_flux < -0.35)] = 0.25
        delta_norm = delta_norm * weights
    if mode == FLUX_DECODE_FINAL_GATE1_NOCLIP:
        return base_norm + delta_norm
    return np.clip(base_norm + delta_norm, 0.0, 1.0)


def _native_obb_from_rows(rows, angle_vectors):
    rows = np.asarray(rows, dtype=np.float64)
    centers_x = 0.5 * (rows[:, 0] + rows[:, 2])
    centers_y = 0.5 * (rows[:, 1] + rows[:, 3])
    w_pix = np.maximum(rows[:, 2] - rows[:, 0], 1.0e-8)
    h_pix = np.maximum(rows[:, 3] - rows[:, 1], 1.0e-8)
    theta = decode_angle_vector(angle_vectors, normalize=True)
    return np.stack(canonicalize_le90(centers_x, centers_y, w_pix, h_pix, theta), axis=-1)


def decode_rows_obb(rows, lims=None, mode: str | None = None):
    """Decode local post-process rows to OBBs for rotated NMS."""

    del lims
    layout = detector_layout(mode)
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    angle_vectors = rows[:, layout.row_angle_start + ANGLE_OBB_START : layout.row_angle_start + ANGLE_OBB_START + 2]
    return _native_obb_from_rows(rows[:, :4], angle_vectors)


def decode_target_box(block, lims=None, mode: str | None = None):
    """Decode one detector target block to an OBB."""

    del lims
    layout = detector_layout(mode)
    block = np.asarray(block, dtype=np.float64)
    rows = np.asarray([[block[1], block[2], block[4], block[5]]], dtype=np.float64)
    angle_vectors = np.asarray(
        [block[layout.row_angle_start + ANGLE_OBB_START : layout.row_angle_start + ANGLE_OBB_START + 2]],
        dtype=np.float64,
    )
    return _native_obb_from_rows(rows, angle_vectors)[0]


def detector_catalog_arrays(
    flat_boxes,
    lims,
    pixel_size_deg: float,
    mode: str | None = None,
    flux_decode_mode: str = FLUX_DECODE_BASE,
    flux_delta_norm_scale: float = FLUX_REFINE_DEFAULT_DELTA_NORM_SCALE,
):
    """Decode final post-NMS rows for SDC1 catalog writing."""

    layout = detector_layout(mode)
    flat_boxes = np.asarray(flat_boxes, dtype=np.float64)
    obb = decode_rows_obb(flat_boxes, lims, layout.mode)
    corners = xywhtheta_to_corners(obb)
    hbb = hbb_from_obb(obb)
    flux_norm = _flux_refine_norm(flat_boxes, lims, flux_decode_mode, flux_delta_norm_scale)
    flux_jy = denorm_log_value(flux_norm, lims[0])
    bmaj_arcsec = denorm_log_value(flat_boxes[:, ROW_PARAM_START + PARAM_PHYS_BMAJ], lims[1])
    bmin_arcsec = denorm_log_value(flat_boxes[:, ROW_PARAM_START + PARAM_PHYS_BMIN], lims[2])
    pa_start = layout.row_angle_start + layout.angle_phys_pa_start
    pa_deg = decode_angle_vector(flat_boxes[:, pa_start : pa_start + 2], normalize=True)
    obb_bmaj_arcsec = pix_to_arcsec(obb[:, 2], pixel_size_deg)
    obb_bmin_arcsec = pix_to_arcsec(obb[:, 3], pixel_size_deg)
    aspect = obb[:, 2] / np.maximum(obb[:, 3], 1.0e-8)
    return {
        "obb": obb,
        "corners": corners,
        "hbb": hbb,
        "flux_jy": flux_jy,
        "bmaj_arcsec": bmaj_arcsec,
        "bmin_arcsec": bmin_arcsec,
        "pa_deg": pa_deg,
        "obb_bmaj_arcsec": obb_bmaj_arcsec,
        "obb_bmin_arcsec": obb_bmin_arcsec,
        "aspect_ratio": aspect,
    }
