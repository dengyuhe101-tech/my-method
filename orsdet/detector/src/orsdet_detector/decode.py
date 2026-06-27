"""Decoding helpers for V4d slim OBB layouts.

V4d removes the duplicated PARAM_OBB_W/H heads from V4b. The OBB width and
height used for RotIoU, NMS and catalog diagnostics are the native YOLO size
branch (`x2-x1`, `y2-y1`). Two experiment layouts are supported:

- v4d_s:  parameters = flux, phys_bmaj, phys_bmin; angles = OBB theta + PA.
- v4d_sa: parameters = flux, phys_bmaj, phys_bmin; angles = shared theta/PA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orsdet_geometry.geometry import canonicalize_le90, hbb_from_obb, xywhtheta_to_corners
from orsdet_angle.angle_codec import decode_angle_vector
from orsdet_candidates.decode import denorm_log_value, pix_to_arcsec


V4D_MODE_SIZE = "v4d_s"
V4D_MODE_SIZE_ANGLE = "v4d_sa"
V4D_MODE_FLUX_REFINE_SIZE_ANGLE = "v4i_dsa"
V4D_MODE_V4M_NATIVE_SIZE_ANGLE = "v4m_native_dsa"
V4D_MODE_FLUX_CALIB_GATE_SIZE_ANGLE = "v4k_dsa"
V4D_MODES = (
    V4D_MODE_SIZE,
    V4D_MODE_SIZE_ANGLE,
    V4D_MODE_FLUX_REFINE_SIZE_ANGLE,
    V4D_MODE_V4M_NATIVE_SIZE_ANGLE,
    V4D_MODE_FLUX_CALIB_GATE_SIZE_ANGLE,
)
V4D_DEFAULT_MODE = V4D_MODE_SIZE

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
class V4DLayout:
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
    text = (mode or V4D_DEFAULT_MODE).strip().lower().replace("-", "_")
    aliases = {
        "s": V4D_MODE_SIZE,
        "size": V4D_MODE_SIZE,
        "size_only": V4D_MODE_SIZE,
        "v4d_s": V4D_MODE_SIZE,
        "sa": V4D_MODE_SIZE_ANGLE,
        "size_angle": V4D_MODE_SIZE_ANGLE,
        "shared_angle": V4D_MODE_SIZE_ANGLE,
        "v4d_sa": V4D_MODE_SIZE_ANGLE,
        "v4i_dsa": V4D_MODE_FLUX_REFINE_SIZE_ANGLE,
        "v4d_sa_flux_refine": V4D_MODE_FLUX_REFINE_SIZE_ANGLE,
        "flux_refine": V4D_MODE_FLUX_REFINE_SIZE_ANGLE,
        "v4m_native_dsa": V4D_MODE_V4M_NATIVE_SIZE_ANGLE,
        "v4m_native": V4D_MODE_V4M_NATIVE_SIZE_ANGLE,
        "native_flux_head": V4D_MODE_V4M_NATIVE_SIZE_ANGLE,
        "v4k_dsa": V4D_MODE_FLUX_CALIB_GATE_SIZE_ANGLE,
        "v4d_sa_flux_calib_gate": V4D_MODE_FLUX_CALIB_GATE_SIZE_ANGLE,
        "flux_calib_gate": V4D_MODE_FLUX_CALIB_GATE_SIZE_ANGLE,
    }
    if text in aliases:
        return aliases[text]
    raise ValueError("Unsupported V4d slim mode: %s" % mode)


def v4d_layout(mode: str | None = None) -> V4DLayout:
    slim_mode = normalize_slim_mode(mode)
    if slim_mode == V4D_MODE_FLUX_CALIB_GATE_SIZE_ANGLE:
        nb_param = 5
    elif slim_mode in (V4D_MODE_FLUX_REFINE_SIZE_ANGLE, V4D_MODE_V4M_NATIVE_SIZE_ANGLE):
        nb_param = 4
    else:
        nb_param = 3
    nb_angle = 4 if slim_mode == V4D_MODE_SIZE else 2
    target_stride = 7 + nb_param + nb_angle + 1
    return V4DLayout(
        mode=slim_mode,
        nb_param=nb_param,
        nb_angle=nb_angle,
        target_stride=target_stride,
        total_aux=nb_param + nb_angle,
        row_angle_start=ROW_PARAM_START + nb_param,
        angle_phys_pa_start=(
            ANGLE_PHYS_PA_START_SIZE if slim_mode == V4D_MODE_SIZE else ANGLE_PHYS_PA_START_SIZE_ANGLE
        ),
    )


# Backward-simple constants for the default V4d-S branch.
V4D_NB_PARAM = v4d_layout().nb_param
V4D_NB_ANGLE = v4d_layout().nb_angle
V4D_TARGET_STRIDE = v4d_layout().target_stride
V4D_TOTAL_AUX = v4d_layout().total_aux


def v4d_target_dim(max_nb_obj_per_image: int, mode: str | None = None) -> int:
    return 1 + int(max_nb_obj_per_image) * v4d_layout(mode).target_stride


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
    layout = v4d_layout(mode)
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    angle_vectors = rows[:, layout.row_angle_start + ANGLE_OBB_START : layout.row_angle_start + ANGLE_OBB_START + 2]
    return _native_obb_from_rows(rows[:, :4], angle_vectors)


def decode_target_box(block, lims=None, mode: str | None = None):
    """Decode one V4d target block to an OBB."""

    del lims
    layout = v4d_layout(mode)
    block = np.asarray(block, dtype=np.float64)
    rows = np.asarray([[block[1], block[2], block[4], block[5]]], dtype=np.float64)
    angle_vectors = np.asarray(
        [block[layout.row_angle_start + ANGLE_OBB_START : layout.row_angle_start + ANGLE_OBB_START + 2]],
        dtype=np.float64,
    )
    return _native_obb_from_rows(rows, angle_vectors)[0]


def v4d_catalog_arrays(
    flat_boxes,
    lims,
    pixel_size_deg: float,
    mode: str | None = None,
    flux_decode_mode: str = FLUX_DECODE_BASE,
    flux_delta_norm_scale: float = FLUX_REFINE_DEFAULT_DELTA_NORM_SCALE,
):
    """Decode final post-NMS rows for SDC1 catalog writing."""

    layout = v4d_layout(mode)
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
