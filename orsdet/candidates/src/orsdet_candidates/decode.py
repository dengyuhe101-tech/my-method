"""Decoding helpers for V2.5 OBB targets and predictions."""

from __future__ import annotations

import numpy as np

from orsdet_geometry.geometry import canonicalize_le90, hbb_from_obb, xywhtheta_to_corners
from orsdet_angle.angle_codec import decode_angle_vector


V25_NB_PARAM = 3
V25_NB_ANGLE = 2
V25_TARGET_STRIDE = 7 + V25_NB_PARAM + V25_NB_ANGLE + 1


def v25_target_dim(max_nb_obj_per_image: int) -> int:
    return 1 + int(max_nb_obj_per_image) * V25_TARGET_STRIDE


def denorm_log_value(norm_values, lims_row):
    norm_values = np.asarray(norm_values, dtype=np.float64)
    lims_row = np.asarray(lims_row, dtype=np.float64)
    return np.exp(norm_values * (lims_row[0] - lims_row[1]) + lims_row[1])


def pix_to_arcsec(size_pix, pixel_size_deg: float, diameter_scale: float = 2.0):
    size_pix = np.asarray(size_pix, dtype=np.float64)
    return size_pix * (3600.0 * float(pixel_size_deg)) / float(diameter_scale)


def arcsec_to_pix(size_arcsec, pixel_size_deg: float, diameter_scale: float = 2.0):
    size_arcsec = np.asarray(size_arcsec, dtype=np.float64)
    return size_arcsec / (3600.0 * float(pixel_size_deg)) * float(diameter_scale)


def decode_obb_boxes(cx_pix, cy_pix, w_norm, h_norm, angle_vectors, lims):
    cx_pix = np.asarray(cx_pix, dtype=np.float64)
    cy_pix = np.asarray(cy_pix, dtype=np.float64)
    w_pix = denorm_log_value(w_norm, lims[1])
    h_pix = denorm_log_value(h_norm, lims[2])
    theta = decode_angle_vector(angle_vectors, normalize=True)
    return np.stack(canonicalize_le90(cx_pix, cy_pix, w_pix, h_pix, theta), axis=-1)


def decode_target_box(block, lims):
    block = np.asarray(block, dtype=np.float64)
    cx = 0.5 * (block[1] + block[4])
    cy = 0.5 * (block[2] + block[5])
    return decode_obb_boxes(
        np.asarray([cx]),
        np.asarray([cy]),
        np.asarray([block[8]]),
        np.asarray([block[9]]),
        np.asarray([block[10:12]]),
        lims,
    )[0]


def obb_catalog_arrays(flat_boxes, lims, pixel_size_deg: float):
    flat_boxes = np.asarray(flat_boxes, dtype=np.float64)
    centers_x = 0.5 * (flat_boxes[:, 0] + flat_boxes[:, 2]) - 0.5
    centers_y = 0.5 * (flat_boxes[:, 1] + flat_boxes[:, 3]) - 0.5
    obb = decode_obb_boxes(
        centers_x,
        centers_y,
        flat_boxes[:, 8],
        flat_boxes[:, 9],
        flat_boxes[:, 10:12],
        lims,
    )
    corners = xywhtheta_to_corners(obb)
    hbb = hbb_from_obb(obb)
    flux_jy = denorm_log_value(flat_boxes[:, 7], lims[0])
    bmaj_arcsec = pix_to_arcsec(obb[:, 2], pixel_size_deg)
    bmin_arcsec = pix_to_arcsec(obb[:, 3], pixel_size_deg)
    aspect = obb[:, 2] / np.maximum(obb[:, 3], 1.0e-8)
    return {
        "obb": obb,
        "corners": corners,
        "hbb": hbb,
        "flux_jy": flux_jy,
        "bmaj_arcsec": bmaj_arcsec,
        "bmin_arcsec": bmin_arcsec,
        "aspect_ratio": aspect,
    }
