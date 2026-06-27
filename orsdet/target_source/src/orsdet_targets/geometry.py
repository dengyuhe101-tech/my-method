"""Rotated bounding-box geometry helpers for the SDC1 OBB migration.

External box convention:
    [cx, cy, w, h, theta_deg]

where w is the long side, h is the short side, and theta_deg follows the
le90 convention: theta is in [-90, 90) and is measured clockwise from the
positive image x-axis to the long side. This matches the PA convention used
by the SDC1 catalog comments and the existing data_gen.py vertex transform.

Internal angle convention:
    [cos(2 theta), sin(2 theta)]

The doubled angle removes the 180 degree periodic discontinuity of oriented
rectangles. Geometry code in this file keeps theta in degrees at the external
API boundary and uses radians only internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


ANGLE_PERIOD_DEG = 180.0
LE90_MIN_DEG = -90.0
LE90_MAX_DEG = 90.0
EPS = 1.0e-9


@dataclass(frozen=True)
class AngleRoundTripError:
    """Compact metric bundle for angle representation checks."""

    max_abs_deg: float
    mean_abs_deg: float
    p99_abs_deg: float


@dataclass(frozen=True)
class BeamModel:
    """Gaussian beam model in angular FWHM units.

    The SDC1 FITS header stores BMAJ/BMIN in degrees. V1a uses this beam for
    the Gaussian SIZE=2 policy before converting source axes to pixel boxes.
    """

    major_arcsec: float = 1.5
    minor_arcsec: float = 1.5
    pa_deg: float = 0.0


@dataclass(frozen=True)
class EnvelopeModelConfig:
    """Parameters for model-derived Gaussian source envelopes.

    The current size-aware V1a default uses `contour_peak_fraction=1/16` for
    SIZE=2 Gaussian sources. SIZE=1 LAS and SIZE=3 Exponential sources are
    handled outside this Gaussian envelope helper.
    """

    contour_peak_fraction: object = 1.0 / 16.0
    visibility_threshold_jy_beam: Optional[float] = None
    min_contour_scale: float = 1.0
    max_contour_scale: float = 4.0
    centroid_extension_scale: float = 0.0
    min_side_pix: float = 5.0
    max_side_pix: Optional[float] = None
    margin_pix: float = 0.0


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def normalize_theta_le90(theta_deg):
    """Normalize an angle to the le90 interval [-90, 90)."""

    theta = _as_array(theta_deg)
    normalized = np.mod(theta - LE90_MIN_DEG, ANGLE_PERIOD_DEG) + LE90_MIN_DEG
    return normalized.item() if normalized.ndim == 0 else normalized


def le90_angle_diff_deg(a_deg, b_deg):
    """Smallest signed difference between two le90 angles in degrees."""

    diff = normalize_theta_le90(_as_array(a_deg) - _as_array(b_deg))
    return diff


def canonicalize_le90(cx, cy, w, h, theta_deg):
    """Return a long-side le90 box.

    If h > w, the sides are swapped and theta is rotated by 90 degrees.
    Equal sides keep their incoming angle because square boxes have no
    geometrically meaningful orientation.
    """

    cx = _as_array(cx)
    cy = _as_array(cy)
    w = np.maximum(_as_array(w), EPS)
    h = np.maximum(_as_array(h), EPS)
    theta = _as_array(theta_deg)

    swap = h > w
    out_w = np.where(swap, h, w)
    out_h = np.where(swap, w, h)
    out_theta = normalize_theta_le90(np.where(swap, theta + 90.0, theta))

    if np.ndim(out_w) == 0:
        return cx.item(), cy.item(), out_w.item(), out_h.item(), float(out_theta)
    return cx, cy, out_w, out_h, out_theta


def theta_to_internal(theta_deg):
    """Encode theta as cos(2 theta), sin(2 theta)."""

    theta_rad = np.deg2rad(_as_array(theta_deg))
    return np.cos(2.0 * theta_rad), np.sin(2.0 * theta_rad)


def internal_to_theta(cos_2theta, sin_2theta):
    """Decode cos(2 theta), sin(2 theta) back to le90 theta degrees."""

    theta_rad = 0.5 * np.arctan2(_as_array(sin_2theta), _as_array(cos_2theta))
    theta_deg = normalize_theta_le90(np.rad2deg(theta_rad))
    return theta_deg


def xywhtheta_to_internal(boxes):
    """Convert external OBB boxes to [cx, cy, w, h, cos2theta, sin2theta]."""

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape[-1] != 5:
        raise ValueError("Expected boxes with last dimension 5: [cx, cy, w, h, theta].")

    cx, cy, w, h, theta = canonicalize_le90(
        boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3], boxes[..., 4]
    )
    cos2, sin2 = theta_to_internal(theta)
    return np.stack([cx, cy, w, h, cos2, sin2], axis=-1)


def internal_to_xywhtheta(encoded):
    """Convert [cx, cy, w, h, cos2theta, sin2theta] to external OBB boxes."""

    encoded = np.asarray(encoded, dtype=np.float64)
    if encoded.shape[-1] != 6:
        raise ValueError(
            "Expected encoded boxes with last dimension 6: "
            "[cx, cy, w, h, cos2theta, sin2theta]."
        )

    theta = internal_to_theta(encoded[..., 4], encoded[..., 5])
    cx, cy, w, h, theta = canonicalize_le90(
        encoded[..., 0], encoded[..., 1], encoded[..., 2], encoded[..., 3], theta
    )
    return np.stack([cx, cy, w, h, theta], axis=-1)


def xywhtheta_to_corners(boxes):
    """Convert le90 boxes to ordered 4-corner polygons.

    The returned corner order is the transformed local rectangle order:
    top-left, top-right, bottom-right, bottom-left in the box's local frame.
    """

    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape[-1] != 5:
        raise ValueError("Expected boxes with last dimension 5: [cx, cy, w, h, theta].")

    cx, cy, w, h, theta = canonicalize_le90(
        boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3], boxes[..., 4]
    )
    cx = np.asarray(cx)
    cy = np.asarray(cy)
    w = np.asarray(w)
    h = np.asarray(h)
    theta = np.asarray(theta)

    local = np.array(
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
        dtype=np.float64,
    )
    lx = local[:, 0][None, :]
    ly = local[:, 1][None, :]
    w = w[..., None]
    h = h[..., None]
    cx = cx[..., None]
    cy = cy[..., None]
    theta_rad = np.deg2rad(theta)[..., None]
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)

    u = lx * w
    v = ly * h
    x = cx + u * c + v * s
    y = cy - u * s + v * c
    return np.stack([x, y], axis=-1)


def corners_to_xywhtheta(corners):
    """Recover [cx, cy, w, h, theta] from ordered OBB corners."""

    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape[-2:] != (4, 2):
        raise ValueError("Expected corners with shape (..., 4, 2).")

    center = corners.mean(axis=-2)
    edge0 = corners[..., 1, :] - corners[..., 0, :]
    edge1 = corners[..., 2, :] - corners[..., 1, :]
    len0 = np.linalg.norm(edge0, axis=-1)
    len1 = np.linalg.norm(edge1, axis=-1)

    use_edge0 = len0 >= len1
    long_vec = np.where(use_edge0[..., None], edge0, edge1)
    w = np.where(use_edge0, len0, len1)
    h = np.where(use_edge0, len1, len0)
    theta = np.rad2deg(np.arctan2(-long_vec[..., 1], long_vec[..., 0]))

    cx, cy, w, h, theta = canonicalize_le90(center[..., 0], center[..., 1], w, h, theta)
    return np.stack([cx, cy, w, h, theta], axis=-1)


def hbb_from_obb(boxes):
    """Return axis-aligned [xmin, ymin, xmax, ymax] boxes enclosing OBBs."""

    corners = xywhtheta_to_corners(boxes)
    xy_min = corners.min(axis=-2)
    xy_max = corners.max(axis=-2)
    return np.concatenate([xy_min, xy_max], axis=-1)


def polygon_area(points):
    """Absolute area of a polygon represented by ordered 2D points."""

    points = np.asarray(points, dtype=np.float64)
    x = points[..., 0]
    y = points[..., 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=-1) - y * np.roll(x, -1, axis=-1), axis=-1))
    return area


def signed_polygon_area(points):
    """Signed area of an ordered polygon."""

    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))


def _line_intersection(p1, p2, q1, q2):
    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < EPS:
        return p2
    t = ((q1[0] - p1[0]) * s[1] - (q1[1] - p1[1]) * s[0]) / denom
    return p1 + t * r


def _inside_half_plane(point, edge_start, edge_end, clip_ccw):
    edge = edge_end - edge_start
    rel = point - edge_start
    cross = edge[0] * rel[1] - edge[1] * rel[0]
    return cross >= -EPS if clip_ccw else cross <= EPS


def polygon_clip(subject_polygon, clip_polygon):
    """Clip a polygon by another convex polygon using Sutherland-Hodgman."""

    output = [np.asarray(p, dtype=np.float64) for p in subject_polygon]
    clip_polygon = np.asarray(clip_polygon, dtype=np.float64)
    clip_ccw = signed_polygon_area(clip_polygon) >= 0.0

    for i in range(len(clip_polygon)):
        edge_start = clip_polygon[i]
        edge_end = clip_polygon[(i + 1) % len(clip_polygon)]
        input_list = output
        output = []
        if not input_list:
            break
        prev = input_list[-1]
        prev_inside = _inside_half_plane(prev, edge_start, edge_end, clip_ccw)
        for curr in input_list:
            curr_inside = _inside_half_plane(curr, edge_start, edge_end, clip_ccw)
            if curr_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, curr, edge_start, edge_end))
                output.append(curr)
            elif prev_inside:
                output.append(_line_intersection(prev, curr, edge_start, edge_end))
            prev = curr
            prev_inside = curr_inside
    return np.asarray(output, dtype=np.float64)


def _single_box_to_polygon(box: Iterable[float]) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64)
    if box.size != 5:
        raise ValueError("Expected a single OBB with 5 values: [cx, cy, w, h, theta].")
    return xywhtheta_to_corners(box.reshape(1, 5))[0]


def _single_rotated_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    poly_a = _single_box_to_polygon(box_a)
    poly_b = _single_box_to_polygon(box_b)
    inter = polygon_clip(poly_a, poly_b)
    inter_area = 0.0 if inter.size == 0 else float(polygon_area(inter))
    area_a = float(polygon_area(poly_a))
    area_b = float(polygon_area(poly_b))
    union = area_a + area_b - inter_area
    if union <= EPS:
        return 0.0
    return min(1.0, max(0.0, inter_area / union))


def rotated_iou(box_a, box_b):
    """Compute OBB IoU for single boxes or broadcast-compatible box arrays."""

    box_a = np.asarray(box_a, dtype=np.float64)
    box_b = np.asarray(box_b, dtype=np.float64)
    if box_a.shape[-1] != 5 or box_b.shape[-1] != 5:
        raise ValueError("Expected boxes with last dimension 5: [cx, cy, w, h, theta].")

    boxes_a, boxes_b = np.broadcast_arrays(box_a, box_b)
    flat_a = boxes_a.reshape(-1, 5)
    flat_b = boxes_b.reshape(-1, 5)
    values = np.array([_single_rotated_iou(a, b) for a, b in zip(flat_a, flat_b)], dtype=np.float64)
    values = values.reshape(boxes_a.shape[:-1])
    return float(values) if values.ndim == 0 else values


def _fwhm_covariance_arcsec2(major_arcsec, minor_arcsec, theta_deg):
    """Return 2x2 Gaussian covariance matrices from FWHM ellipse axes."""

    major = np.maximum(_as_array(major_arcsec), EPS)
    minor = np.maximum(_as_array(minor_arcsec), EPS)
    theta = np.deg2rad(_as_array(theta_deg))
    major, minor, theta = np.broadcast_arrays(major, minor, theta)

    sigma_major = major / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_minor = minor / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    c = np.cos(theta)
    s = np.sin(theta)

    # The project angle convention is clockwise in image coordinates. The long
    # axis unit vector therefore matches xywhtheta_to_corners: [cos, -sin].
    long_x = c
    long_y = -s
    short_x = s
    short_y = c

    cov = np.empty(major.shape + (2, 2), dtype=np.float64)
    cov[..., 0, 0] = sigma_major**2 * long_x**2 + sigma_minor**2 * short_x**2
    cov[..., 0, 1] = sigma_major**2 * long_x * long_y + sigma_minor**2 * short_x * short_y
    cov[..., 1, 0] = cov[..., 0, 1]
    cov[..., 1, 1] = sigma_major**2 * long_y**2 + sigma_minor**2 * short_y**2
    return cov


def _covariance_to_fwhm_axes(cov):
    """Convert covariance matrices to FWHM major/minor axes and le90 theta."""

    cov = np.asarray(cov, dtype=np.float64)
    flat = cov.reshape(-1, 2, 2)
    evals, evecs = np.linalg.eigh(flat)
    order = np.argsort(evals, axis=1)[:, ::-1]
    evals = np.take_along_axis(evals, order, axis=1)
    evecs = np.take_along_axis(evecs, order[:, None, :], axis=2)

    sigma_major = np.sqrt(np.maximum(evals[:, 0], EPS))
    sigma_minor = np.sqrt(np.maximum(evals[:, 1], EPS))
    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    major = sigma_major * fwhm_factor
    minor = sigma_minor * fwhm_factor

    major_vec = evecs[:, :, 0]
    theta = np.rad2deg(np.arctan2(-major_vec[:, 1], major_vec[:, 0]))
    theta = normalize_theta_le90(theta)

    out_shape = cov.shape[:-2]
    return major.reshape(out_shape), minor.reshape(out_shape), theta.reshape(out_shape)


def _fixed_contour_scale(fraction):
    fraction = _as_array(fraction)
    if np.any((fraction <= 0.0) | (fraction >= 1.0)):
        raise ValueError("contour_peak_fraction must be in (0, 1).")
    scale = np.sqrt(np.log(1.0 / fraction) / np.log(2.0))
    return scale.item() if scale.ndim == 0 else scale


def _threshold_contour_scale(
    flux_jy,
    obs_major_arcsec,
    obs_minor_arcsec,
    beam: BeamModel,
    threshold_jy_beam: float,
    fallback_scale,
):
    """Estimate visible Gaussian contour scale from integrated flux and threshold."""

    flux = np.maximum(_as_array(flux_jy), EPS)
    obs_area = np.maximum(_as_array(obs_major_arcsec) * _as_array(obs_minor_arcsec), EPS)
    beam_area = max(float(beam.major_arcsec) * float(beam.minor_arcsec), EPS)
    peak_jy_beam = flux * beam_area / obs_area
    ratio = np.maximum(peak_jy_beam / max(float(threshold_jy_beam), EPS), 1.0 + EPS)
    scale = np.sqrt(np.log(ratio) / np.log(2.0))
    return np.where(np.isfinite(scale), scale, fallback_scale)


def source_params_to_envelope_box(
    x_pix,
    y_pix,
    bmaj_arcsec,
    bmin_arcsec,
    pa_deg,
    pixel_size_deg,
    flux_jy=None,
    centroid_dx_pix=None,
    centroid_dy_pix=None,
    beam: Optional[BeamModel] = None,
    model: Optional[EnvelopeModelConfig] = None,
):
    """Convert source physics to a model-derived visible OBB envelope.

    V1 used `2 * BMAJ/BMIN` directly. V1a instead treats catalog axes as an
    intrinsic Gaussian source, convolves that source with the FITS beam, chooses
    a visible Gaussian contour, then expands the result by optional
    core-centroid displacement and a small pixel margin. No image segmentation
    is used.
    """

    model = model or EnvelopeModelConfig()
    beam = beam or BeamModel()

    x_pix = _as_array(x_pix)
    y_pix = _as_array(y_pix)
    src_cov = _fwhm_covariance_arcsec2(bmaj_arcsec, bmin_arcsec, pa_deg)
    beam_cov = _fwhm_covariance_arcsec2(beam.major_arcsec, beam.minor_arcsec, beam.pa_deg)
    obs_cov = src_cov + beam_cov
    obs_major, obs_minor, theta = _covariance_to_fwhm_axes(obs_cov)

    fixed_scale = _fixed_contour_scale(model.contour_peak_fraction)
    contour_scale = np.full_like(obs_major, fixed_scale, dtype=np.float64)
    if model.visibility_threshold_jy_beam is not None and flux_jy is not None:
        contour_scale = _threshold_contour_scale(
            flux_jy,
            obs_major,
            obs_minor,
            beam,
            model.visibility_threshold_jy_beam,
            fixed_scale,
        )

    contour_scale = np.clip(contour_scale, model.min_contour_scale, model.max_contour_scale)
    w_arcsec = obs_major * contour_scale
    h_arcsec = obs_minor * contour_scale

    w_pix = w_arcsec / (3600.0 * float(pixel_size_deg))
    h_pix = h_arcsec / (3600.0 * float(pixel_size_deg))

    if centroid_dx_pix is not None and centroid_dy_pix is not None and model.centroid_extension_scale != 0.0:
        dx = _as_array(centroid_dx_pix)
        dy = _as_array(centroid_dy_pix)
        theta_rad = np.deg2rad(theta)
        long_x = np.cos(theta_rad)
        long_y = -np.sin(theta_rad)
        short_x = np.sin(theta_rad)
        short_y = np.cos(theta_rad)
        long_proj = np.abs(dx * long_x + dy * long_y)
        short_proj = np.abs(dx * short_x + dy * short_y)
        w_pix = w_pix + 2.0 * model.centroid_extension_scale * long_proj
        h_pix = h_pix + 2.0 * model.centroid_extension_scale * short_proj

    if model.margin_pix > 0.0:
        w_pix = w_pix + 2.0 * model.margin_pix
        h_pix = h_pix + 2.0 * model.margin_pix

    w_pix = np.maximum(w_pix, model.min_side_pix)
    h_pix = np.maximum(h_pix, model.min_side_pix)
    if model.max_side_pix is not None:
        w_pix = np.minimum(w_pix, float(model.max_side_pix))
        h_pix = np.minimum(h_pix, float(model.max_side_pix))

    return np.stack(canonicalize_le90(x_pix, y_pix, w_pix, h_pix, theta), axis=-1)


def source_params_to_le90_box(
    x_pix,
    y_pix,
    bmaj_arcsec,
    bmin_arcsec,
    pa_deg,
    pixel_size_deg,
    diameter_scale: float = 2.0,
    side_clip: Optional[Tuple[float, float]] = None,
):
    """Convert SDC1 source geometry to le90 OBBs in pixel units.

    The current horizontal-box pipeline used 2 * BMAJ/BMIN as the starting
    source extent. The same diameter_scale default is kept here so this OBB
    representation is comparable to the existing target geometry.
    """

    x_pix = _as_array(x_pix)
    y_pix = _as_array(y_pix)
    bmaj_pix = _as_array(bmaj_arcsec) / (3600.0 * float(pixel_size_deg)) * diameter_scale
    bmin_pix = _as_array(bmin_arcsec) / (3600.0 * float(pixel_size_deg)) * diameter_scale
    theta = _as_array(pa_deg)

    if side_clip is not None:
        lo, hi = side_clip
        bmaj_pix = np.clip(bmaj_pix, lo, hi)
        bmin_pix = np.clip(bmin_pix, lo, hi)

    return np.stack(canonicalize_le90(x_pix, y_pix, bmaj_pix, bmin_pix, theta), axis=-1)


def angle_round_trip_error(theta_original, theta_recovered) -> AngleRoundTripError:
    """Return absolute le90 angle-error summary in degrees."""

    abs_err = np.abs(le90_angle_diff_deg(theta_recovered, theta_original))
    return AngleRoundTripError(
        max_abs_deg=float(np.max(abs_err)),
        mean_abs_deg=float(np.mean(abs_err)),
        p99_abs_deg=float(np.percentile(abs_err, 99.0)),
    )
