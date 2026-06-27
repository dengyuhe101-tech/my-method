"""Build SDC1 rotated-box target representations without touching CIANNA flow.

This module converts SDC1 source catalogs into an explicit OBB source table:

    [cx, cy, w, h, theta_le90]

It also exposes the internal doubled-angle representation:

    [cx, cy, w, h, cos(2 theta), sin(2 theta)]

The output is intentionally independent from the existing train_network.py and
data_gen.py files. It is a preparation layer for a later native OBB YOLO head.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .bootstrap import CIANNA_NEW_ROOT, SKAO_SDC1_DIR
from .geometry import (
    BeamModel,
    EnvelopeModelConfig,
    canonicalize_le90,
    hbb_from_obb,
    source_params_to_envelope_box,
    source_params_to_le90_box,
    theta_to_internal,
    xywhtheta_to_corners,
    xywhtheta_to_internal,
)


SCRIPT_DIR = SKAO_SDC1_DIR
DEFAULT_SHARED_RAW_DATA_DIR = Path(
    "/shared/main/dengyuhe/SDC1_YOLO_OBB/raw_data/560Mhz-1kh"
)
DEFAULT_LOCAL_RAW_DATA_DIR = CIANNA_NEW_ROOT / "raw_data"
DEFAULT_PIXEL_SIZE_DEG = 0.000167847
DEFAULT_MIN_SIDE_PIX = 5.0
DEFAULT_SIDE_CLIP = (5.0, None)
DEFAULT_LEGACY_V1_SIDE_CLIP = (5.0, 64.0)
DEFAULT_VISIBILITY_THRESHOLD_JY_BEAM = None
DEFAULT_GAUSSIAN_CONTOUR_PEAK_FRACTION = 1.0 / 16.0
DEFAULT_EXPONENTIAL_RADIUS_SCALE = 1.6783469900166605
DEFAULT_EXPONENTIAL_CONTOUR_PEAK_FRACTION = float(np.exp(-DEFAULT_EXPONENTIAL_RADIUS_SCALE))
DEFAULT_EXPONENTIAL_LONG_CAP_FACTOR = 1.0
DEFAULT_CONTOUR_PEAK_FRACTION = DEFAULT_GAUSSIAN_CONTOUR_PEAK_FRACTION
DEFAULT_ELONGATED_CONTOUR_PEAK_FRACTION = DEFAULT_GAUSSIAN_CONTOUR_PEAK_FRACTION
DEFAULT_COMPLEX_CONTOUR_PEAK_FRACTION = DEFAULT_EXPONENTIAL_CONTOUR_PEAK_FRACTION
DEFAULT_COMPLEX_SIZE_TYPE_MIN = 3.0
DEFAULT_ELONGATED_ASPECT_MIN = 4.0
DEFAULT_ELONGATED_BMAJ_MIN_ARCSEC = 10.0
SIZE_POLICY_LAS_V1 = 0.0
SIZE_POLICY_GAUSSIAN = 1.0
SIZE_POLICY_EXPONENTIAL = 2.0
DEFAULT_BEAM_MODEL = BeamModel(major_arcsec=1.5, minor_arcsec=1.5, pa_deg=0.0)
DEFAULT_ENVELOPE_MODEL = EnvelopeModelConfig(
    contour_peak_fraction=DEFAULT_CONTOUR_PEAK_FRACTION,
    visibility_threshold_jy_beam=DEFAULT_VISIBILITY_THRESHOLD_JY_BEAM,
    min_contour_scale=1.0,
    max_contour_scale=4.0,
    centroid_extension_scale=0.0,
    min_side_pix=DEFAULT_MIN_SIDE_PIX,
    max_side_pix=None,
    margin_pix=0.0,
)

ROTATED_TABLE_COLUMNS = (
    "source_id",
    "cx_pix",
    "cy_pix",
    "w_pix",
    "h_pix",
    "theta_le90_deg",
    "cos_2theta",
    "sin_2theta",
    "flux_jy",
    "bmaj_arcsec",
    "bmin_arcsec",
    "pa_deg_original",
    "ra_core_deg",
    "dec_core_deg",
    "ra_centroid_deg",
    "dec_centroid_deg",
    "size_type",
    "class_id",
    "hbb_xmin",
    "hbb_ymin",
    "hbb_xmax",
    "hbb_ymax",
    "aspect_ratio",
    "contour_peak_fraction",
    "adaptive_source_group",
)


@dataclass
class RotatedSourceTable:
    data: np.ndarray
    columns: Tuple[str, ...] = ROTATED_TABLE_COLUMNS

    def col(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]

    def boxes(self) -> np.ndarray:
        return self.data[:, [self.columns.index(k) for k in (
            "cx_pix",
            "cy_pix",
            "w_pix",
            "h_pix",
            "theta_le90_deg",
        )]]

    def internal_boxes(self) -> np.ndarray:
        return self.data[:, [self.columns.index(k) for k in (
            "cx_pix",
            "cy_pix",
            "w_pix",
            "h_pix",
            "cos_2theta",
            "sin_2theta",
        )]]

    def hbb(self) -> np.ndarray:
        return self.data[:, [self.columns.index(k) for k in (
            "hbb_xmin",
            "hbb_ymin",
            "hbb_xmax",
            "hbb_ymax",
        )]]

    def subset(self, mask_or_indices) -> "RotatedSourceTable":
        return RotatedSourceTable(np.asarray(self.data[mask_or_indices], dtype=np.float64), self.columns)

    def save_csv(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ",".join(self.columns)
        np.savetxt(path, self.data, delimiter=",", header=header, comments="", fmt="%.10g")


def default_raw_data_dir() -> Path:
    env_raw = os.environ.get("CIANNA_SDC1_RAW_DATA", "").strip()
    if env_raw:
        return Path(env_raw)
    if (DEFAULT_LOCAL_RAW_DATA_DIR / "sdc1_560MHz_1000h.fits").is_file():
        return DEFAULT_LOCAL_RAW_DATA_DIR
    return DEFAULT_SHARED_RAW_DATA_DIR


def default_paths(raw_data_dir=None) -> Dict[str, Path]:
    raw = Path(raw_data_dir) if raw_data_dir is not None else default_raw_data_dir()
    return {
        "raw_data_dir": raw,
        "full_image": raw / "sdc1_560MHz_1000h.fits",
        "training_catalog": raw / "TrainingSet_560MHz.txt",
        "truth_catalog": raw / "True_560MHz.txt",
        "truth_pixel_catalog": raw / "True_560_pixel.csv",
        "local_training_selection": SCRIPT_DIR / "TrainingSet_perscut.txt",
    }


def load_beam_model(image_path=None, fallback: BeamModel = DEFAULT_BEAM_MODEL) -> BeamModel:
    """Read the image beam from FITS BMAJ/BMIN/BPA, falling back to 1.5 arcsec."""

    if image_path is None:
        return fallback
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError(
            "Astropy is required to read the FITS beam model. "
            "Use the CIANNA conda environment for this project."
        ) from exc

    with fits.open(image_path, memmap=True) as hdul:
        header = hdul[0].header
        major_deg = header.get("BMAJ")
        minor_deg = header.get("BMIN", major_deg)
        pa_deg = header.get("BPA", header.get("BEAMPA", fallback.pa_deg))

    if major_deg is None:
        return fallback
    if minor_deg is None:
        minor_deg = major_deg
    return BeamModel(
        major_arcsec=float(major_deg) * 3600.0,
        minor_arcsec=float(minor_deg) * 3600.0,
        pa_deg=float(pa_deg),
    )


def catalog_skiprows(path) -> int:
    name = Path(path).name
    if name == "TrainingSet_560MHz.txt":
        return 18
    return 0


def load_sdc1_catalog(path, skiprows=None, max_rows=None) -> np.ndarray:
    path = Path(path)
    if skiprows is None:
        skiprows = catalog_skiprows(path)
    return np.loadtxt(path, skiprows=skiprows, max_rows=max_rows)


@lru_cache(maxsize=4)
def load_image_wcs(image_path):
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
    except ImportError as exc:
        raise RuntimeError(
            "Astropy is required to convert RA/DEC catalogs to pixel OBB targets. "
            "Use the CIANNA conda environment for this project."
        ) from exc

    with fits.open(image_path, memmap=True) as hdul:
        return WCS(hdul[0].header)


def _radec_to_pixels(ra_deg, dec_deg, image_path, wcs=None) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astropy.wcs import utils
    except ImportError as exc:
        raise RuntimeError(
            "Astropy is required to convert RA/DEC catalogs to pixel OBB targets."
        ) from exc

    if image_path is None and wcs is None:
        raise ValueError("image_path is required when catalog x/y columns are unavailable or disabled.")
    if wcs is None:
        wcs = load_image_wcs(Path(image_path))
    coords = SkyCoord(ra=np.asarray(ra_deg) * u.degree, dec=np.asarray(dec_deg) * u.degree, frame="icrs")
    x_pix, y_pix = utils.skycoord_to_pixel(coords, wcs)
    return np.asarray(x_pix, dtype=np.float64), np.asarray(y_pix, dtype=np.float64)


def catalog_xy_pixels(
    catalog: np.ndarray,
    image_path=None,
    coord_mode: str = "core",
    prefer_catalog_xy: bool = False,
    wcs=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return source x/y pixels for a SDC1 text catalog.

    For the original TrainingSet file, columns 14/15 contain catalog-provided
    pixel coordinates. For reduced files such as TrainingSet_perscut.txt, WCS
    conversion is required.
    """

    catalog = np.asarray(catalog)
    if prefer_catalog_xy and catalog.shape[1] >= 15:
        return catalog[:, 13].astype(np.float64), catalog[:, 14].astype(np.float64)

    if image_path is None:
        raise ValueError("image_path is required when catalog x/y columns are unavailable or disabled.")

    if coord_mode == "core":
        ra = catalog[:, 1]
        dec = catalog[:, 2]
    elif coord_mode == "centroid":
        ra = catalog[:, 3]
        dec = catalog[:, 4]
    else:
        raise ValueError("coord_mode must be 'core' or 'centroid'.")

    return _radec_to_pixels(ra, dec, image_path, wcs=wcs)


def catalog_core_centroid_pixels(
    catalog: np.ndarray,
    image_path=None,
    prefer_catalog_xy: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return core and centroid pixels while doing WCS work once where possible."""

    catalog = np.asarray(catalog)
    if image_path is None and not (prefer_catalog_xy and catalog.shape[1] >= 15):
        raise ValueError("image_path is required when catalog x/y columns are unavailable or disabled.")

    wcs = None if image_path is None else load_image_wcs(Path(image_path))
    if prefer_catalog_xy and catalog.shape[1] >= 15:
        core_x = catalog[:, 13].astype(np.float64)
        core_y = catalog[:, 14].astype(np.float64)
        if image_path is None:
            return core_x, core_y, core_x.copy(), core_y.copy()
        centroid_x, centroid_y = _radec_to_pixels(catalog[:, 3], catalog[:, 4], image_path, wcs=wcs)
        return core_x, core_y, centroid_x, centroid_y

    ra = np.concatenate([catalog[:, 1], catalog[:, 3]])
    dec = np.concatenate([catalog[:, 2], catalog[:, 4]])
    x_pix, y_pix = _radec_to_pixels(ra, dec, image_path, wcs=wcs)
    split = catalog.shape[0]
    return x_pix[:split], y_pix[:split], x_pix[split:], y_pix[split:]


def adaptive_contour_peak_fraction(
    catalog: np.ndarray,
    base_fraction: float = DEFAULT_CONTOUR_PEAK_FRACTION,
    elongated_fraction: float = DEFAULT_ELONGATED_CONTOUR_PEAK_FRACTION,
    complex_fraction: float = DEFAULT_COMPLEX_CONTOUR_PEAK_FRACTION,
    complex_size_type_min: float = DEFAULT_COMPLEX_SIZE_TYPE_MIN,
    elongated_aspect_min: float = DEFAULT_ELONGATED_ASPECT_MIN,
    elongated_bmaj_min_arcsec: float = DEFAULT_ELONGATED_BMAJ_MIN_ARCSEC,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return historical Version D contour fractions and diagnostic group ids.

    The current default target builder no longer calls this helper; Version E
    uses `_size_aware_source_boxes` so the source groups directly follow
    catalog SIZE semantics. This function is kept for old experiments that
    explicitly want the pre-Version-E adaptive Gaussian envelope.

    Group ids:
        0 = ordinary source, base contour
        1 = highly elongated large source, intermediate loose contour
        2 = complex/extended source by size_type, loosest contour
    """

    catalog = np.asarray(catalog, dtype=np.float64)
    fractions = np.full(catalog.shape[0], float(base_fraction), dtype=np.float64)
    groups = np.zeros(catalog.shape[0], dtype=np.float64)

    bmaj = np.maximum(catalog[:, 7], 1.0e-9)
    bmin = np.maximum(catalog[:, 8], 1.0e-9)
    intrinsic_aspect = bmaj / bmin
    elongated = (intrinsic_aspect >= float(elongated_aspect_min)) & (
        bmaj >= float(elongated_bmaj_min_arcsec)
    )
    fractions[elongated] = float(elongated_fraction)
    groups[elongated] = 1.0

    size_type = catalog[:, 10] if catalog.shape[1] > 10 else np.zeros(catalog.shape[0], dtype=np.float64)
    complex_mask = size_type >= float(complex_size_type_min)
    fractions[complex_mask] = float(complex_fraction)
    groups[complex_mask] = 2.0
    return fractions, groups


def _broadcast_source_value(value, n_sources: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n_sources, float(arr), dtype=np.float64)
    if arr.shape[0] != n_sources:
        raise ValueError(f"{name} must be scalar or have one value per source.")
    return arr.astype(np.float64, copy=False)


def _size_type_masks(catalog: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    size_type = catalog[:, 10] if catalog.shape[1] > 10 else np.full(catalog.shape[0], 2.0)
    las_mask = size_type == 1.0
    exponential_mask = size_type >= DEFAULT_COMPLEX_SIZE_TYPE_MIN
    gaussian_mask = ~(las_mask | exponential_mask)
    return las_mask, gaussian_mask, exponential_mask


def _size_aware_source_boxes(
    catalog: np.ndarray,
    x_pix: np.ndarray,
    y_pix: np.ndarray,
    pixel_size_deg: float,
    min_side_pix: float,
    beam_model: BeamModel,
    model: EnvelopeModelConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build V1a boxes using source-type-specific size semantics.

    SIZE=1 LAS keeps the legacy V1 box exactly. SIZE=2 Gaussian uses a
    beam-convolved 2-FWHM contour. SIZE=3 Exponential uses the half-light
    exponential radius to thicken the short axis while capping the long axis
    at the V1 long side.
    """

    n_sources = catalog.shape[0]
    las_mask, gaussian_mask, exponential_mask = _size_type_masks(catalog)

    legacy_v1_boxes = source_params_to_le90_box(
        x_pix,
        y_pix,
        catalog[:, 7],
        catalog[:, 8],
        catalog[:, 9],
        pixel_size_deg=pixel_size_deg,
        side_clip=DEFAULT_LEGACY_V1_SIDE_CLIP,
    )
    boxes = legacy_v1_boxes.copy()
    contour_fraction = np.zeros(n_sources, dtype=np.float64)
    policy_group = np.full(n_sources, SIZE_POLICY_GAUSSIAN, dtype=np.float64)
    policy_group[las_mask] = SIZE_POLICY_LAS_V1
    policy_group[exponential_mask] = SIZE_POLICY_EXPONENTIAL

    gaussian_fraction = _broadcast_source_value(
        model.contour_peak_fraction,
        n_sources,
        "contour_peak_fraction",
    )
    contour_fraction[gaussian_mask] = gaussian_fraction[gaussian_mask]
    contour_fraction[exponential_mask] = DEFAULT_EXPONENTIAL_CONTOUR_PEAK_FRACTION

    if np.any(gaussian_mask):
        gaussian_model = EnvelopeModelConfig(
            contour_peak_fraction=gaussian_fraction[gaussian_mask],
            visibility_threshold_jy_beam=model.visibility_threshold_jy_beam,
            min_contour_scale=model.min_contour_scale,
            max_contour_scale=model.max_contour_scale,
            centroid_extension_scale=model.centroid_extension_scale,
            min_side_pix=min_side_pix,
            max_side_pix=model.max_side_pix,
            margin_pix=model.margin_pix,
        )
        boxes[gaussian_mask] = source_params_to_envelope_box(
            x_pix[gaussian_mask],
            y_pix[gaussian_mask],
            catalog[gaussian_mask, 7],
            catalog[gaussian_mask, 8],
            catalog[gaussian_mask, 9],
            pixel_size_deg=pixel_size_deg,
            flux_jy=catalog[gaussian_mask, 5],
            centroid_dx_pix=None,
            centroid_dy_pix=None,
            beam=beam_model,
            model=gaussian_model,
        )

    if np.any(exponential_mask):
        exp_boxes = source_params_to_le90_box(
            x_pix[exponential_mask],
            y_pix[exponential_mask],
            catalog[exponential_mask, 7],
            catalog[exponential_mask, 8],
            catalog[exponential_mask, 9],
            pixel_size_deg=pixel_size_deg,
            diameter_scale=2.0 * DEFAULT_EXPONENTIAL_RADIUS_SCALE,
            side_clip=(min_side_pix, None),
        )
        v1_exp = legacy_v1_boxes[exponential_mask]
        long_cap = DEFAULT_EXPONENTIAL_LONG_CAP_FACTOR * v1_exp[:, 2]
        new_w = np.minimum(exp_boxes[:, 2], long_cap)
        new_w = np.maximum(new_w, v1_exp[:, 2])
        new_h = np.maximum(v1_exp[:, 3], exp_boxes[:, 3])
        new_h = np.minimum(new_h, new_w)
        cx, cy, w, h, theta = canonicalize_le90(
            v1_exp[:, 0],
            v1_exp[:, 1],
            new_w,
            new_h,
            v1_exp[:, 4],
        )
        boxes[exponential_mask] = np.stack([cx, cy, w, h, theta], axis=-1)

    return boxes, contour_fraction, policy_group


def catalog_to_rotated_source_table(
    catalog: np.ndarray,
    image_path=None,
    pixel_size_deg: float = DEFAULT_PIXEL_SIZE_DEG,
    coord_mode: str = "core",
    side_clip: Optional[Tuple[float, Optional[float]]] = DEFAULT_SIDE_CLIP,
    prefer_catalog_xy: bool = False,
    beam: Optional[BeamModel] = None,
    envelope_model: Optional[EnvelopeModelConfig] = DEFAULT_ENVELOPE_MODEL,
    use_envelope_model: bool = True,
) -> RotatedSourceTable:
    """Convert a SDC1 source catalog to a le90 OBB source table.

    V1a defaults to a size-aware model-derived target. Set
    `use_envelope_model=False` to recover the V1 direct `2 * BMAJ/BMIN` rule.
    """

    catalog = np.asarray(catalog, dtype=np.float64)
    if catalog.ndim != 2 or catalog.shape[1] < 12:
        raise ValueError("Expected a SDC1 catalog array with at least 12 columns.")

    if use_envelope_model:
        if coord_mode not in ("core", "centroid"):
            raise ValueError("coord_mode must be 'core' or 'centroid'.")
        core_x, core_y, centroid_x, centroid_y = catalog_core_centroid_pixels(
            catalog,
            image_path=image_path,
            prefer_catalog_xy=prefer_catalog_xy,
        )
        if coord_mode == "core":
            x_pix, y_pix = core_x, core_y
        else:
            x_pix, y_pix = centroid_x, centroid_y
        model = envelope_model or DEFAULT_ENVELOPE_MODEL
        min_side_pix = model.min_side_pix
        max_side_pix = model.max_side_pix
        if side_clip is not None:
            min_side_pix = float(side_clip[0])
            max_side_pix = None if side_clip[1] is None else float(side_clip[1])
        model = EnvelopeModelConfig(
            contour_peak_fraction=model.contour_peak_fraction,
            visibility_threshold_jy_beam=model.visibility_threshold_jy_beam,
            min_contour_scale=model.min_contour_scale,
            max_contour_scale=model.max_contour_scale,
            centroid_extension_scale=model.centroid_extension_scale,
            min_side_pix=min_side_pix,
            max_side_pix=max_side_pix,
            margin_pix=model.margin_pix,
        )
        beam_model = beam or load_beam_model(image_path)
        boxes, contour_fraction, adaptive_group = _size_aware_source_boxes(
            catalog,
            x_pix,
            y_pix,
            pixel_size_deg,
            min_side_pix,
            beam_model,
            model,
        )
    else:
        contour_fraction = np.full(catalog.shape[0], DEFAULT_CONTOUR_PEAK_FRACTION, dtype=np.float64)
        adaptive_group = np.zeros(catalog.shape[0], dtype=np.float64)
        x_pix, y_pix = catalog_xy_pixels(
            catalog,
            image_path=image_path,
            coord_mode=coord_mode,
            prefer_catalog_xy=prefer_catalog_xy,
        )
        legacy_side_clip = None
        if side_clip is not None and side_clip[1] is not None:
            legacy_side_clip = (float(side_clip[0]), float(side_clip[1]))
        boxes = source_params_to_le90_box(
            x_pix,
            y_pix,
            catalog[:, 7],
            catalog[:, 8],
            catalog[:, 9],
            pixel_size_deg=pixel_size_deg,
            side_clip=legacy_side_clip,
        )
    cos2, sin2 = theta_to_internal(boxes[:, 4])
    hbb = hbb_from_obb(boxes)
    aspect = boxes[:, 2] / np.maximum(boxes[:, 3], 1.0e-9)

    data = np.column_stack(
        [
            catalog[:, 0],
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2],
            boxes[:, 3],
            boxes[:, 4],
            cos2,
            sin2,
            catalog[:, 5],
            catalog[:, 7],
            catalog[:, 8],
            catalog[:, 9],
            catalog[:, 1],
            catalog[:, 2],
            catalog[:, 3],
            catalog[:, 4],
            catalog[:, 10],
            catalog[:, 11],
            hbb[:, 0],
            hbb[:, 1],
            hbb[:, 2],
            hbb[:, 3],
            aspect,
            contour_fraction,
            adaptive_group,
        ]
    )
    return RotatedSourceTable(data=data)


def boxes_fully_inside_patch(boxes: np.ndarray, x0: float, y0: float, patch_size: int) -> np.ndarray:
    """Mask boxes whose four OBB corners are fully inside a patch."""

    corners = xywhtheta_to_corners(boxes)
    return (
        (corners[..., 0] >= x0).all(axis=-1)
        & (corners[..., 0] < x0 + patch_size).all(axis=-1)
        & (corners[..., 1] >= y0).all(axis=-1)
        & (corners[..., 1] < y0 + patch_size).all(axis=-1)
    )


def localize_boxes_to_patch(boxes: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """Shift OBB centers from full-image coordinates to patch-local coordinates."""

    local = np.asarray(boxes, dtype=np.float64).copy()
    local[:, 0] -= x0
    local[:, 1] -= y0
    return local


def make_rotated_patch_target_vector(
    source_table: RotatedSourceTable,
    x0: float,
    y0: float,
    patch_size: int,
    max_objects: int,
    angle_encoding: str = "theta",
) -> np.ndarray:
    """Create a compact future OBB target vector for one patch.

    This is not wired into the current CIANNA target reader yet. It documents
    the intended target layout for the future OBB YOLO head.

    angle_encoding="theta" per-object layout:
        [1, cx, cy, w, h, theta, 1]

    angle_encoding="internal" per-object layout:
        [1, cx, cy, w, h, cos2theta, sin2theta, 1]
    """

    boxes = source_table.boxes()
    keep = boxes_fully_inside_patch(boxes, x0, y0, patch_size)
    local_boxes = localize_boxes_to_patch(boxes[keep], x0, y0)
    n_obj = min(max_objects, local_boxes.shape[0])

    if angle_encoding == "theta":
        stride = 7
        target = np.zeros(1 + max_objects * stride, dtype=np.float32)
        target[0] = n_obj
        for i in range(n_obj):
            row = np.array([1.0, *local_boxes[i], 1.0], dtype=np.float32)
            target[1 + i * stride : 1 + (i + 1) * stride] = row
        return target

    if angle_encoding == "internal":
        stride = 8
        target = np.zeros(1 + max_objects * stride, dtype=np.float32)
        target[0] = n_obj
        local_internal = xywhtheta_to_internal(local_boxes)
        for i in range(n_obj):
            row = np.array([1.0, *local_internal[i], 1.0], dtype=np.float32)
            target[1 + i * stride : 1 + (i + 1) * stride] = row
        return target

    raise ValueError("angle_encoding must be 'theta' or 'internal'.")


def build_default_training_table(coord_mode: str = "core", raw_data_dir=None) -> RotatedSourceTable:
    paths = default_paths(raw_data_dir)
    catalog_path = paths["local_training_selection"]
    catalog = load_sdc1_catalog(catalog_path)
    return catalog_to_rotated_source_table(
        catalog,
        image_path=paths["full_image"],
        coord_mode=coord_mode,
        side_clip=DEFAULT_SIDE_CLIP,
    )


def filter_table_to_bounds(
    table: RotatedSourceTable,
    x_min: float = 0.0,
    y_min: float = 0.0,
    x_max: float = 32768.0,
    y_max: float = 32768.0,
) -> RotatedSourceTable:
    boxes = table.boxes()
    corners = xywhtheta_to_corners(boxes)
    mask = (
        (corners[..., 0] >= x_min).all(axis=-1)
        & (corners[..., 0] < x_max).all(axis=-1)
        & (corners[..., 1] >= y_min).all(axis=-1)
        & (corners[..., 1] < y_max).all(axis=-1)
    )
    return table.subset(mask)


def summarize_table(table: RotatedSourceTable) -> Dict[str, float]:
    boxes = table.boxes()
    summary = {
        "count": float(boxes.shape[0]),
        "cx_min": float(np.min(boxes[:, 0])),
        "cx_max": float(np.max(boxes[:, 0])),
        "cy_min": float(np.min(boxes[:, 1])),
        "cy_max": float(np.max(boxes[:, 1])),
        "w_mean": float(np.mean(boxes[:, 2])),
        "h_mean": float(np.mean(boxes[:, 3])),
        "aspect_p95": float(np.percentile(boxes[:, 2] / np.maximum(boxes[:, 3], 1.0e-9), 95.0)),
    }
    if "adaptive_source_group" in table.columns:
        groups = table.col("adaptive_source_group")
        summary.update(
            {
                "policy_las_v1_count": float(np.sum(groups == SIZE_POLICY_LAS_V1)),
                "policy_gaussian_count": float(np.sum(groups == SIZE_POLICY_GAUSSIAN)),
                "policy_exponential_count": float(np.sum(groups == SIZE_POLICY_EXPONENTIAL)),
            }
        )
    if "contour_peak_fraction" in table.columns:
        fractions = table.col("contour_peak_fraction")
        summary.update(
            {
                "contour_fraction_min": float(np.min(fractions)),
                "contour_fraction_max": float(np.max(fractions)),
            }
        )
    return summary
