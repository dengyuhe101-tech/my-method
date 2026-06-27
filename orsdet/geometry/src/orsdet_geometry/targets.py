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
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .bootstrap import SKAO_SDC1_DIR
from .geometry import (
    hbb_from_obb,
    source_params_to_le90_box,
    theta_to_internal,
    xywhtheta_to_corners,
    xywhtheta_to_internal,
)


SCRIPT_DIR = SKAO_SDC1_DIR
DEFAULT_RAW_DATA_DIR = Path(
    "/shared/main/dengyuhe/SDC1_YOLO_OBB/raw_data/560Mhz-1kh"
)
DEFAULT_PIXEL_SIZE_DEG = 0.000167847
DEFAULT_SIDE_CLIP = (5.0, 64.0)

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


def default_paths(raw_data_dir=None) -> Dict[str, Path]:
    raw = Path(raw_data_dir) if raw_data_dir is not None else DEFAULT_RAW_DATA_DIR
    return {
        "raw_data_dir": raw,
        "full_image": raw / "sdc1_560MHz_1000h.fits",
        "training_catalog": raw / "TrainingSet_560MHz.txt",
        "truth_catalog": raw / "True_560MHz.txt",
        "truth_pixel_catalog": raw / "True_560_pixel.csv",
        "local_training_selection": SCRIPT_DIR / "TrainingSet_perscut.txt",
    }


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


def catalog_xy_pixels(
    catalog: np.ndarray,
    image_path=None,
    coord_mode: str = "core",
    prefer_catalog_xy: bool = False,
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

    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astropy.wcs import utils
    except ImportError as exc:
        raise RuntimeError(
            "Astropy is required to convert RA/DEC catalogs to pixel OBB targets."
        ) from exc

    if coord_mode == "core":
        ra = catalog[:, 1]
        dec = catalog[:, 2]
    elif coord_mode == "centroid":
        ra = catalog[:, 3]
        dec = catalog[:, 4]
    else:
        raise ValueError("coord_mode must be 'core' or 'centroid'.")

    wcs = load_image_wcs(image_path)
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    x_pix, y_pix = utils.skycoord_to_pixel(coords, wcs)
    return np.asarray(x_pix, dtype=np.float64), np.asarray(y_pix, dtype=np.float64)


def catalog_to_rotated_source_table(
    catalog: np.ndarray,
    image_path=None,
    pixel_size_deg: float = DEFAULT_PIXEL_SIZE_DEG,
    coord_mode: str = "core",
    side_clip: Optional[Tuple[float, float]] = DEFAULT_SIDE_CLIP,
    prefer_catalog_xy: bool = False,
) -> RotatedSourceTable:
    """Convert a SDC1 source catalog to a le90 OBB source table."""

    catalog = np.asarray(catalog, dtype=np.float64)
    if catalog.ndim != 2 or catalog.shape[1] < 12:
        raise ValueError("Expected a SDC1 catalog array with at least 12 columns.")

    x_pix, y_pix = catalog_xy_pixels(
        catalog,
        image_path=image_path,
        coord_mode=coord_mode,
        prefer_catalog_xy=prefer_catalog_xy,
    )
    boxes = source_params_to_le90_box(
        x_pix,
        y_pix,
        catalog[:, 7],
        catalog[:, 8],
        catalog[:, 9],
        pixel_size_deg=pixel_size_deg,
        side_clip=side_clip,
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


def build_default_training_table(coord_mode: str = "core") -> RotatedSourceTable:
    paths = default_paths()
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
    return {
        "count": float(boxes.shape[0]),
        "cx_min": float(np.min(boxes[:, 0])),
        "cx_max": float(np.max(boxes[:, 0])),
        "cy_min": float(np.min(boxes[:, 1])),
        "cy_max": float(np.max(boxes[:, 1])),
        "w_mean": float(np.mean(boxes[:, 2])),
        "h_mean": float(np.mean(boxes[:, 3])),
        "aspect_p95": float(np.percentile(boxes[:, 2] / np.maximum(boxes[:, 3], 1.0e-9), 95.0)),
    }
