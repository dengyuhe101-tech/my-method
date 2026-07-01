"""CSV table IO for angle angle target validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .angle_codec import encode_theta_le90
from .angle_loss import AspectWeightConfig, angle_weight_from_aspect


ANGLE_DIR = Path(__file__).resolve().parents[2]
SKAO_DIR = ANGLE_DIR.parent
DEFAULT_SOURCE_TABLE = SKAO_DIR / "geometry" / "rotated_training_source_table.csv"
DEFAULT_OUTPUT_DIR = ANGLE_DIR / "outputs"

ANGLE_TARGET_COLUMNS = (
    "source_id",
    "theta_le90_deg",
    "cos_2theta",
    "sin_2theta",
    "angle_weight",
    "aspect_ratio",
    "w_pix",
    "h_pix",
    "sqrt_area_pix",
    "flux_jy",
    "bmaj_arcsec",
    "bmin_arcsec",
)


@dataclass
class AngleTargetTable:
    data: np.ndarray
    columns: Sequence[str] = ANGLE_TARGET_COLUMNS

    def col(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]

    @property
    def source_id(self) -> np.ndarray:
        return self.col("source_id").astype(np.int64)

    @property
    def target_vectors(self) -> np.ndarray:
        return self.data[:, [self.columns.index("cos_2theta"), self.columns.index("sin_2theta")]]

    @property
    def weights(self) -> np.ndarray:
        return self.col("angle_weight")

    @property
    def theta_deg(self) -> np.ndarray:
        return self.col("theta_le90_deg")


def load_named_csv(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding=None)


def structured_columns(table: np.ndarray) -> Sequence[str]:
    if table.dtype.names is None:
        raise ValueError("Expected a CSV with a header row.")
    return table.dtype.names


def load_rotated_source_table(path: Path = DEFAULT_SOURCE_TABLE) -> np.ndarray:
    return load_named_csv(path)


def build_angle_target_table(
    source_table: np.ndarray,
    weight_config: AspectWeightConfig | None = None,
) -> AngleTargetTable:
    names = structured_columns(source_table)
    required = {
        "source_id",
        "theta_le90_deg",
        "cos_2theta",
        "sin_2theta",
        "aspect_ratio",
        "w_pix",
        "h_pix",
        "flux_jy",
        "bmaj_arcsec",
        "bmin_arcsec",
    }
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError("geometry rotated table is missing columns: %s" % ", ".join(missing))

    theta = np.asarray(source_table["theta_le90_deg"], dtype=np.float64)
    encoded = encode_theta_le90(theta)
    aspect = np.asarray(source_table["aspect_ratio"], dtype=np.float64)
    weights = angle_weight_from_aspect(aspect, weight_config)
    w_pix = np.asarray(source_table["w_pix"], dtype=np.float64)
    h_pix = np.asarray(source_table["h_pix"], dtype=np.float64)
    sqrt_area = np.sqrt(np.maximum(w_pix * h_pix, 0.0))

    data = np.column_stack(
        [
            source_table["source_id"],
            theta,
            encoded[:, 0],
            encoded[:, 1],
            weights,
            aspect,
            w_pix,
            h_pix,
            sqrt_area,
            source_table["flux_jy"],
            source_table["bmaj_arcsec"],
            source_table["bmin_arcsec"],
        ]
    )
    return AngleTargetTable(data=data)


def save_angle_target_table(table: AngleTargetTable, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        table.data,
        delimiter=",",
        header=",".join(table.columns),
        comments="",
        fmt="%.10g",
    )


def load_angle_target_table(path: Path) -> AngleTargetTable:
    raw = load_named_csv(path)
    names = structured_columns(raw)
    data = np.column_stack([raw[name] for name in names])
    return AngleTargetTable(data=data, columns=names)
