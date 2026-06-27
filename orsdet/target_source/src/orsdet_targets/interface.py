"""Stable adapter layer for replacing HBB targets with OBB targets later."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from . import geometry
from .geometry import BeamModel, EnvelopeModelConfig
from .targets import (
    DEFAULT_ENVELOPE_MODEL,
    DEFAULT_PIXEL_SIZE_DEG,
    DEFAULT_SIDE_CLIP,
    RotatedSourceTable,
    catalog_to_rotated_source_table,
    default_paths,
    load_sdc1_catalog,
    make_rotated_patch_target_vector,
)


@dataclass(frozen=True)
class OBBTargetConfig:
    """Configuration for the first OBB target layout."""

    coord_mode: str = "core"
    angle_encoding: str = "internal"
    pixel_size_deg: float = DEFAULT_PIXEL_SIZE_DEG
    side_clip: Tuple[float, Optional[float]] = DEFAULT_SIDE_CLIP
    diameter_scale: float = 2.0
    use_envelope_model: bool = True
    envelope_model: EnvelopeModelConfig = DEFAULT_ENVELOPE_MODEL
    beam: Optional[BeamModel] = None
    patch_size: int = 256
    max_objects: int = 512

    @property
    def target_stride(self) -> int:
        if self.angle_encoding == "theta":
            return 7
        if self.angle_encoding == "internal":
            return 8
        raise ValueError("angle_encoding must be 'theta' or 'internal'.")


class OBBTargetAdapter:
    """Small API surface intended for future `data_gen.py` integration."""

    def __init__(self, config: OBBTargetConfig | None = None):
        self.config = config or OBBTargetConfig()

    def build_source_table(
        self,
        catalog_path: Optional[Path] = None,
        image_path: Optional[Path] = None,
        raw_data_dir: Optional[Path] = None,
    ) -> RotatedSourceTable:
        paths = default_paths(raw_data_dir)
        catalog_path = Path(catalog_path) if catalog_path is not None else paths["local_training_selection"]
        image_path = Path(image_path) if image_path is not None else paths["full_image"]
        catalog = load_sdc1_catalog(catalog_path)
        return catalog_to_rotated_source_table(
            catalog,
            image_path=image_path,
            pixel_size_deg=self.config.pixel_size_deg,
            coord_mode=self.config.coord_mode,
            side_clip=self.config.side_clip,
            beam=self.config.beam,
            envelope_model=self.config.envelope_model,
            use_envelope_model=self.config.use_envelope_model,
        )

    def make_patch_target(
        self,
        source_table: RotatedSourceTable,
        x0: float,
        y0: float,
        patch_size: Optional[int] = None,
        max_objects: Optional[int] = None,
    ) -> np.ndarray:
        return make_rotated_patch_target_vector(
            source_table,
            x0=x0,
            y0=y0,
            patch_size=patch_size or self.config.patch_size,
            max_objects=max_objects or self.config.max_objects,
            angle_encoding=self.config.angle_encoding,
        )

    def encode_for_network(self, boxes_xywhtheta: np.ndarray) -> np.ndarray:
        """Convert external `[cx, cy, w, h, theta]` boxes to network format."""

        boxes = np.asarray(boxes_xywhtheta, dtype=np.float64)
        if self.config.angle_encoding == "theta":
            return np.asarray(boxes, dtype=np.float64)
        if self.config.angle_encoding == "internal":
            return geometry.xywhtheta_to_internal(boxes)
        raise ValueError("angle_encoding must be 'theta' or 'internal'.")

    def decode_from_network(self, encoded_boxes: np.ndarray) -> np.ndarray:
        """Convert network box encoding back to external `[cx, cy, w, h, theta]`."""

        encoded = np.asarray(encoded_boxes, dtype=np.float64)
        if self.config.angle_encoding == "theta":
            if encoded.shape[-1] != 5:
                raise ValueError("theta encoding expects boxes with 5 values.")
            return np.stack(
                geometry.canonicalize_le90(
                    encoded[..., 0],
                    encoded[..., 1],
                    encoded[..., 2],
                    encoded[..., 3],
                    encoded[..., 4],
                ),
                axis=-1,
            )
        if self.config.angle_encoding == "internal":
            return geometry.internal_to_xywhtheta(encoded)
        raise ValueError("angle_encoding must be 'theta' or 'internal'.")

    @staticmethod
    def to_horizontal_envelope(boxes_xywhtheta: np.ndarray) -> np.ndarray:
        """Return `[xmin, ymin, xmax, ymax]` envelopes for HBB-compatible code."""

        return geometry.hbb_from_obb(boxes_xywhtheta)

    @staticmethod
    def corners(boxes_xywhtheta: np.ndarray) -> np.ndarray:
        return geometry.xywhtheta_to_corners(boxes_xywhtheta)

    @staticmethod
    def rotated_iou(box_a, box_b) -> float:
        return geometry.rotated_iou(box_a, box_b)
