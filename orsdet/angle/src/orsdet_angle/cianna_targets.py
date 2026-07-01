"""Target conversion helpers for the native CIANNA angle angle head.

The original SKAO_SDC1 data generator writes five additional YOLO parameters:

    flux, bmaj, bmin, cos(PA), shifted_sin(PA)

angle keeps the first three scalar parameters in the existing parameter branch and
moves PA into a dedicated encoded angle branch:

    cos(2 PA), sin(2 PA), angle_weight

The target block therefore changes from 7 + 5 values to 7 + 3 + 2 + 1 values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .angle_codec import encode_theta_le90, normalize_theta_le90
from .angle_loss import AspectWeightConfig, angle_weight_from_aspect


LEGACY_NB_PARAM = 5
ANGLE_NB_PARAM = 3
ANGLE_NB_ANGLE = 2


@dataclass(frozen=True)
class AngleTargetSpec:
    nb_param: int = ANGLE_NB_PARAM
    nb_angle: int = ANGLE_NB_ANGLE
    angle_weight_channels: int = 1

    @property
    def target_stride(self) -> int:
        return 7 + self.nb_param + self.nb_angle + self.angle_weight_channels

    def target_dim(self, max_nb_obj_per_image: int) -> int:
        return 1 + max_nb_obj_per_image * self.target_stride


def decode_legacy_pa(cos_pa, shifted_sin_pa):
    """Decode original CIANNA PA channels to le90 degrees."""

    cos_pa = np.asarray(cos_pa, dtype=np.float64)
    sin_pa = 2.0 * np.asarray(shifted_sin_pa, dtype=np.float64) - 1.0
    pa = np.rad2deg(np.arctan2(sin_pa, cos_pa))
    return normalize_theta_le90(pa)


def aspect_from_legacy_params(bmaj_norm, bmin_norm, lims=None):
    """Recover aspect ratio from normalized original bmaj/bmin parameters."""

    bmaj_norm = np.asarray(bmaj_norm, dtype=np.float64)
    bmin_norm = np.asarray(bmin_norm, dtype=np.float64)
    if lims is None:
        return np.ones_like(bmaj_norm, dtype=np.float64)

    lims = np.asarray(lims, dtype=np.float64)
    bmaj_log = bmaj_norm * (lims[1, 0] - lims[1, 1]) + lims[1, 1]
    bmin_log = bmin_norm * (lims[2, 0] - lims[2, 1]) + lims[2, 1]
    bmaj = np.exp(bmaj_log)
    bmin = np.exp(bmin_log)
    return np.maximum(bmaj, bmin) / np.maximum(np.minimum(bmaj, bmin), 1.0e-8)


def aspect_from_box_block(block):
    """Fallback aspect ratio from the axis-aligned target box."""

    width = np.maximum(np.abs(block[:, 4] - block[:, 1]), 1.0e-8)
    height = np.maximum(np.abs(block[:, 5] - block[:, 2]), 1.0e-8)
    return np.maximum(width, height) / np.minimum(width, height)


def convert_legacy_targets_to_angle(
    legacy_targets,
    max_nb_obj_per_image: int,
    lims=None,
    weight_config: AspectWeightConfig | None = None,
    spec: AngleTargetSpec | None = None,
):
    """Convert original HBB+PA target arrays to angle angle-head target arrays."""

    spec = spec or AngleTargetSpec()
    legacy = np.asarray(legacy_targets, dtype=np.float32)
    if legacy.ndim != 2:
        raise ValueError("legacy_targets must be a 2-D array.")

    legacy_stride = 7 + LEGACY_NB_PARAM
    expected_legacy_dim = 1 + max_nb_obj_per_image * legacy_stride
    if legacy.shape[1] != expected_legacy_dim:
        raise ValueError(
            "legacy target dim mismatch: got %d, expected %d"
            % (legacy.shape[1], expected_legacy_dim)
        )

    out = np.zeros((legacy.shape[0], spec.target_dim(max_nb_obj_per_image)), dtype=np.float32)
    out[:, 0] = legacy[:, 0]

    for obj_i in range(max_nb_obj_per_image):
        old_start = 1 + obj_i * legacy_stride
        new_start = 1 + obj_i * spec.target_stride
        old_block = legacy[:, old_start : old_start + legacy_stride]
        new_block = out[:, new_start : new_start + spec.target_stride]

        active = obj_i < legacy[:, 0]
        if not np.any(active):
            continue

        new_block[active, 0:7] = old_block[active, 0:7]
        new_block[active, 7:10] = old_block[active, 7:10]

        theta_deg = decode_legacy_pa(old_block[active, 10], old_block[active, 11])
        encoded = encode_theta_le90(theta_deg)

        aspect = aspect_from_legacy_params(old_block[active, 8], old_block[active, 9], lims=lims)
        if lims is None:
            aspect = aspect_from_box_block(old_block[active])
        weights = angle_weight_from_aspect(aspect, weight_config)

        new_block[active, 10] = encoded[:, 0]
        new_block[active, 11] = encoded[:, 1]
        new_block[active, 12] = weights

    return out


def angle_target_dim(max_nb_obj_per_image: int) -> int:
    return AngleTargetSpec().target_dim(max_nb_obj_per_image)
