"""Small bridge helpers for a future native CIANNA angle head.

The current V2 module is intentionally external to the C/CUDA YOLO layer.
These helpers define the contract that the future CIANNA integration should
use for its angle target channels and prediction decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .angle_codec import decode_angle_vector, encode_theta_le90
from .angle_loss import AspectWeightConfig, angle_weight_from_aspect


@dataclass(frozen=True)
class V2AngleHeadSpec:
    angle_channels: int = 2
    target_columns: tuple[str, ...] = ("cos_2theta", "sin_2theta", "angle_weight")
    prediction_columns: tuple[str, ...] = ("pred_cos_2theta", "pred_sin_2theta")
    theta_name: str = "theta_le90_deg"


def make_angle_target_channels(theta_le90_deg, aspect_ratio, weight_config: AspectWeightConfig | None = None):
    """Return `[cos(2theta), sin(2theta), angle_weight]` target channels."""

    encoded = encode_theta_le90(theta_le90_deg)
    weights = angle_weight_from_aspect(aspect_ratio, weight_config)
    return np.column_stack([encoded[:, 0], encoded[:, 1], weights])


def decode_angle_prediction(pred_cos_2theta, pred_sin_2theta):
    """Decode future CIANNA angle-head outputs to le90 theta degrees."""

    return decode_angle_vector(np.column_stack([pred_cos_2theta, pred_sin_2theta]), normalize=True)
