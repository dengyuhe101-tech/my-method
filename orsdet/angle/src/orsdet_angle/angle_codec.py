"""Angle encoding helpers for le90 OBB regression.

External theta convention is the geometry convention:

    theta in [-90, 90), measured against the long side.

The internal regression vector is:

    [cos(2 theta), sin(2 theta)]

This is the Nstep=2, omega=2 PSC special case. It is continuous at the
le90 boundary and respects the 180 degree equivalence of oriented boxes.
"""

from __future__ import annotations

import numpy as np


LE90_MIN_DEG = -90.0
LE90_PERIOD_DEG = 180.0
EPS = 1.0e-8


def normalize_theta_le90(theta_deg):
    """Normalize angles to [-90, 90)."""

    theta = np.asarray(theta_deg, dtype=np.float64)
    out = np.mod(theta - LE90_MIN_DEG, LE90_PERIOD_DEG) + LE90_MIN_DEG
    return out.item() if out.ndim == 0 else out


def angle_diff_le90_deg(pred_deg, target_deg):
    """Smallest signed difference between two le90 angles in degrees."""

    return normalize_theta_le90(np.asarray(pred_deg, dtype=np.float64) - np.asarray(target_deg, dtype=np.float64))


def encode_theta_le90(theta_deg):
    """Encode theta to [cos(2 theta), sin(2 theta)]."""

    theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    return np.stack([np.cos(2.0 * theta_rad), np.sin(2.0 * theta_rad)], axis=-1)


def decode_angle_vector(vectors, normalize=True):
    """Decode [cos(2 theta), sin(2 theta)] vectors to le90 theta degrees."""

    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[-1] != 2:
        raise ValueError("Expected angle vectors with last dimension 2.")

    cos2 = vectors[..., 0]
    sin2 = vectors[..., 1]
    if normalize:
        norm = np.maximum(np.sqrt(cos2 * cos2 + sin2 * sin2), EPS)
        cos2 = cos2 / norm
        sin2 = sin2 / norm

    theta = 0.5 * np.rad2deg(np.arctan2(sin2, cos2))
    return normalize_theta_le90(theta)
