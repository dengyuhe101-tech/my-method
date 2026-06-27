"""Experimental angle-weight variants for targeted small elongated sources."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SmallElongatedBoostConfig:
    """Boost only small sources that also have clear aspect ratio."""

    small_sqrt_area_cutoff: float = 8.0
    elongated_aspect_cutoff: float = 1.5
    boost_factor: float = 1.4
    max_weight: float = 2.0


def boost_small_elongated_weights(base_weights, aspect_ratio, sqrt_area_pix, config: SmallElongatedBoostConfig | None = None):
    """Return a copy of ``base_weights`` with a targeted small-source boost.

    The boost only applies to sources whose ``sqrt_area_pix`` is small and whose
    aspect ratio is clearly elongated. Near-square small sources remain unchanged.
    """

    config = config or SmallElongatedBoostConfig()
    base_weights = np.asarray(base_weights, dtype=np.float64)
    aspect_ratio = np.asarray(aspect_ratio, dtype=np.float64)
    sqrt_area_pix = np.asarray(sqrt_area_pix, dtype=np.float64)
    if base_weights.shape != aspect_ratio.shape or base_weights.shape != sqrt_area_pix.shape:
        raise ValueError("base_weights, aspect_ratio and sqrt_area_pix must have the same shape.")

    boosted = np.array(base_weights, dtype=np.float64, copy=True)
    mask = (sqrt_area_pix < config.small_sqrt_area_cutoff) & (aspect_ratio >= config.elongated_aspect_cutoff)
    boosted[mask] = np.minimum(config.max_weight, boosted[mask] * config.boost_factor)
    return boosted, mask

