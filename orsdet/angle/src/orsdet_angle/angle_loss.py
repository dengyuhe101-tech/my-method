"""Aspect-weighted angle loss for the angle OBB angle head."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1.0e-8


@dataclass(frozen=True)
class AspectWeightConfig:
    """Aspect-ratio weighting for angle supervision.

    Near-square OBBs have ambiguous orientation and should not dominate the
    angle gradient. Elongated OBBs have a clearer long-axis direction and can
    receive stronger angle supervision.
    """

    square_aspect: float = 1.15
    elongated_aspect: float = 3.0
    min_weight: float = 0.08
    max_weight: float = 2.0
    gamma: float = 1.5


@dataclass(frozen=True)
class AngleLossConfig:
    """Loss configuration for encoded angle vectors."""

    reduction: str = "mean"
    normalize_prediction_for_loss: bool = False
    unit_norm_weight: float = 0.02
    eps: float = EPS


def angle_weight_from_aspect(aspect, config: AspectWeightConfig | None = None):
    """Return per-source angle weights from aspect ratio."""

    config = config or AspectWeightConfig()
    aspect = np.asarray(aspect, dtype=np.float64)
    denom = max(config.elongated_aspect - config.square_aspect, config.min_weight)
    t = (aspect - config.square_aspect) / denom
    t = np.clip(t, 0.0, 1.0)
    t = np.power(t, config.gamma)
    return config.min_weight + (config.max_weight - config.min_weight) * t


def _normalize_vectors(vectors, eps):
    norm = np.maximum(np.sqrt(np.sum(vectors * vectors, axis=-1, keepdims=True)), eps)
    return vectors / norm


def angle_loss_and_grad(pred_vectors, target_vectors, weights=None, config: AngleLossConfig | None = None):
    """Compute weighted encoded-angle MSE and its gradient w.r.t. predictions.

    The main loss is:

        0.5 * weight * ||pred - target||^2

    A small unit-norm regularizer can be added so the two linear outputs keep a
    decodable cosine/sine magnitude.
    """

    config = config or AngleLossConfig()
    pred = np.asarray(pred_vectors, dtype=np.float64)
    target = np.asarray(target_vectors, dtype=np.float64)
    if pred.shape != target.shape or pred.shape[-1] != 2:
        raise ValueError("pred_vectors and target_vectors must have the same shape (..., 2).")

    if weights is None:
        weights = np.ones(pred.shape[:-1], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != pred.shape[:-1]:
        raise ValueError("weights must match pred_vectors shape without the angle dimension.")

    effective_pred = _normalize_vectors(pred, config.eps) if config.normalize_prediction_for_loss else pred
    diff = effective_pred - target
    per_sample = 0.5 * weights * np.sum(diff * diff, axis=-1)
    grad = weights[..., None] * diff

    unit_loss = np.zeros_like(per_sample)
    if config.unit_norm_weight > 0.0:
        norm = np.maximum(np.sqrt(np.sum(pred * pred, axis=-1)), config.eps)
        norm_diff = norm - 1.0
        unit_loss = 0.5 * config.unit_norm_weight * norm_diff * norm_diff
        grad += config.unit_norm_weight * norm_diff[..., None] * pred / norm[..., None]

    total = per_sample + unit_loss
    if config.reduction == "mean":
        denom = max(float(np.size(total)), 1.0)
        return float(np.sum(total) / denom), grad / denom, per_sample, unit_loss
    if config.reduction == "sum":
        return float(np.sum(total)), grad, per_sample, unit_loss
    if config.reduction == "none":
        return total, grad, per_sample, unit_loss
    raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
