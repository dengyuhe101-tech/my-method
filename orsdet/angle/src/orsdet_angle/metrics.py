"""Angle prediction metrics and grouping summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .angle_codec import angle_diff_le90_deg, decode_angle_vector
from .angle_loss import AngleLossConfig, angle_loss_and_grad
from .tables import AngleTargetTable


@dataclass
class PredictionMetrics:
    source_id: np.ndarray
    target_theta_deg: np.ndarray
    pred_cos_2theta: np.ndarray
    pred_sin_2theta: np.ndarray
    pred_theta_deg: np.ndarray
    signed_error_deg: np.ndarray
    abs_error_deg: np.ndarray
    vector_loss: np.ndarray
    unit_loss: np.ndarray
    angle_weight: np.ndarray
    aspect_ratio: np.ndarray
    sqrt_area_pix: np.ndarray
    flux_jy: np.ndarray


def prediction_metrics(
    table: AngleTargetTable,
    pred_vectors,
    loss_config: AngleLossConfig | None = None,
) -> PredictionMetrics:
    pred_vectors = np.asarray(pred_vectors, dtype=np.float64)
    if pred_vectors.shape != table.target_vectors.shape:
        raise ValueError("pred_vectors must have shape (N, 2), matching the angle target table.")

    loss_config = loss_config or AngleLossConfig(reduction="mean")
    _, _, per_sample, unit_loss = angle_loss_and_grad(
        pred_vectors,
        table.target_vectors,
        table.weights,
        loss_config,
    )
    pred_theta = decode_angle_vector(pred_vectors, normalize=True)
    signed_error = angle_diff_le90_deg(pred_theta, table.theta_deg)
    return PredictionMetrics(
        source_id=table.source_id,
        target_theta_deg=table.theta_deg,
        pred_cos_2theta=pred_vectors[:, 0],
        pred_sin_2theta=pred_vectors[:, 1],
        pred_theta_deg=pred_theta,
        signed_error_deg=signed_error,
        abs_error_deg=np.abs(signed_error),
        vector_loss=per_sample,
        unit_loss=unit_loss,
        angle_weight=table.weights,
        aspect_ratio=table.col("aspect_ratio"),
        sqrt_area_pix=table.col("sqrt_area_pix"),
        flux_jy=table.col("flux_jy"),
    )


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_quantile(values, q):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def summarize_mask(metrics: PredictionMetrics, mask, group_name: str, group_value: str) -> Dict[str, float | str]:
    mask = np.asarray(mask, dtype=bool)
    n = int(np.sum(mask))
    if n == 0:
        return {
            "group": group_name,
            "value": group_value,
            "count": 0,
            "mae_deg": float("nan"),
            "median_abs_deg": float("nan"),
            "p90_abs_deg": float("nan"),
            "weighted_mae_deg": float("nan"),
            "mean_vector_loss": float("nan"),
            "mean_unit_loss": float("nan"),
            "mean_weight": float("nan"),
            "mean_aspect": float("nan"),
            "mean_sqrt_area_pix": float("nan"),
            "mean_flux_jy": float("nan"),
        }

    abs_err = metrics.abs_error_deg[mask]
    weights = metrics.angle_weight[mask]
    return {
        "group": group_name,
        "value": group_value,
        "count": n,
        "mae_deg": _safe_mean(abs_err),
        "median_abs_deg": _safe_quantile(abs_err, 0.5),
        "p90_abs_deg": _safe_quantile(abs_err, 0.9),
        "weighted_mae_deg": float(np.sum(abs_err * weights) / max(np.sum(weights), 1.0e-8)),
        "mean_vector_loss": _safe_mean(metrics.vector_loss[mask]),
        "mean_unit_loss": _safe_mean(metrics.unit_loss[mask]),
        "mean_weight": _safe_mean(weights),
        "mean_aspect": _safe_mean(metrics.aspect_ratio[mask]),
        "mean_sqrt_area_pix": _safe_mean(metrics.sqrt_area_pix[mask]),
        "mean_flux_jy": _safe_mean(metrics.flux_jy[mask]),
    }


def fixed_group_masks(values, bins: Iterable[Tuple[str, float, float]]):
    values = np.asarray(values, dtype=np.float64)
    for label, low, high in bins:
        yield label, (values >= low) & (values < high)


def quantile_group_masks(values, labels: Iterable[str]):
    labels = list(labels)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        for label in labels:
            yield label, np.zeros(values.shape, dtype=bool)
        return

    edges = np.quantile(values[finite], np.linspace(0.0, 1.0, len(labels) + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i, label in enumerate(labels):
        yield label, (values >= edges[i]) & (values < edges[i + 1])


def grouped_summaries(metrics: PredictionMetrics) -> List[Dict[str, float | str]]:
    summaries: List[Dict[str, float | str]] = []
    all_mask = np.ones_like(metrics.abs_error_deg, dtype=bool)
    summaries.append(summarize_mask(metrics, all_mask, "all", "all"))

    aspect_bins = (
        ("near_square_[1,1.15)", 1.0, 1.15),
        ("mild_[1.15,1.5)", 1.15, 1.5),
        ("elongated_[1.5,3)", 1.5, 3.0),
        ("very_elongated_[3,inf)", 3.0, np.inf),
    )
    for label, mask in fixed_group_masks(metrics.aspect_ratio, aspect_bins):
        summaries.append(summarize_mask(metrics, mask, "aspect", label))

    size_bins = (
        ("small_sqrt_area_[0,8)", 0.0, 8.0),
        ("medium_sqrt_area_[8,16)", 8.0, 16.0),
        ("large_sqrt_area_[16,32)", 16.0, 32.0),
        ("xlarge_sqrt_area_[32,inf)", 32.0, np.inf),
    )
    for label, mask in fixed_group_masks(metrics.sqrt_area_pix, size_bins):
        summaries.append(summarize_mask(metrics, mask, "size", label))

    log_flux = np.log10(np.maximum(metrics.flux_jy, 1.0e-20))
    for label, mask in quantile_group_masks(log_flux, ("q1_low", "q2_midlow", "q3_midhigh", "q4_high")):
        summaries.append(summarize_mask(metrics, mask, "flux_quantile", label))

    return summaries


def write_summary_csv(summaries: List[Dict[str, float | str]], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "group",
        "value",
        "count",
        "mae_deg",
        "median_abs_deg",
        "p90_abs_deg",
        "weighted_mae_deg",
        "mean_vector_loss",
        "mean_unit_loss",
        "mean_weight",
        "mean_aspect",
        "mean_sqrt_area_pix",
        "mean_flux_jy",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in summaries:
            values = []
            for col in columns:
                value = row[col]
                if isinstance(value, str):
                    values.append(value)
                elif isinstance(value, int):
                    values.append(str(value))
                else:
                    values.append("%.10g" % value)
            f.write(",".join(values) + "\n")


def write_prediction_csv(metrics: PredictionMetrics, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack(
        [
            metrics.source_id,
            metrics.target_theta_deg,
            metrics.pred_cos_2theta,
            metrics.pred_sin_2theta,
            metrics.pred_theta_deg,
            metrics.signed_error_deg,
            metrics.abs_error_deg,
            metrics.vector_loss,
            metrics.unit_loss,
            metrics.angle_weight,
            metrics.aspect_ratio,
            metrics.sqrt_area_pix,
            metrics.flux_jy,
        ]
    )
    header = (
        "source_id,target_theta_deg,pred_cos_2theta,pred_sin_2theta,pred_theta_deg,"
        "signed_error_deg,abs_error_deg,"
        "vector_loss,unit_loss,angle_weight,aspect_ratio,sqrt_area_pix,flux_jy"
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.10g")


def format_summary_text(summaries: List[Dict[str, float | str]]) -> str:
    lines = []
    lines.append("angle angle prediction grouped summary")
    lines.append("")
    lines.append(
        "%-15s %-28s %8s %10s %10s %10s %10s %10s"
        % ("group", "value", "count", "MAE", "median", "P90", "wMAE", "loss")
    )
    for row in summaries:
        lines.append(
            "%-15s %-28s %8d %10.4f %10.4f %10.4f %10.4f %10.6g"
            % (
                row["group"],
                row["value"],
                row["count"],
                row["mae_deg"],
                row["median_abs_deg"],
                row["p90_abs_deg"],
                row["weighted_mae_deg"],
                row["mean_vector_loss"],
            )
        )
    lines.append("")
    return "\n".join(lines)
