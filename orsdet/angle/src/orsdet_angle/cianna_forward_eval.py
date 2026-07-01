"""Evaluate native CIANNA YOLO angle-head forward outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .angle_codec import angle_diff_le90_deg, decode_angle_vector
from .angle_loss import AngleLossConfig, angle_loss_and_grad
from .cianna_targets import ANGLE_NB_ANGLE, ANGLE_NB_PARAM, AngleTargetSpec
from .metrics import (
    PredictionMetrics,
    format_summary_text,
    grouped_summaries,
    write_prediction_csv,
    write_summary_csv,
)


def parse_yolo_forward(path: Path, n_samples: int, grid_size: int, nb_filters: int) -> np.ndarray:
    """Parse text or flat CIANNA YOLO forward output into (N, C, grid^2)."""

    rows = np.loadtxt(path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    batch_offset = grid_size * grid_size
    expected = n_samples * nb_filters * batch_offset
    flat = rows.reshape(-1)
    if flat.size != expected:
        raise ValueError("Unexpected forward element count: got %d expected %d" % (flat.size, expected))
    return flat.reshape(n_samples, nb_filters, batch_offset)


def _denorm_log_param(norm_values: np.ndarray, lims: np.ndarray, row: int) -> np.ndarray:
    log_values = norm_values * (lims[row, 0] - lims[row, 1]) + lims[row, 1]
    return np.exp(log_values)


def metrics_from_valid_targets_and_forward(
    targets: np.ndarray,
    forward: np.ndarray,
    *,
    image_size: int,
    grid_size: int,
    nb_box: int,
    lims=None,
    nb_param: int = ANGLE_NB_PARAM,
    nb_angle: int = ANGLE_NB_ANGLE,
    unit_norm_weight: float = 0.02,
) -> PredictionMetrics:
    """Compare target-cell angle vectors with the highest-objectness box."""

    target_spec = AngleTargetSpec(nb_param=nb_param, nb_angle=nb_angle)
    target_stride = target_spec.target_stride
    output_stride = 8 + nb_param + nb_angle
    if forward.shape[1] != nb_box * output_stride:
        raise ValueError("forward channel count does not match nb_box/output stride.")

    lims = None if lims is None else np.asarray(lims, dtype=np.float64)
    cell_size = float(image_size) / float(grid_size)
    source_ids = []
    target_vectors = []
    pred_vectors = []
    weights = []
    aspects = []
    sqrt_areas = []
    fluxes = []

    for sample_i in range(targets.shape[0]):
        n_obj = int(min(max(targets[sample_i, 0], 0), (targets.shape[1] - 1) // target_stride))
        for obj_i in range(n_obj):
            start = 1 + obj_i * target_stride
            block = targets[sample_i, start : start + target_stride]
            xmin, ymin, xmax, ymax = block[1], block[2], block[4], block[5]
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            gx = int(np.clip(cx / cell_size, 0, grid_size - 1))
            gy = int(np.clip(cy / cell_size, 0, grid_size - 1))
            cell = gy * grid_size + gx

            objectness = np.asarray(
                [forward[sample_i, box_i * output_stride + 7, cell] for box_i in range(nb_box)],
                dtype=np.float64,
            )
            box_i = int(np.argmax(objectness))
            angle_start = box_i * output_stride + 8 + nb_param

            target_vectors.append(block[7 + nb_param : 7 + nb_param + nb_angle])
            pred_vectors.append(forward[sample_i, angle_start : angle_start + nb_angle, cell])
            weights.append(block[7 + nb_param + nb_angle])
            source_ids.append(sample_i * 1000 + obj_i)

            width = max(float(abs(xmax - xmin)), 1.0e-8)
            height = max(float(abs(ymax - ymin)), 1.0e-8)
            sqrt_areas.append(np.sqrt(width * height))

            if lims is not None:
                fluxes.append(float(_denorm_log_param(np.asarray([block[7]]), lims, 0)[0]))
                bmaj = float(_denorm_log_param(np.asarray([block[8]]), lims, 1)[0])
                bmin = float(_denorm_log_param(np.asarray([block[9]]), lims, 2)[0])
                aspects.append(max(bmaj, bmin) / max(min(bmaj, bmin), 1.0e-8))
            else:
                fluxes.append(float(block[7]))
                aspects.append(max(width, height) / max(min(width, height), 1.0e-8))

    target_vectors = np.asarray(target_vectors, dtype=np.float64)
    pred_vectors = np.asarray(pred_vectors, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if target_vectors.size == 0:
        raise ValueError("No active targets found for angle validation.")

    _, _, vector_loss, unit_loss = angle_loss_and_grad(
        pred_vectors,
        target_vectors,
        weights,
        AngleLossConfig(reduction="mean", unit_norm_weight=unit_norm_weight),
    )
    target_theta = decode_angle_vector(target_vectors, normalize=True)
    pred_theta = decode_angle_vector(pred_vectors, normalize=True)
    signed_error = angle_diff_le90_deg(pred_theta, target_theta)

    return PredictionMetrics(
        source_id=np.asarray(source_ids, dtype=np.int64),
        target_theta_deg=target_theta,
        pred_cos_2theta=pred_vectors[:, 0],
        pred_sin_2theta=pred_vectors[:, 1],
        pred_theta_deg=pred_theta,
        signed_error_deg=signed_error,
        abs_error_deg=np.abs(signed_error),
        vector_loss=vector_loss,
        unit_loss=unit_loss,
        angle_weight=weights,
        aspect_ratio=np.asarray(aspects, dtype=np.float64),
        sqrt_area_pix=np.asarray(sqrt_areas, dtype=np.float64),
        flux_jy=np.asarray(fluxes, dtype=np.float64),
    )


def write_valid_angle_report(metrics: PredictionMetrics, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = grouped_summaries(metrics)
    write_prediction_csv(metrics, output_dir / "metrics.csv")
    write_summary_csv(summaries, output_dir / "groups.csv")
    text = format_summary_text(summaries)
    text += "\n" + angle_distribution_report(metrics, output_dir)
    with (output_dir / "report.txt").open("w", encoding="utf-8") as f:
        f.write(text)
    return text


def summarize_all_metrics(metrics: PredictionMetrics) -> dict[str, float]:
    abs_err = np.asarray(metrics.abs_error_deg, dtype=np.float64)
    weights = np.asarray(metrics.angle_weight, dtype=np.float64)
    return {
        "count": float(abs_err.size),
        "mae_deg": float(np.mean(abs_err)),
        "median_abs_deg": float(np.quantile(abs_err, 0.5)),
        "p90_abs_deg": float(np.quantile(abs_err, 0.9)),
        "weighted_mae_deg": float(np.sum(abs_err * weights) / max(np.sum(weights), 1.0e-8)),
        "mean_vector_loss": float(np.mean(metrics.vector_loss)),
        "mean_unit_loss": float(np.mean(metrics.unit_loss)),
        "pred_boundary_frac": float(np.mean(np.abs(metrics.pred_theta_deg) > 85.0)),
        "target_boundary_frac": float(np.mean(np.abs(metrics.target_theta_deg) > 85.0)),
        "wrap_cross_frac": float(
            np.mean(
                ((metrics.target_theta_deg < -80.0) & (metrics.pred_theta_deg > 80.0))
                | ((metrics.target_theta_deg > 80.0) & (metrics.pred_theta_deg < -80.0))
            )
        ),
    }


def append_angle_history_row(path: Path, epoch: int, metrics: PredictionMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = summarize_all_metrics(metrics)
    columns = ["epoch"] + list(row.keys())
    exists = path.is_file()
    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(columns) + "\n")
        f.write("%d,%s\n" % (epoch, ",".join("%.10g" % row[col] for col in row.keys())))


def angle_distribution_report(metrics: PredictionMetrics, output_dir: Path) -> str:
    """Write theta/error histograms and a compact boundary-jump diagnostic."""

    output_dir.mkdir(parents=True, exist_ok=True)
    theta_bins = np.linspace(-90.0, 90.0, 37)
    err_bins = np.linspace(0.0, 90.0, 31)
    target_hist, _ = np.histogram(metrics.target_theta_deg, bins=theta_bins)
    pred_hist, _ = np.histogram(metrics.pred_theta_deg, bins=theta_bins)
    err_hist, _ = np.histogram(metrics.abs_error_deg, bins=err_bins)

    hist_path = output_dir / "hist.csv"
    with hist_path.open("w", encoding="utf-8") as f:
        f.write("kind,bin_left,bin_right,count\n")
        for left, right, count in zip(theta_bins[:-1], theta_bins[1:], target_hist):
            f.write("target_theta,%.6g,%.6g,%d\n" % (left, right, int(count)))
        for left, right, count in zip(theta_bins[:-1], theta_bins[1:], pred_hist):
            f.write("pred_theta,%.6g,%.6g,%d\n" % (left, right, int(count)))
        for left, right, count in zip(err_bins[:-1], err_bins[1:], err_hist):
            f.write("abs_error,%.6g,%.6g,%d\n" % (left, right, int(count)))

    summary = summarize_all_metrics(metrics)
    lines = [
        "Angle distribution and boundary diagnostic",
        "count: %d" % int(summary["count"]),
        "pred_boundary_frac_abs_theta_gt_85: %.8g" % summary["pred_boundary_frac"],
        "target_boundary_frac_abs_theta_gt_85: %.8g" % summary["target_boundary_frac"],
        "wrap_cross_frac_target_pred_opposite_edges: %.8g" % summary["wrap_cross_frac"],
        "histogram_csv: %s" % hist_path,
    ]
    text = "\n".join(lines) + "\n"
    with (output_dir / "boundary.txt").open("w", encoding="utf-8") as f:
        f.write(text)
    return text
