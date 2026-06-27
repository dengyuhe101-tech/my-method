#!/usr/bin/env python3
"""Lightweight V4d candidate decode plus V3 OBB NMS diagnostic."""

from __future__ import annotations

import argparse
import os
import site
import sys
from pathlib import Path

from env_guard import drop_user_site as drop_user_site_before_imports

drop_user_site_before_imports()

import numpy as np


def drop_user_site():
    os.environ["PYTHONNOUSERSITE"] = "1"
    try:
        user_paths = site.getusersitepackages()
    except Exception:
        user_paths = []
    if isinstance(user_paths, str):
        user_paths = [user_paths]
    filtered = []
    resolved_user_paths = []
    for path in user_paths:
        if not path:
            continue
        try:
            resolved_user_paths.append(Path(path).expanduser().resolve())
        except OSError:
            pass
    for path_entry in sys.path:
        if not path_entry:
            filtered.append(path_entry)
            continue
        try:
            resolved = Path(path_entry).expanduser().resolve()
        except OSError:
            filtered.append(path_entry)
            continue
        if any(resolved == user_path or user_path in resolved.parents for user_path in resolved_user_paths):
            continue
        filtered.append(path_entry)
    sys.path[:] = filtered


drop_user_site()

SCRIPT_DIR = Path(__file__).resolve().parent
V4D_DIR = SCRIPT_DIR.parent
SKAO_DIR = V4D_DIR.parent
V3_DIR = SKAO_DIR / "nms"
for path in (
    V4D_DIR / "src",
    V3_DIR / "src",
    SKAO_DIR / "candidates" / "src",
    SKAO_DIR / "angle" / "src",
    SKAO_DIR / "target_source" / "src",
    SKAO_DIR / "geometry" / "src",
    SKAO_DIR,
):
    sys.path.insert(0, str(path))


def expected_floats(nb_area_h, nb_area_w, nb_box, channels, yolo_nb_reg):
    return int(nb_area_h * nb_area_w * nb_box * channels * yolo_nb_reg * yolo_nb_reg)


def top_indices(values: np.ndarray, count: int, threshold: float):
    flat = values.reshape(-1)
    if threshold > 0.0:
        valid = np.flatnonzero(flat >= threshold)
        if valid.size == 0:
            return valid
        if valid.size > count:
            local = np.argpartition(flat[valid], -count)[-count:]
            valid = valid[local]
        return valid[np.argsort(flat[valid])[::-1]]
    if flat.size <= count:
        valid = np.arange(flat.size, dtype=np.int64)
    else:
        valid = np.argpartition(flat, -count)[-count:]
    return valid[np.argsort(flat[valid])[::-1]]


def collect_candidates(
    pred,
    af,
    *,
    nb_aux: int,
    max_candidates: int,
    per_prior_per_tile: int,
    obj_threshold: float,
    out_range: int,
):
    channels = 8 + nb_aux
    rows = []
    total_scanned = 0
    total_above_threshold = 0

    for ph in range(af.nb_area_h):
        y_offset = ph * af.patch_shift - af.orig_offset
        for pw in range(af.nb_area_w):
            x_offset = pw * af.patch_shift - af.orig_offset
            if ph < out_range or ph >= af.nb_area_h - out_range or pw < out_range or pw >= af.nb_area_w - out_range:
                continue
            c_pred = pred[ph, pw]
            for prior_id in range(af.nb_box):
                offset = prior_id * channels
                obj = np.asarray(c_pred[offset + 7], dtype=np.float64)
                total_scanned += obj.size
                total_above_threshold += int(np.count_nonzero(obj >= obj_threshold))
                selected = top_indices(obj, per_prior_per_tile, obj_threshold)
                if selected.size == 0:
                    continue
                yy, xx = np.divmod(selected, af.yolo_nb_reg)
                for y_idx, x_idx in zip(yy, xx):
                    x0 = float(c_pred[offset + 0, y_idx, x_idx]) + x_offset
                    y0 = float(c_pred[offset + 1, y_idx, x_idx]) + y_offset
                    x1 = float(c_pred[offset + 3, y_idx, x_idx]) + x_offset
                    y1 = float(c_pred[offset + 4, y_idx, x_idx]) + y_offset
                    prob = float(c_pred[offset + 6, y_idx, x_idx])
                    objectness = float(c_pred[offset + 7, y_idx, x_idx])
                    params = [float(c_pred[offset + 8 + p, y_idx, x_idx]) for p in range(nb_aux)]
                    rows.append([x0, y0, x1, y1, prob, objectness, prior_id, *params, y_idx * af.yolo_nb_reg + x_idx])

    if not rows:
        return np.zeros((0, 7 + nb_aux + 1), dtype=np.float64), total_scanned, total_above_threshold

    rows = np.asarray(rows, dtype=np.float64)
    order = np.argsort(rows[:, 5])[::-1]
    rows = rows[order[:max_candidates]]
    return rows, total_scanned, total_above_threshold


def write_rows_csv(path: Path, rows: np.ndarray, boxes: np.ndarray, slim_mode: str):
    if slim_mode == "v4d_s":
        aux_header = "flux_norm,phys_bmaj_norm,phys_bmin_norm,obb_cos2,obb_sin2,phys_pa_cos2,phys_pa_sin2"
    elif slim_mode == "v4k_dsa":
        aux_header = "flux_norm,phys_bmaj_norm,phys_bmin_norm,delta_log_flux_raw,flux_gate_raw,shared_cos2,shared_sin2"
    elif slim_mode == "v4i_dsa":
        aux_header = "flux_norm,phys_bmaj_norm,phys_bmin_norm,delta_log_flux_raw,shared_cos2,shared_sin2"
    else:
        aux_header = "flux_norm,phys_bmaj_norm,phys_bmin_norm,shared_cos2,shared_sin2"
    header = (
        "xmin,ymin,xmax,ymax,probability,objectness,prior_id,"
        + aux_header
        + ",grid_index,cx_pix,cy_pix,obb_w_pix,obb_h_pix,theta_le90_deg"
    )
    if rows.shape[0] == 0:
        path.write_text(header + "\n", encoding="utf-8")
        return
    out = np.column_stack([rows, boxes])
    np.savetxt(path, out, delimiter=",", header=header, comments="", fmt="%.10g")


def main():
    from orsdet_nms import local_nms
    from orsdet_detector import DEFAULT_RUN_DIR, configure_paths, decode_rows_obb, install_numba_fallback_if_needed
    from orsdet_detector import normalize_slim_mode, v4d_layout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epoch", nargs="?", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-candidates", type=int, default=1200)
    parser.add_argument("--per-prior-per-tile", type=int, default=1)
    parser.add_argument("--obj-threshold", type=float, default=0.0)
    parser.add_argument("--out-range", type=int, default=2)
    parser.add_argument(
        "--slim-mode",
        default="v4d_s",
        choices=("v4d_s", "v4d-sa", "v4d_sa", "size", "size-angle", "size_angle"),
    )
    args = parser.parse_args()
    args.slim_mode = normalize_slim_mode(args.slim_mode)
    layout = v4d_layout(args.slim_mode)

    configure_paths()
    install_numba_fallback_if_needed()

    import aux_fct as af

    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "nms_check")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    lims = np.loadtxt(run_dir / "train_norm.txt")

    channels = 8 + layout.total_aux
    fwd_path = run_dir / "fwd_res" / ("net0_%04d.dat" % args.epoch)
    if not fwd_path.is_file():
        raise FileNotFoundError(fwd_path)
    exp = expected_floats(af.nb_area_h, af.nb_area_w, af.nb_box, channels, af.yolo_nb_reg)
    got = fwd_path.stat().st_size // 4
    if got != exp:
        raise ValueError("%s has %d float32 values, expected %d" % (fwd_path, got, exp))

    pred = np.memmap(
        fwd_path,
        dtype="float32",
        mode="r",
        shape=(af.nb_area_h, af.nb_area_w, af.nb_box * channels, af.yolo_nb_reg, af.yolo_nb_reg),
    )
    rows, total_scanned, total_above_threshold = collect_candidates(
        pred,
        af,
        nb_aux=layout.total_aux,
        max_candidates=args.max_candidates,
        per_prior_per_tile=args.per_prior_per_tile,
        obj_threshold=args.obj_threshold,
        out_range=args.out_range,
    )
    boxes = decode_rows_obb(rows, lims, layout.mode)
    kept_rows, kept_boxes = local_nms(rows, boxes)

    raw_path = out_dir / ("v4d_candidates_%04d.csv" % args.epoch)
    kept_path = out_dir / ("v4d_obb_nms_%04d.csv" % args.epoch)
    summary_path = out_dir / ("summary_%04d.txt" % args.epoch)
    write_rows_csv(raw_path, rows, boxes, layout.mode)
    write_rows_csv(kept_path, kept_rows, kept_boxes, layout.mode)

    finite_angle = int(np.count_nonzero(np.isfinite(kept_boxes[:, 4]))) if kept_boxes.shape[0] else 0
    lines = [
        "epoch: %d" % args.epoch,
        "fwd_path: %s" % fwd_path,
        "nms: orsdet_nms.local_nms",
        "slim_mode: %s" % layout.mode,
        "candidate_channels: box4 + prob + obj + prior + %d aux + grid_index" % layout.total_aux,
        "total_scanned_cells: %d" % total_scanned,
        "total_above_threshold: %d" % total_above_threshold,
        "sampled_before_nms: %d" % rows.shape[0],
        "kept_after_obb_nms: %d" % kept_rows.shape[0],
        "deleted_by_obb_nms: %d" % (rows.shape[0] - kept_rows.shape[0]),
        "finite_angle_kept: %d" % finite_angle,
        "max_objectness_before: %.8g" % (float(np.max(rows[:, 5])) if rows.shape[0] else 0.0),
        "max_objectness_after: %.8g" % (float(np.max(kept_rows[:, 5])) if kept_rows.shape[0] else 0.0),
        "raw_csv: %s" % raw_path,
        "kept_csv: %s" % kept_path,
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
