#!/usr/bin/env python3
"""Core Stage9 target building and ridge helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from ska_sdc.sdc1.utils import prep as scorer_prep

from flux_common import (
    EPS,
    FEATURE_COLUMNS,
    add_stage9_feature_expansion,
    build_detector_feature_frame,
    read_catalog,
    regression_metrics,
)


TRUTH_COLUMNS = [
    "truth_id",
    "ra_core",
    "dec_core",
    "ra_cent",
    "dec_cent",
    "flux",
    "core_frac",
    "b_maj",
    "b_min",
    "pa",
    "size_id",
    "class_id",
]
BASE_FEATURE_COLUMNS = list(FEATURE_COLUMNS) + ["bright_protected", "size_id", "class"]


def read_truth_catalog(path: Path) -> pd.DataFrame:
    truth = pd.read_csv(path, sep=r"\s+", header=None, names=TRUTH_COLUMNS, engine="c")
    truth = truth.dropna().reset_index(drop=True)
    truth = truth[
        (truth["flux"] > 0.0)
        & (truth["core_frac"] >= 0.0)
        & (truth["b_min"] >= 0.0)
        & (truth["b_maj"] >= 0.0)
    ].copy()
    truth.loc[truth["ra_core"] > 180.0, "ra_core"] -= 360.0
    truth.loc[truth["ra_cent"] > 180.0, "ra_cent"] -= 360.0
    return truth.reset_index(drop=True)


def training_mask(df: pd.DataFrame) -> pd.Series:
    lim = scorer_prep.TRAIN_LIM[560]
    return (
        (df["ra_core"] > lim["ra_min"])
        & (df["ra_core"] < lim["ra_max"])
        & (df["dec_core"] > lim["dec_min"])
        & (df["dec_core"] < lim["dec_max"])
    )


def planar_coords_arcsec(ra: np.ndarray, dec: np.ndarray, *, ra0: float, dec0: float) -> np.ndarray:
    ra_arr = np.asarray(ra, dtype=np.float64)
    dec_arr = np.asarray(dec, dtype=np.float64)
    cos_dec = float(np.cos(np.deg2rad(dec0)))
    x = (ra_arr - float(ra0)) * cos_dec * 3600.0
    y = (dec_arr - float(dec0)) * 3600.0
    return np.column_stack([x, y])


def candidate_radii_arcsec(features: pd.DataFrame, *, min_radius: float, max_radius: float, radius_scale: float) -> np.ndarray:
    size_proxy = np.sqrt(
        np.maximum(features["obb_bmaj_arcsec"].to_numpy(dtype=np.float64), EPS)
        * np.maximum(features["obb_bmin_arcsec"].to_numpy(dtype=np.float64), EPS)
    )
    return np.minimum(float(max_radius), np.maximum(float(min_radius), float(radius_scale) * size_proxy))


def pair_confidence_score(
    *,
    pair_dist: float,
    pair_radius: float,
    nearest_dist: float,
    second_dist: float,
    matched_is_nearest: bool,
    candidate_crowding: float,
    candidate_area: float,
    truth_area: float,
) -> tuple[float, dict[str, float]]:
    center_score = 1.0 - np.clip(float(pair_dist) / max(float(pair_radius), EPS), 0.0, 1.0)
    if np.isfinite(second_dist):
        margin_ratio = float(second_dist) / max(float(nearest_dist), EPS)
        margin_arcsec = float(second_dist) - float(nearest_dist)
        ambiguity_score = np.clip((margin_ratio - 1.0) / 3.0, 0.0, 1.0)
    else:
        margin_ratio = np.inf
        margin_arcsec = np.inf
        ambiguity_score = 1.0
    if not bool(matched_is_nearest):
        ambiguity_score *= 0.5
    crowding_score = np.clip((float(candidate_crowding) - 1.0) / 3.0, 0.0, 1.0) if np.isfinite(candidate_crowding) else 1.0
    area_ratio = float(candidate_area) / max(float(truth_area), EPS)
    size_score = float(np.exp(-abs(np.log(max(area_ratio, EPS))) / np.log(10.0)))
    confidence = 0.45 * center_score + 0.30 * ambiguity_score + 0.15 * crowding_score + 0.10 * size_score
    return float(np.clip(confidence, 0.0, 1.0)), {
        "pair_center_score": float(center_score),
        "pair_ambiguity_score": float(ambiguity_score),
        "pair_crowding_score": float(crowding_score),
        "pair_size_score": float(size_score),
        "nearest_second_margin_arcsec": float(margin_arcsec),
        "nearest_second_ratio": float(margin_ratio),
    }


def greedy_geometric_pairing(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    min_radius_arcsec: float,
    max_radius_arcsec: float,
    radius_scale: float,
) -> pd.DataFrame:
    if candidates.empty or truth.empty:
        return pd.DataFrame()

    ra0 = float(np.nanmedian(np.concatenate([candidates["ra_core"].to_numpy(), truth["ra_core"].to_numpy()])))
    dec0 = float(np.nanmedian(np.concatenate([candidates["dec_core"].to_numpy(), truth["dec_core"].to_numpy()])))
    cand_xy = planar_coords_arcsec(
        candidates["ra_core"].to_numpy(dtype=np.float64),
        candidates["dec_core"].to_numpy(dtype=np.float64),
        ra0=ra0,
        dec0=dec0,
    )
    truth_xy = planar_coords_arcsec(
        truth["ra_core"].to_numpy(dtype=np.float64),
        truth["dec_core"].to_numpy(dtype=np.float64),
        ra0=ra0,
        dec0=dec0,
    )
    radii = candidate_radii_arcsec(
        candidates,
        min_radius=float(min_radius_arcsec),
        max_radius=float(max_radius_arcsec),
        radius_scale=float(radius_scale),
    )

    truth_tree = cKDTree(truth_xy)
    k_truth = min(2, len(truth_xy))
    nearest_dist, nearest_idx = truth_tree.query(cand_xy, k=k_truth, workers=-1)
    if k_truth == 1:
        nearest_dist_all = np.asarray(nearest_dist, dtype=np.float64)
        nearest_idx_all = np.asarray(nearest_idx, dtype=np.int64)
        second_dist_all = np.full(len(candidates), np.inf, dtype=np.float64)
    else:
        nearest_dist_all = np.asarray(nearest_dist[:, 0], dtype=np.float64)
        nearest_idx_all = np.asarray(nearest_idx[:, 0], dtype=np.int64)
        second_dist_all = np.asarray(nearest_dist[:, 1], dtype=np.float64)

    proposals: list[tuple[float, float, int, int]] = []
    for cand_i, truth_ids in enumerate(truth_tree.query_ball_point(cand_xy, r=float(np.max(radii)), workers=-1)):
        if not truth_ids:
            continue
        truth_idx = np.asarray(truth_ids, dtype=np.int64)
        diff = truth_xy[truth_idx] - cand_xy[cand_i]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        keep = dist <= radii[cand_i]
        for truth_i, d in zip(truth_idx[keep], dist[keep]):
            proposals.append((float(d / max(radii[cand_i], EPS)), float(d), int(cand_i), int(truth_i)))

    proposals.sort(key=lambda row: (row[0], row[1]))
    used_cand: set[int] = set()
    used_truth: set[int] = set()
    selected: list[tuple[float, int, int]] = []
    for _norm_dist, dist, cand_i, truth_i in proposals:
        if cand_i in used_cand or truth_i in used_truth:
            continue
        used_cand.add(cand_i)
        used_truth.add(truth_i)
        selected.append((dist, cand_i, truth_i))

    cand_crowding = np.full(len(candidates), np.nan, dtype=np.float64)
    if len(candidates) > 1:
        cand_tree = cKDTree(cand_xy)
        dists, _idx = cand_tree.query(cand_xy, k=2, workers=-1)
        cand_crowding = np.asarray(dists[:, 1], dtype=np.float64)

    rows: list[dict[str, object]] = []
    for dist, cand_i, truth_i in selected:
        cand = candidates.iloc[cand_i]
        tru = truth.iloc[truth_i]
        cand_area = float(cand["b_maj"]) * float(cand["b_min"])
        truth_area = float(tru["b_maj"]) * float(tru["b_min"])
        pair_confidence, confidence_parts = pair_confidence_score(
            pair_dist=float(dist),
            pair_radius=float(radii[cand_i]),
            nearest_dist=float(nearest_dist_all[cand_i]),
            second_dist=float(second_dist_all[cand_i]),
            matched_is_nearest=bool(int(truth_i) == int(nearest_idx_all[cand_i])),
            candidate_crowding=float(cand_crowding[cand_i]),
            candidate_area=cand_area,
            truth_area=truth_area,
        )
        rows.append(
            {
                "det_id": int(cand["det_id"]),
                "truth_id": int(tru["truth_id"]),
                "candidate_index": int(cand_i),
                "truth_index": int(truth_i),
                "pair_dist_arcsec": float(dist),
                "pair_radius_arcsec": float(radii[cand_i]),
                "pair_norm_dist": float(dist / max(radii[cand_i], EPS)),
                "pair_confidence": pair_confidence,
                "nearest_truth_dist_arcsec": float(nearest_dist_all[cand_i]),
                "second_truth_dist_arcsec": float(second_dist_all[cand_i]),
                "matched_is_nearest_truth": bool(int(truth_i) == int(nearest_idx_all[cand_i])),
                **confidence_parts,
                "flux_base": float(cand["flux_base"]),
                "truth_flux": float(tru["flux"]),
                "delta_log_truth": float(np.log(max(float(tru["flux"]), EPS)) - np.log(max(float(cand["flux_base"]), EPS))),
                "objectness": float(cand["objectness"]),
                "probability": float(cand["probability"]),
                "prior_id": int(cand["prior_id"]),
                "b_maj": float(cand["b_maj"]),
                "b_min": float(cand["b_min"]),
                "pa": float(cand["pa"]),
                "truth_b_maj": float(tru["b_maj"]),
                "truth_b_min": float(tru["b_min"]),
                "truth_pa": float(tru["pa"]),
                "truth_size_id": int(tru["size_id"]),
                "truth_class_id": int(tru["class_id"]),
                "ra_core": float(cand["ra_core"]),
                "dec_core": float(cand["dec_core"]),
                "truth_ra_core": float(tru["ra_core"]),
                "truth_dec_core": float(tru["dec_core"]),
                "cx_pix": float(cand["cx_pix"]),
                "cy_pix": float(cand["cy_pix"]),
                "candidate_crowding_arcsec": float(cand_crowding[cand_i]),
            }
        )
    return pd.DataFrame(rows)


def pairing_audit(pairs: pd.DataFrame, *, n_candidates: int, n_truth: int, args: argparse.Namespace) -> pd.DataFrame:
    row: dict[str, float | int | str] = {
        "pairing": "geom_arcsec_greedy",
        "n_candidates_train": int(n_candidates),
        "n_truth_train": int(n_truth),
        "n_pairs": int(len(pairs)),
        "candidate_coverage": float(len(pairs) / n_candidates) if n_candidates else np.nan,
        "truth_coverage": float(len(pairs) / n_truth) if n_truth else np.nan,
        "min_radius_arcsec": float(args.min_radius_arcsec),
        "max_radius_arcsec": float(args.max_radius_arcsec),
        "radius_scale": float(args.radius_scale),
    }
    if len(pairs):
        for col in ("pair_dist_arcsec", "pair_radius_arcsec", "pair_norm_dist", "pair_confidence", "delta_log_truth"):
            row[col + "_p10"] = float(pairs[col].quantile(0.1))
            row[col + "_p50"] = float(pairs[col].quantile(0.5))
            row[col + "_p90"] = float(pairs[col].quantile(0.9))
        row["nearest_pair_share"] = float(pairs["matched_is_nearest_truth"].mean())
        row["truth_flux_median"] = float(pairs["truth_flux"].median())
        row["base_flux_median"] = float(pairs["flux_base"].median())
    return pd.DataFrame([row])


def build_geom_target_table_from_paths(
    catalog_path: Path,
    pred_obb_path: Path,
    truth_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog = read_catalog(catalog_path)
    catalog.loc[catalog["ra_core"] > 180.0, "ra_core"] -= 360.0
    catalog.loc[catalog["ra_cent"] > 180.0, "ra_cent"] -= 360.0
    pred = pd.read_csv(pred_obb_path)
    features = build_detector_feature_frame(catalog, pred)
    features["catalog_row"] = np.arange(len(features), dtype=np.int64)
    truth = read_truth_catalog(truth_path)

    cand_train = features.loc[training_mask(features)].copy().reset_index(drop=True)
    truth_train = truth.loc[training_mask(truth)].copy().reset_index(drop=True)
    pairs = greedy_geometric_pairing(
        cand_train,
        truth_train,
        min_radius_arcsec=float(args.min_radius_arcsec),
        max_radius_arcsec=float(args.max_radius_arcsec),
        radius_scale=float(args.radius_scale),
    )
    audit = pairing_audit(pairs, n_candidates=len(cand_train), n_truth=len(truth_train), args=args)
    if pairs.empty:
        raise RuntimeError("Geometry pairing returned no rows.")

    feature_cols = ["det_id"] + [col for col in FEATURE_COLUMNS if col in features.columns]
    target_cols = [
        "det_id",
        "truth_id",
        "truth_flux",
        "delta_log_truth",
        "pair_dist_arcsec",
        "pair_radius_arcsec",
        "pair_norm_dist",
        "pair_confidence",
        "matched_is_nearest_truth",
    ]
    table = features[feature_cols].merge(pairs[target_cols], on="det_id", how="inner", validate="one_to_one")
    return features, table, audit


def add_feature_expansion(features: pd.DataFrame, *, nb_prior: int = 9) -> pd.DataFrame:
    return add_stage9_feature_expansion(features, nb_prior=nb_prior)


def feature_columns(frame: pd.DataFrame, *, use_expansion: bool) -> list[str]:
    cols = [col for col in BASE_FEATURE_COLUMNS if col in frame.columns]
    if use_expansion:
        extras = [
            "log_flux_sq",
            "score_proxy_sq",
            "log_area_sq",
            "log_aspect_abs",
            "log_flux_x_score",
            "log_flux_x_objectness",
            "log_flux_x_probability",
            "log_flux_x_log_area",
            "score_x_log_area",
            "bright_x_log_flux",
            "bright_x_score",
        ]
        extras += [f"prior_{i}" for i in range(9)]
        extras += [f"prior_{i}_x_log_flux" for i in range(9)]
        cols += [col for col in extras if col in frame.columns]
    return cols


def split_by_det(det_id: np.ndarray, *, holdout_mod: int, holdout_value: int) -> tuple[np.ndarray, np.ndarray]:
    det_id = np.asarray(det_id, dtype=np.int64)
    valid = (det_id % int(holdout_mod)) == int(holdout_value)
    if valid.sum() == 0 or valid.sum() == len(valid):
        idx = np.arange(len(det_id), dtype=np.int64)
        valid = (idx % int(holdout_mod)) == int(holdout_value)
    return ~valid, valid


def robust_center_scale(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmedian(x_train, axis=0)
    q25 = np.nanpercentile(x_train, 25, axis=0)
    q75 = np.nanpercentile(x_train, 75, axis=0)
    scale = (q75 - q25) / 1.349
    fallback = np.nanstd(x_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    return mean.astype(np.float64), scale.astype(np.float64)


def weighted_ridge_fit(
    frame: pd.DataFrame,
    *,
    target: np.ndarray,
    weight: np.ndarray,
    columns: list[str],
    train_mask: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    x = frame[columns].to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    fit_mask = np.asarray(train_mask, dtype=bool) & np.isfinite(y)
    if int(fit_mask.sum()) < 20:
        raise ValueError("Need at least 20 train rows; got %d" % int(fit_mask.sum()))

    mean, scale = robust_center_scale(x[fit_mask])
    z = (x - mean) / scale
    z[~np.isfinite(z)] = 0.0
    y_fit = y[fit_mask]
    z_fit = z[fit_mask]
    w_fit = w[fit_mask]
    intercept = float(np.average(y_fit, weights=w_fit))
    y_center = y_fit - intercept

    sw = np.sqrt(w_fit / max(float(np.mean(w_fit)), EPS))[:, None]
    zw = z_fit * sw
    yw = y_center * sw[:, 0]
    xtx = zw.T @ zw
    penalty = np.eye(xtx.shape[0], dtype=np.float64) * float(alpha)
    coef = np.linalg.solve(xtx + penalty, zw.T @ yw)
    return {
        "model": "stage9_weighted_decoded_ridge",
        "columns": columns,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept,
        "alpha": float(alpha),
    }


def predict_delta(frame: pd.DataFrame, model: dict[str, object], *, clip: float) -> np.ndarray:
    columns = [str(col) for col in model["columns"]]
    x = frame[columns].to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.maximum(np.asarray(model["scale"], dtype=np.float64), EPS)
    coef = np.asarray(model["coef"], dtype=np.float64)
    delta = float(model["intercept"]) + ((x - mean) / scale) @ coef
    return np.clip(delta, -float(clip), float(clip))


def weighted_metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool) & np.isfinite(y) & np.isfinite(pred)
    if not np.any(mask):
        return {"n": 0}
    out = regression_metrics(y, pred, mask)
    w = np.asarray(weight[mask], dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    err = np.asarray(pred[mask] - y[mask], dtype=np.float64)
    out["weighted_mae"] = float(np.sum(w * np.abs(err)) / np.sum(w))
    out["weighted_bias"] = float(np.sum(w * err) / np.sum(w))
    out["n"] = int(mask.sum())
    return out
