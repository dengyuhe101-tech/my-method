#!/usr/bin/env python3
"""Shared paths and helpers for V4m Stage9 flux head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
V4M_DIR = SCRIPT_DIR.parent
SKAO_DIR = V4M_DIR.parent
REPO_DIR = SKAO_DIR.parents[1]
EPS = 1.0e-12
CATALOG_COLUMNS = [
    "det_id",
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
    "class",
]
FEATURE_COLUMNS = [
    "log_flux_base",
    "objectness",
    "probability",
    "score_proxy",
    "prior_id",
    "obb_w_pix",
    "obb_h_pix",
    "hbb_w_pix",
    "hbb_h_pix",
    "theta_le90_deg",
    "b_maj",
    "b_min",
    "pa",
    "log_obb_area_pix",
    "log_phys_area_arcsec",
    "log_aspect",
    "aspect_ratio_pred",
    "phys_bmaj_arcsec",
    "phys_bmin_arcsec",
    "obb_bmaj_arcsec",
    "obb_bmin_arcsec",
    "cx_pix",
    "cy_pix",
]
STAGE9_EXPANSION_COLUMNS = [
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


class NumpyRidge:
    """Small weighted ridge helper used by the detached Stage9/geom heads."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.columns: list[str] = []
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept: float = 0.0

    def fit(
        self,
        frame: pd.DataFrame,
        target: np.ndarray,
        columns: list[str],
        *,
        sample_weight: np.ndarray | None = None,
    ) -> "NumpyRidge":
        self.columns = list(columns)
        x = frame[self.columns].to_numpy(dtype=np.float64)
        x[~np.isfinite(x)] = 0.0
        y = np.asarray(target, dtype=np.float64)
        mask = np.isfinite(y)
        if int(mask.sum()) < 2:
            raise ValueError("Need at least 2 finite target rows for ridge fit.")
        if sample_weight is None:
            w = np.ones(mask.sum(), dtype=np.float64)
        else:
            w_all = np.asarray(sample_weight, dtype=np.float64)
            w = w_all[mask]
            w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)

        x_fit = x[mask]
        y_fit = y[mask]
        self.mean = np.nanmedian(x_fit, axis=0).astype(np.float64)
        q25 = np.nanpercentile(x_fit, 25, axis=0)
        q75 = np.nanpercentile(x_fit, 75, axis=0)
        scale = (q75 - q25) / 1.349
        fallback = np.nanstd(x_fit, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > EPS), scale, fallback)
        self.scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0).astype(np.float64)
        z_fit = (x_fit - self.mean) / self.scale
        z_fit[~np.isfinite(z_fit)] = 0.0
        self.intercept = float(np.average(y_fit, weights=w))
        sw = np.sqrt(w / max(float(np.mean(w)), EPS))[:, None]
        zw = z_fit * sw
        yw = (y_fit - self.intercept) * sw[:, 0]
        xtx = zw.T @ zw
        penalty = np.eye(xtx.shape[0], dtype=np.float64) * self.alpha
        self.coef = np.linalg.solve(xtx + penalty, zw.T @ yw)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("NumpyRidge must be fitted before predict().")
        x = frame[self.columns].to_numpy(dtype=np.float64)
        x[~np.isfinite(x)] = 0.0
        z = (x - self.mean) / np.maximum(self.scale, EPS)
        z[~np.isfinite(z)] = 0.0
        return self.intercept + z @ self.coef

    def to_dict(self) -> dict[str, object]:
        if self.mean is None or self.scale is None or self.coef is None:
            raise RuntimeError("NumpyRidge must be fitted before serialization.")
        return {
            "model": "numpy_ridge",
            "alpha": float(self.alpha),
            "columns": list(self.columns),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": float(self.intercept),
        }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    mask = np.asarray(mask, dtype=bool) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {"n": 0}
    y = np.asarray(y_true, dtype=np.float64)[mask]
    pred = np.asarray(y_pred, dtype=np.float64)[mask]
    err = pred - y
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / denom if denom > EPS else np.nan
    return {
        "n": int(mask.sum()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "r2": float(r2),
    }


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_DIR.resolve()))
    except ValueError:
        return str(path.resolve())


def read_catalog(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=CATALOG_COLUMNS, engine="c")


def write_catalog(path: Path, catalog: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack(
        [
            catalog["det_id"].to_numpy(dtype=np.int64),
            catalog["ra_core"].to_numpy(dtype=np.float64),
            catalog["dec_core"].to_numpy(dtype=np.float64),
            catalog["ra_cent"].to_numpy(dtype=np.float64),
            catalog["dec_cent"].to_numpy(dtype=np.float64),
            catalog["flux"].to_numpy(dtype=np.float64),
            catalog["core_frac"].to_numpy(dtype=np.float64),
            catalog["b_maj"].to_numpy(dtype=np.float64),
            catalog["b_min"].to_numpy(dtype=np.float64),
            catalog["pa"].to_numpy(dtype=np.float64),
            catalog["size_id"].to_numpy(dtype=np.int64),
            catalog["class"].to_numpy(dtype=np.int64),
        ]
    )
    np.savetxt(path, rows, fmt="%d %1.8f %2.8f %1.8f %2.8f %g %0.8f %f %f %f %d %d")


def safe_log(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.asarray(values, dtype=np.float64), EPS))


def build_detector_feature_frame(catalog: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    merged = catalog.merge(pred, how="left", on="det_id", suffixes=("", "_pred"))
    required = [
        "objectness",
        "probability",
        "prior_id",
        "cx_pix",
        "cy_pix",
        "obb_w_pix",
        "obb_h_pix",
        "theta_le90_deg",
        "flux_jy",
        "phys_bmaj_arcsec",
        "phys_bmin_arcsec",
        "phys_pa_deg",
    ]
    if merged[required].isna().any().any():
        missing = int(merged["objectness"].isna().sum())
        raise ValueError("Could not match %d catalog rows to pred_obb rows by det_id." % missing)

    flux_base = np.maximum(merged["flux"].to_numpy(dtype=np.float64), EPS)
    obb_w = np.maximum(merged["obb_w_pix"].to_numpy(dtype=np.float64), EPS)
    obb_h = np.maximum(merged["obb_h_pix"].to_numpy(dtype=np.float64), EPS)
    bmaj = np.maximum(merged["b_maj"].to_numpy(dtype=np.float64), EPS)
    bmin = np.maximum(merged["b_min"].to_numpy(dtype=np.float64), EPS)
    score_proxy = (
        merged["objectness"].to_numpy(dtype=np.float64)
        * merged["probability"].to_numpy(dtype=np.float64)
    )
    return pd.DataFrame(
        {
            "det_id": merged["det_id"].to_numpy(dtype=np.int64),
            "ra_core": merged["ra_core"].to_numpy(dtype=np.float64),
            "dec_core": merged["dec_core"].to_numpy(dtype=np.float64),
            "ra_cent": merged["ra_cent"].to_numpy(dtype=np.float64),
            "dec_cent": merged["dec_cent"].to_numpy(dtype=np.float64),
            "cx_pix": merged["cx_pix"].to_numpy(dtype=np.float64),
            "cy_pix": merged["cy_pix"].to_numpy(dtype=np.float64),
            "flux_base": flux_base,
            "objectness": merged["objectness"].to_numpy(dtype=np.float64),
            "probability": merged["probability"].to_numpy(dtype=np.float64),
            "prior_id": merged["prior_id"].to_numpy(dtype=np.int16),
            "obb_w_pix": obb_w,
            "obb_h_pix": obb_h,
            "theta_le90_deg": merged["theta_le90_deg"].to_numpy(dtype=np.float64),
            "hbb_w_pix": (
                merged["hbb_xmax"].to_numpy(dtype=np.float64) - merged["hbb_xmin"].to_numpy(dtype=np.float64)
            ),
            "hbb_h_pix": (
                merged["hbb_ymax"].to_numpy(dtype=np.float64) - merged["hbb_ymin"].to_numpy(dtype=np.float64)
            ),
            "b_maj": bmaj,
            "b_min": bmin,
            "pa": merged["pa"].to_numpy(dtype=np.float64),
            "size_id": merged["size_id"].to_numpy(dtype=np.int16),
            "class": merged["class"].to_numpy(dtype=np.int16),
            "phys_bmaj_arcsec": merged["phys_bmaj_arcsec"].to_numpy(dtype=np.float64),
            "phys_bmin_arcsec": merged["phys_bmin_arcsec"].to_numpy(dtype=np.float64),
            "phys_pa_deg": merged["phys_pa_deg"].to_numpy(dtype=np.float64),
            "obb_bmaj_arcsec": merged["obb_bmaj_arcsec"].to_numpy(dtype=np.float64),
            "obb_bmin_arcsec": merged["obb_bmin_arcsec"].to_numpy(dtype=np.float64),
            "aspect_ratio_pred": merged["aspect_ratio"].to_numpy(dtype=np.float64),
            "log_flux_base": safe_log(flux_base),
            "log_obb_area_pix": safe_log(obb_w * obb_h),
            "log_phys_area_arcsec": safe_log(bmaj * bmin),
            "log_aspect": safe_log(bmaj / bmin),
            "score_proxy": score_proxy,
            "bright_protected": (flux_base >= 1.0e-5).astype(np.int8),
        }
    )


def add_stage9_feature_expansion(feature_frame: pd.DataFrame, *, nb_prior: int = 9) -> pd.DataFrame:
    out = feature_frame.copy()
    log_flux = out["log_flux_base"].to_numpy(dtype=np.float64)
    score = out["score_proxy"].to_numpy(dtype=np.float64)
    obj = out["objectness"].to_numpy(dtype=np.float64)
    prob = out["probability"].to_numpy(dtype=np.float64)
    log_area = out["log_phys_area_arcsec"].to_numpy(dtype=np.float64)
    log_aspect = out["log_aspect"].to_numpy(dtype=np.float64)
    bright = out["bright_protected"].to_numpy(dtype=np.float64)

    out["log_flux_sq"] = log_flux * log_flux
    out["score_proxy_sq"] = score * score
    out["log_area_sq"] = log_area * log_area
    out["log_aspect_abs"] = np.abs(log_aspect)
    out["log_flux_x_score"] = log_flux * score
    out["log_flux_x_objectness"] = log_flux * obj
    out["log_flux_x_probability"] = log_flux * prob
    out["log_flux_x_log_area"] = log_flux * log_area
    out["score_x_log_area"] = score * log_area
    out["bright_x_log_flux"] = bright * log_flux
    out["bright_x_score"] = bright * score

    prior = out["prior_id"].to_numpy(dtype=np.int64)
    for prior_id in range(nb_prior):
        one_hot = (prior == prior_id).astype(np.float64)
        out[f"prior_{prior_id}"] = one_hot
        out[f"prior_{prior_id}_x_log_flux"] = one_hot * log_flux
    return out


def predict_delta_from_model(feature_frame: pd.DataFrame, model: dict[str, object]) -> np.ndarray:
    columns = [str(col) for col in model["columns"]]
    frame = feature_frame
    needs_stage9 = bool(model.get("feature_expansion")) or str(model.get("feature_builder", "")).startswith("stage9")
    if needs_stage9:
        frame = add_stage9_feature_expansion(frame)
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError("Feature frame is missing model columns: %s" % ", ".join(missing[:20]))
    x = frame[columns].to_numpy(dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.maximum(np.asarray(model["scale"], dtype=np.float64), EPS)
    coef = np.asarray(model["coef"], dtype=np.float64)
    delta = float(model["intercept"]) + ((x - mean) / scale) @ coef
    clip = float(model.get("clip", 0.5))
    return np.clip(delta, -clip, clip)
