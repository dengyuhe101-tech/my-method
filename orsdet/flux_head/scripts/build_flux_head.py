#!/usr/bin/env python3
"""Train/package the V4m Stage9 head for one checkpoint.

This is the formal checkpoint hook for V4m: detector bytes stay unchanged,
Stage9 is trained from training-region post-NMS candidates, and both parts are
packed into one .dat trailer artifact.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from stage9_core import (
    add_feature_expansion,
    build_geom_target_table_from_paths,
    feature_columns,
    predict_delta,
    split_by_det,
    weighted_metrics,
    weighted_ridge_fit,
)
from flux_common import EPS, read_catalog, rel_or_abs, require_file, write_catalog, write_json
from flux_common import build_detector_feature_frame
from dat_package import pack_v4m_dat, read_v4m_dat_info
from score_utils import default_raw_data_dir, score_catalog, score_rows, score_summary_dict


def apply_bright_gate(
    delta_log_truth: np.ndarray,
    flux_base: np.ndarray,
    *,
    bright_flux_min: float,
    soft_delta: float,
    hard_delta: float,
    soft_weight: float,
    hard_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(delta_log_truth, dtype=np.float64)
    base = np.asarray(flux_base, dtype=np.float64)
    weights = np.ones_like(delta)
    bright = base >= float(bright_flux_min)
    weights[bright & (delta < -abs(float(soft_delta)))] = float(soft_weight)
    weights[bright & (delta < -abs(float(hard_delta)))] = float(hard_weight)
    return delta * weights, weights


def build_targets(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features, table, audit = build_geom_target_table_from_paths(
        args.train_catalog.resolve(),
        args.train_pred_obb.resolve(),
        args.truth_catalog.resolve(),
        args,
    )
    targets = table.copy()
    if float(args.pair_confidence_min_fit) > 0.0 and "pair_confidence" in targets:
        filtered = targets[targets["pair_confidence"].to_numpy(dtype=np.float64) >= float(args.pair_confidence_min_fit)]
        if len(filtered) >= 20:
            targets = filtered.copy()
    if len(targets) < 20:
        raise RuntimeError("Too few Stage9 target rows after filtering: %d" % len(targets))

    if "flux_base" not in targets.columns:
        if "log_flux_base" in targets.columns:
            targets["flux_base"] = np.exp(targets["log_flux_base"].to_numpy(dtype=np.float64))
        elif {"det_id", "flux_base"}.issubset(features.columns):
            targets = targets.merge(
                features[["det_id", "flux_base"]],
                on="det_id",
                how="left",
                validate="one_to_one",
            )
        else:
            raise KeyError("Stage9 targets need flux_base or log_flux_base for bright-gate weighting.")
    targets["flux_base"] = np.maximum(
        np.where(
            np.isfinite(targets["flux_base"].to_numpy(dtype=np.float64)),
            targets["flux_base"].to_numpy(dtype=np.float64),
            EPS,
        ),
        EPS,
    )
    flux_base = np.maximum(targets["flux_base"].to_numpy(dtype=np.float64), EPS)
    _gated_delta, gate_weight = apply_bright_gate(
        targets["delta_log_truth"].to_numpy(dtype=np.float64),
        flux_base,
        bright_flux_min=float(args.bright_flux_min),
        soft_delta=float(args.bright_soft_delta),
        hard_delta=float(args.bright_hard_delta),
        soft_weight=float(args.bright_soft_weight),
        hard_weight=float(args.bright_hard_weight),
    )
    targets["stage7_gate_weight"] = gate_weight
    targets["stage9_target_policy"] = "post_nms_candidate_geom_truth_with_bright_weight"
    targets["candidate_target_level"] = "post_nms_detection"
    targets["source_level_target"] = False
    return features, targets.reset_index(drop=True), audit


def write_model_package(
    *,
    package_dir: Path,
    model_path: Path,
    args: argparse.Namespace,
    force: bool,
) -> Path:
    if package_dir.exists():
        if not force:
            raise FileExistsError(package_dir)
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, package_dir / "fit_model.json")
    config = {
        "format": "V4m Stage9 decoded-candidate checkpoint package",
        "display": args.display,
        "epoch": int(args.epoch),
        "detector_path": str(args.detector.resolve()),
        "training_catalog_path": str(args.train_catalog.resolve()),
        "training_pred_obb_path": str(args.train_pred_obb.resolve()),
        "catalog_path": str(args.apply_catalog.resolve()),
        "pred_obb_path": str(args.apply_pred_obb.resolve()),
        "fit_model": "fit_model.json",
        "apply_entry": "scripts/apply_flux_head.py",
        "notes": (
            "Detector bytes are unchanged. Stage9 is a detached decoded-candidate "
            "flux head trained from training-region geometric truth pairs."
        ),
    }
    (package_dir / "model_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {args.display} V4m Stage9 单 `.dat` 包",
        "",
        "该包由 checkpoint hook 生成：detector `.dat` 保持原始 bytes，Stage9 decoded-candidate head 作为 trailer payload 追加。",
        "",
        f"- detector: `{rel_or_abs(args.detector)}`",
        f"- training catalog: `{rel_or_abs(args.train_catalog)}`",
        f"- training pred_obb: `{rel_or_abs(args.train_pred_obb)}`",
        f"- apply catalog: `{rel_or_abs(args.apply_catalog)}`",
        f"- apply pred_obb: `{rel_or_abs(args.apply_pred_obb)}`",
        f"- fit model: `{rel_or_abs(model_path)}`",
        "",
        "注意：Stage9 是 checkpoint-integrated 二阶段测光头，不回传 detector。",
        "",
    ]
    (package_dir / "MODEL_CARD_zh.md").write_text("\n".join(lines), encoding="utf-8")
    return package_dir


def write_report(path: Path, row: dict[str, object], metrics: dict[str, dict[str, float]], audit: pd.DataFrame) -> None:
    audit_row = audit.iloc[0] if len(audit) else pd.Series(dtype=object)
    lines = [
        f"# {row['display']} V4m Stage9 checkpoint hook 结果",
        "",
        "本报告验证单 checkpoint 的正式 V4m Stage9 训练/打包流程。",
        "训练输入来自 training-region post-NMS catalog/pred_obb；应用输入来自 full/eval catalog/pred_obb。",
        "detector 权重不变，Stage9 head 写入单 `.dat` trailer。",
        "",
        "## 输入输出",
        "",
        f"- detector: `{row['detector']}`",
        f"- train catalog: `{row['train_catalog']}`",
        f"- train pred_obb: `{row['train_pred_obb']}`",
        f"- apply catalog: `{row['apply_catalog']}`",
        f"- apply pred_obb: `{row['apply_pred_obb']}`",
        f"- output dat: `{row['dat_path']}`",
        f"- output catalog: `{row['out_catalog']}`",
        "",
        "## 配对与拟合",
        "",
        "| item | value |",
        "|---|---:|",
        f"| target rows | `{row['n_pair_rows']}` |",
        f"| fit rows | `{row['n_fit_rows']}` |",
        f"| feature columns | `{row['n_columns']}` |",
        f"| valid R2 | `{metrics['valid'].get('r2', np.nan):.6f}` |",
        f"| valid weighted MAE | `{metrics['valid'].get('weighted_mae', np.nan):.6f}` |",
        f"| pair confidence p50 | `{float(audit_row.get('pair_confidence_p50', np.nan)):.6f}` |",
        "",
    ]
    if row.get("scored"):
        lines.extend(
            [
                "## 评分",
                "",
                "| scheme | score | n_det | n_match | n_false | purity | flux | b_maj | b_min | pa |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                "| base | %.4f | %d | %d | %d | %.6f | %.6f | %.6f | %.6f | %.6f |"
                % (
                    row["base_score"],
                    row["base_n_det"],
                    row["base_n_match"],
                    row["base_n_false"],
                    row["base_purity"],
                    row["base_flux"],
                    row["base_b_maj"],
                    row["base_b_min"],
                    row["base_pa"],
                ),
                "| stage9 | %.4f | %d | %d | %d | %.6f | %.6f | %.6f | %.6f | %.6f |"
                % (
                    row["stage9_score"],
                    row["stage9_n_det"],
                    row["stage9_n_match"],
                    row["stage9_n_false"],
                    row["stage9_purity"],
                    row["stage9_flux"],
                    row["stage9_b_maj"],
                    row["stage9_b_min"],
                    row["stage9_pa"],
                ),
                "",
                "## 判断",
                "",
                f"- 相对 base：`{row['delta_score']:+.4f}`。",
                f"- `n_det` 变化：`{row['delta_n_det']}`。",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_hook(args: argparse.Namespace) -> dict[str, object]:
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = out_dir / "stage9_decoded_truth_head"
    apply_dir = out_dir / "stage9_apply"
    package_dir = out_dir / "model_package_stage9"
    fit_dir.mkdir(parents=True, exist_ok=True)
    apply_dir.mkdir(parents=True, exist_ok=True)

    features, targets, audit = build_targets(args)
    targets.to_csv(fit_dir / "stage9_candidate_targets.csv", index=False, float_format="%.10g")
    audit.to_csv(fit_dir / "stage9_pairing_audit.csv", index=False)

    features_expanded = add_feature_expansion(features)
    fit = features_expanded.merge(
        targets[["det_id", "truth_id", "delta_log_truth", "pair_confidence", "stage7_gate_weight"]],
        on="det_id",
        how="inner",
        validate="one_to_one",
    )
    columns = feature_columns(fit, use_expansion=True)
    y = fit["delta_log_truth"].to_numpy(dtype=np.float64)
    sample_weight = (
        fit["pair_confidence"].to_numpy(dtype=np.float64)
        * fit["stage7_gate_weight"].to_numpy(dtype=np.float64)
    )
    sample_weight = np.where(np.isfinite(sample_weight) & (sample_weight > 0), sample_weight, 1.0)
    train_mask, valid_mask = split_by_det(
        fit["det_id"].to_numpy(dtype=np.int64),
        holdout_mod=int(args.holdout_mod),
        holdout_value=int(args.holdout_value),
    )
    model = weighted_ridge_fit(
        fit,
        target=y,
        weight=sample_weight,
        columns=columns,
        train_mask=train_mask,
        alpha=float(args.alpha),
    )
    model.update(
        {
            "target_col": "delta_log_truth",
            "target_source": "train_region_geometric_truth_pairs",
            "feature_builder": "stage9_decoded_v1",
            "feature_expansion": True,
            "training_stage": "V4m Stage9 checkpoint hook",
            "integration": "checkpoint_integrated_two_stage_head",
            "detector_path": str(args.detector.resolve()),
            "training_catalog_path": str(args.train_catalog.resolve()),
            "training_pred_obb_path": str(args.train_pred_obb.resolve()),
            "apply_catalog_path": str(args.apply_catalog.resolve()),
            "apply_pred_obb_path": str(args.apply_pred_obb.resolve()),
            "clip": float(args.clip),
            "holdout_mod": int(args.holdout_mod),
            "holdout_value": int(args.holdout_value),
            "pair_confidence_min_fit": float(args.pair_confidence_min_fit),
            "bright_flux_min": float(args.bright_flux_min),
            "bright_soft_delta": float(args.bright_soft_delta),
            "bright_hard_delta": float(args.bright_hard_delta),
            "bright_soft_weight": float(args.bright_soft_weight),
            "bright_hard_weight": float(args.bright_hard_weight),
            "n_pair_rows": int(len(targets)),
            "n_fit_rows": int(len(fit)),
            "n_train": int(train_mask.sum()),
            "n_valid": int(valid_mask.sum()),
        }
    )
    pred_fit = predict_delta(fit, model, clip=float(args.clip))
    metrics = {
        "train": weighted_metrics(y, pred_fit, train_mask, sample_weight),
        "valid": weighted_metrics(y, pred_fit, valid_mask, sample_weight),
        "all": weighted_metrics(y, pred_fit, np.ones(len(fit), dtype=bool), sample_weight),
    }
    write_json(fit_dir / "fit_model.json", model)
    write_json(fit_dir / "fit_metrics.json", metrics)
    fit.assign(delta_pred=pred_fit, sample_weight=sample_weight).to_csv(
        fit_dir / "stage9_fit_rows.csv",
        index=False,
        float_format="%.10g",
    )

    apply_catalog = read_catalog(args.apply_catalog.resolve())
    apply_pred = pd.read_csv(args.apply_pred_obb.resolve())
    apply_features = add_feature_expansion(build_detector_feature_frame(apply_catalog, apply_pred))
    delta = predict_delta(apply_features, model, clip=float(args.clip))
    out_catalog = apply_catalog.copy()
    flux_before = np.maximum(out_catalog["flux"].to_numpy(dtype=np.float64), EPS)
    out_catalog["flux"] = flux_before * np.exp(delta)
    out_catalog_path = apply_dir / "catalog_v4m_post_head.txt"
    write_catalog(out_catalog_path, out_catalog)
    pd.DataFrame(
        {
            "det_id": apply_features["det_id"].to_numpy(dtype=np.int64),
            "flux_before": flux_before,
            "flux_after": out_catalog["flux"].to_numpy(dtype=np.float64),
            "delta_pred": delta,
        }
    ).to_csv(apply_dir / "flux_corrections.csv", index=False, float_format="%.10g")

    package = write_model_package(package_dir=package_dir, model_path=fit_dir / "fit_model.json", args=args, force=args.force)
    dat_path = args.out_dat.resolve() if args.out_dat else out_dir / ("v4m_stage9_net0_s%04d.dat" % int(args.epoch))
    info = pack_v4m_dat(args.detector.resolve(), package, dat_path, force=bool(args.force))
    if args.verify_dat:
        read_v4m_dat_info(dat_path, verify=True)

    row: dict[str, object] = {
        "display": args.display,
        "epoch": int(args.epoch),
        "detector": rel_or_abs(args.detector),
        "train_catalog": rel_or_abs(args.train_catalog),
        "train_pred_obb": rel_or_abs(args.train_pred_obb),
        "apply_catalog": rel_or_abs(args.apply_catalog),
        "apply_pred_obb": rel_or_abs(args.apply_pred_obb),
        "out_catalog": rel_or_abs(out_catalog_path),
        "dat_path": rel_or_abs(dat_path),
        "dat_path_abs": str(dat_path.resolve()),
        "detector_len": int(info.detector_len),
        "payload_len": int(info.payload_len),
        "n_pair_rows": int(len(targets)),
        "n_fit_rows": int(len(fit)),
        "n_columns": int(len(columns)),
        "scored": bool(args.score),
    }
    if args.score:
        before = score_summary_dict(score_catalog(args.apply_catalog.resolve(), args.truth_catalog.resolve(), train=False))
        after = score_summary_dict(score_catalog(out_catalog_path, args.truth_catalog.resolve(), train=False))
        score_rows(before, after).to_csv(apply_dir / "score_delta.csv", index=False)
        pd.DataFrame([{"scheme": "base", **before}, {"scheme": "v4m_stage9", **after}]).to_csv(
            apply_dir / "score_summary.csv",
            index=False,
        )
        for prefix, score in (("base", before), ("stage9", after)):
            for key, value in score.items():
                row[f"{prefix}_{key}"] = value
        row["delta_score"] = float(after["score"] - before["score"])
        row["delta_n_det"] = int(after["n_det"] - before["n_det"])

    write_json(out_dir / "stage9_checkpoint_hook_report.json", row)
    write_report(out_dir / "stage9_checkpoint_hook_report_zh.md", row, metrics, audit)
    print("Wrote", dat_path)
    print("Wrote", out_catalog_path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--display", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--train-catalog", type=Path, required=True)
    parser.add_argument("--train-pred-obb", type=Path, required=True)
    parser.add_argument("--apply-catalog", type=Path, required=True)
    parser.add_argument("--apply-pred-obb", type=Path, required=True)
    parser.add_argument("--truth-catalog", type=Path, default=default_raw_data_dir() / "True_560MHz.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-dat", type=Path)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--holdout-mod", type=int, default=5)
    parser.add_argument("--holdout-value", type=int, default=0)
    parser.add_argument("--min-radius-arcsec", type=float, default=0.9)
    parser.add_argument("--max-radius-arcsec", type=float, default=2.5)
    parser.add_argument("--radius-scale", type=float, default=0.75)
    parser.add_argument("--pair-confidence-min-fit", type=float, default=0.7)
    parser.add_argument("--bright-flux-min", type=float, default=1.0e-5)
    parser.add_argument("--bright-soft-delta", type=float, default=0.10)
    parser.add_argument("--bright-hard-delta", type=float, default=0.35)
    parser.add_argument("--bright-soft-weight", type=float, default=0.75)
    parser.add_argument("--bright-hard-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", help="Accepted for eval hook compatibility; Stage9 ridge runs on CPU.")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--verify-dat", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for path, label in (
        (args.detector, "detector checkpoint"),
        (args.train_catalog, "training catalog"),
        (args.train_pred_obb, "training pred_obb"),
        (args.apply_catalog, "apply catalog"),
        (args.apply_pred_obb, "apply pred_obb"),
        (args.truth_catalog, "truth catalog"),
    ):
        require_file(path.resolve(), label)
    run_hook(args)


if __name__ == "__main__":
    main()
