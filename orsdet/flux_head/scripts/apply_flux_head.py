#!/usr/bin/env python3
"""Apply an ORSDet flux head frozen post-head package to a detector catalog."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from dat_package import extract_flux_head_dat


def default_raw_data_dir() -> Path:
    import os

    return Path(os.environ.get("SDC1_RAW_DATA_DIR", "/shared/main/dengyuhe/SDC1_YOLO_OBB/raw_data/560Mhz-1kh"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-dir", type=Path)
    model_group.add_argument(
        "--flux-head-dat",
        dest="dat_package",
        type=Path,
        help="Single-file ORSDet flux head .dat package.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--pred-obb", type=Path, default=None)
    parser.add_argument("--truth-catalog", type=Path, default=default_raw_data_dir() / "True_560MHz.txt")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    from flux_common import EPS, build_detector_feature_frame, predict_delta_from_model, read_catalog, require_file, write_catalog
    from score_utils import score_catalog, score_summary_dict, score_rows, write_apply_report

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dat_package is not None:
        with tempfile.TemporaryDirectory(prefix="flux_head_dat_") as tmp_name:
            _detector, model_dir = extract_flux_head_dat(
                args.dat_package.resolve(),
                Path(tmp_name),
                extract_detector=False,
                force=True,
            )
            config_path = model_dir / "model_config.json"
            require_file(config_path, "ORSDet flux head model_config.json")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_path = model_dir / str(config.get("fit_model", "fit_model.json"))
            require_file(model_path, "ORSDet flux head fit_model.json")
            model = json.loads(model_path.read_text(encoding="utf-8"))
            catalog_path = args.catalog.resolve() if args.catalog else Path(str(config["catalog_path"])).resolve()
            pred_obb_path = args.pred_obb.resolve() if args.pred_obb else Path(str(config["pred_obb_path"])).resolve()
    else:
        model_dir = args.model_dir.resolve()
        config_path = model_dir / "model_config.json"
        require_file(config_path, "ORSDet flux head model_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_path = model_dir / str(config.get("fit_model", "fit_model.json"))
        require_file(model_path, "ORSDet flux head fit_model.json")
        model = json.loads(model_path.read_text(encoding="utf-8"))
        catalog_path = args.catalog.resolve() if args.catalog else Path(str(config["catalog_path"])).resolve()
        pred_obb_path = args.pred_obb.resolve() if args.pred_obb else Path(str(config["pred_obb_path"])).resolve()

    require_file(catalog_path, "base catalog")
    require_file(pred_obb_path, "pred_obb")
    if args.score:
        require_file(args.truth_catalog.resolve(), "truth catalog")

    catalog = read_catalog(catalog_path)
    pred = pd.read_csv(pred_obb_path)
    features = build_detector_feature_frame(catalog, pred)
    delta = predict_delta_from_model(features, model)

    out_catalog = catalog.copy()
    flux_before = np.maximum(out_catalog["flux"].to_numpy(dtype=np.float64), EPS)
    flux_after = flux_before * np.exp(delta)
    out_catalog["flux"] = flux_after
    catalog_out = out_dir / "catalog_flux_head_post_head.txt"
    write_catalog(catalog_out, out_catalog)

    corrections = pd.DataFrame(
        {
            "det_id": out_catalog["det_id"].to_numpy(dtype=np.int64),
            "flux_before": flux_before,
            "flux_after": flux_after,
            "delta_pred": delta,
        }
    )
    corrections.to_csv(out_dir / "flux_corrections.csv", index=False, float_format="%.10g")

    before = after = None
    if args.score:
        before = score_summary_dict(score_catalog(catalog_path, args.truth_catalog.resolve(), train=False))
        after = score_summary_dict(score_catalog(catalog_out, args.truth_catalog.resolve(), train=False))
        score_rows(before, after).to_csv(out_dir / "score_delta.csv", index=False)
        pd.DataFrame([{"scheme": "base", **before}, {"scheme": "flux_head_single_dat_apply", **after}]).to_csv(
            out_dir / "score_summary.csv",
            index=False,
        )

    args.catalog = catalog_path
    args.pred_obb = pred_obb_path
    args.model = args.dat_package.resolve() if args.dat_package else model_dir.resolve()
    args.out_catalog = catalog_out
    write_apply_report(out_dir / "single_dat_apply_report_zh.md", args=args, before=before, after=after, corrections=corrections)
    print("Wrote", catalog_out)
    print("Wrote", out_dir / "flux_corrections.csv")


if __name__ == "__main__":
    main()
