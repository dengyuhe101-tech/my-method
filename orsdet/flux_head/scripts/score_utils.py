#!/usr/bin/env python3
"""Scoring helpers shared by the ORSDet flux head tools."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from ska_sdc import Sdc1Scorer

from flux_common import rel_or_abs


def default_raw_data_dir() -> Path:
    return Path(os.environ.get("SDC1_RAW_DATA_DIR", "/shared/main/dengyuhe/SDC1_YOLO_OBB/raw_data/560Mhz-1kh"))


def score_catalog(catalog_path: Path, truth_path: Path, *, train: bool):
    scorer = Sdc1Scorer.from_txt(str(catalog_path), str(truth_path), freq=560, sub_skiprows=0, truth_skiprows=0)
    scorer.run(mode=0, train=train, detail=True)
    return scorer


def score_summary_dict(scorer) -> dict[str, float | int]:
    score = scorer.score
    scores_df = score.scores_df
    row: dict[str, float | int] = {
        "score": float(score.value),
        "n_det": int(score.n_det),
        "n_match": int(score.n_match),
        "n_bad": int(score.n_bad),
        "n_false": int(score.n_false),
        "purity": float(score.n_match / score.n_det) if int(score.n_det) else np.nan,
        "acc": float(score.acc_pc),
    }
    if scores_df is not None and len(scores_df):
        for col in ("position", "flux", "b_maj", "b_min", "pa", "core_frac", "class"):
            if col in scores_df:
                row[col] = float(scores_df[col].mean())
    return row


def score_rows(before: dict[str, float | int], after: dict[str, float | int]) -> pd.DataFrame:
    rows = []
    for key in ("score", "n_det", "n_match", "n_bad", "n_false", "purity", "acc", "flux", "b_maj", "b_min", "pa"):
        if key not in before or key not in after:
            continue
        rows.append({"metric": key, "before": before[key], "after": after[key], "delta": float(after[key]) - float(before[key])})
    return pd.DataFrame(rows)


def write_apply_report(
    path: Path,
    *,
    args,
    before: dict[str, float | int] | None,
    after: dict[str, float | int] | None,
    corrections: pd.DataFrame,
) -> None:
    lines = [
        "# ORSDet flux head 应用报告",
        "",
        "本入口读取 ORSDet flux head 模型包或单 `.dat` trailer，只替换 catalog `flux`。",
        "",
        "## 输入输出",
        "",
        f"- catalog: `{rel_or_abs(args.catalog)}`",
        f"- pred_obb: `{rel_or_abs(args.pred_obb)}`",
        f"- model: `{rel_or_abs(args.model)}`",
        f"- output catalog: `{rel_or_abs(args.out_catalog)}`",
        f"- corrections: `{rel_or_abs(args.out_dir / 'flux_corrections.csv')}`",
        "",
        "## 修正分布",
        "",
        f"- rows: `{len(corrections)}`",
        f"- delta mean: `{float(corrections['delta_pred'].mean()):.10f}`",
        f"- delta p10/p50/p90: `{float(corrections['delta_pred'].quantile(0.1)):.10f}` / `{float(corrections['delta_pred'].quantile(0.5)):.10f}` / `{float(corrections['delta_pred'].quantile(0.9)):.10f}`",
        "",
    ]
    if before is not None and after is not None:
        delta = score_rows(before, after)
        lines.extend(["## 评分", "", "| metric | before | after | delta |", "|---|---:|---:|---:|"])
        for _, row in delta.iterrows():
            lines.append("| %s | %.10g | %.10g | %.10g |" % (row["metric"], row["before"], row["after"], row["delta"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
