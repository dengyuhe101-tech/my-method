"""Scoring, summary, and matched-error helpers for V5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np


SCORE_COLUMNS = ("position", "flux", "b_maj", "b_min", "pa", "core_frac", "class")


@dataclass
class ScoreResult:
    epoch: int
    score: float
    n_det: int
    n_match: int
    n_bad: int
    n_false: int
    score_det: float
    acc: float
    purity: float
    mean_scores: dict[str, float]
    catalog_path: Path
    score_path: Path


def le90_diff_deg(pred, truth):
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return (pred - truth + 90.0) % 180.0 - 90.0


def score_catalog(catalog_path: Path, truth_path: str, train: bool = False):
    from ska_sdc import Sdc1Scorer

    scorer = Sdc1Scorer.from_txt(str(catalog_path), truth_path, freq=560, sub_skiprows=0, truth_skiprows=0)
    scorer.run(mode=0, train=train, detail=True)
    return scorer


def result_from_scorer(epoch: int, scorer, catalog_path: Path, score_path: Path) -> ScoreResult:
    scores_df = scorer.score.scores_df
    mean_scores = {}
    for name in SCORE_COLUMNS:
        if scores_df is not None and name in scores_df.columns and len(scores_df) > 0:
            mean_scores[name] = float(scores_df[name].mean())
        else:
            mean_scores[name] = float("nan")

    n_det = int(scorer.score.n_det)
    n_match = int(scorer.score.n_match)
    purity = float(n_match / n_det) if n_det > 0 else 0.0
    return ScoreResult(
        epoch=int(epoch),
        score=float(scorer.score.value),
        n_det=n_det,
        n_match=n_match,
        n_bad=int(scorer.score.n_bad),
        n_false=int(scorer.score.n_false),
        score_det=float(scorer.score.score_det),
        acc=float(scorer.score.acc_pc),
        purity=purity,
        mean_scores=mean_scores,
        catalog_path=Path(catalog_path),
        score_path=Path(score_path),
    )


def write_score_epoch(path: Path, result: ScoreResult) -> None:
    lines = [
        "epoch: %d" % result.epoch,
        "score: %.10f" % result.score,
        "n_det: %d" % result.n_det,
        "n_match: %d" % result.n_match,
        "n_bad: %d" % result.n_bad,
        "n_false: %d" % result.n_false,
        "score_det: %.10f" % result.score_det,
        "acc: %.10f" % result.acc,
        "purity: %.10f" % result.purity,
        "catalog: %s" % result.catalog_path,
    ]
    for name in SCORE_COLUMNS:
        lines.append("mean_%s: %.10f" % (name, result.mean_scores.get(name, float("nan"))))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_score_history_csv(path: Path, results: list[ScoreResult]) -> None:
    header = [
        "epoch",
        "score",
        "n_det",
        "n_match",
        "n_bad",
        "n_false",
        "acc",
        "purity",
    ] + ["mean_" + col for col in SCORE_COLUMNS]
    rows = []
    for res in sorted(results, key=lambda item: item.epoch):
        rows.append(
            [
                res.epoch,
                res.score,
                res.n_det,
                res.n_match,
                res.n_bad,
                res.n_false,
                res.acc,
                res.purity,
            ]
            + [res.mean_scores.get(col, np.nan) for col in SCORE_COLUMNS]
        )
    np.savetxt(path, np.asarray(rows, dtype=np.float64), delimiter=",", header=",".join(header), comments="", fmt="%.10g")


def load_score_history_csv(path: Path, out_dir: Path) -> list[ScoreResult]:
    if not path.is_file():
        return []
    data = np.genfromtxt(path, names=True, delimiter=",", dtype=np.float64, encoding="utf-8")
    if data.size == 0:
        return []
    if data.shape == ():
        data = data.reshape(1)

    results = []
    for row in data:
        epoch = int(row["epoch"])
        catalog_path = out_dir / "catalogs" / ("catalog_sdc1_%04d.txt" % epoch)
        if not catalog_path.is_file():
            catalog_path = out_dir / ("catalog_sdc1_%04d.txt" % epoch)
        score_path = out_dir / "scores" / ("score_epoch_%04d.txt" % epoch)
        if not score_path.is_file():
            score_path = out_dir / ("score_epoch_%04d.txt" % epoch)
        mean_scores = {}
        for name in SCORE_COLUMNS:
            col = "mean_" + name
            mean_scores[name] = float(row[col]) if col in data.dtype.names else float("nan")
        results.append(
            ScoreResult(
                epoch=epoch,
                score=float(row["score"]),
                n_det=int(row["n_det"]),
                n_match=int(row["n_match"]),
                n_bad=int(row["n_bad"]),
                n_false=int(row["n_false"]),
                score_det=float("nan"),
                acc=float(row["acc"]),
                purity=float(row["purity"]),
                mean_scores=mean_scores,
                catalog_path=catalog_path,
                score_path=score_path,
            )
        )
    return results


def write_score_summary(path: Path, results: list[ScoreResult]) -> ScoreResult:
    if not results:
        raise ValueError("No score results to summarize.")
    ordered = sorted(results, key=lambda item: item.epoch)
    best = max(ordered, key=lambda item: item.score)

    lines = []
    lines.append("V5 Score Summary")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Best epoch")
    lines.append("-" * 80)
    lines.append("epoch      : %d" % best.epoch)
    lines.append("score      : %.10f" % best.score)
    lines.append("n_det      : %d" % best.n_det)
    lines.append("n_match    : %d" % best.n_match)
    lines.append("n_bad      : %d" % best.n_bad)
    lines.append("n_false    : %d" % best.n_false)
    lines.append("acc        : %.10f" % best.acc)
    lines.append("purity     : %.10f" % best.purity)
    lines.append("catalog    : %s" % best.catalog_path)
    lines.append("")
    lines.append("Mean per-attribute scores for best epoch")
    lines.append("-" * 80)
    for name in SCORE_COLUMNS:
        lines.append("%-10s : %.10f" % (name, best.mean_scores.get(name, float("nan"))))
    lines.append("")
    lines.append("All scored epochs")
    lines.append("-" * 80)
    lines.append("%8s  %16s  %12s" % ("epoch", "score", "n_match"))
    lines.append("%8s  %16s  %12s" % ("-" * 8, "-" * 16, "-" * 12))
    for res in ordered:
        marker = " *" if res.epoch == best.epoch else "  "
        lines.append("%8d  %16.6f  %12d%s" % (res.epoch, res.score, res.n_match, marker))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best


def update_best_aliases(out_dir: Path, best: ScoreResult) -> None:
    catalog_alias = out_dir / "best_catalog_sdc1.txt"
    catalog_alias.unlink(missing_ok=True)
    try:
        target = best.catalog_path.relative_to(out_dir)
    except ValueError:
        target = best.catalog_path
    try:
        catalog_alias.symlink_to(target)
    except OSError:
        shutil.copy2(best.catalog_path, catalog_alias)


def make_matched_errors(scorer) -> np.ndarray:
    match_df = scorer.score.match_df
    scores_df = scorer.score.scores_df
    if match_df is None or len(match_df) == 0:
        return np.zeros((0,), dtype=[])

    score_by_id = {}
    if scores_df is not None and len(scores_df) > 0:
        for _, row in scores_df.iterrows():
            score_by_id[int(row["id"])] = row

    dtype = [
        ("det_id", "i8"),
        ("truth_id", "i8"),
        ("ra_deg", "f8"),
        ("dec_deg", "f8"),
        ("truth_ra_deg", "f8"),
        ("truth_dec_deg", "f8"),
        ("flux", "f8"),
        ("truth_flux", "f8"),
        ("flux_rel_err", "f8"),
        ("bmaj", "f8"),
        ("truth_bmaj", "f8"),
        ("bmaj_rel_err", "f8"),
        ("bmin", "f8"),
        ("truth_bmin", "f8"),
        ("bmin_rel_err", "f8"),
        ("pa", "f8"),
        ("truth_pa", "f8"),
        ("pa_err_deg", "f8"),
        ("pa_abs_err_deg", "f8"),
        ("truth_aspect", "f8"),
        ("truth_size_arcsec", "f8"),
        ("truth_flux_log10", "f8"),
        ("multi_d_err", "f8"),
    ] + [("score_" + col, "f8") for col in SCORE_COLUMNS]
    out = np.zeros((len(match_df),), dtype=dtype)

    for i, (_, row) in enumerate(match_df.iterrows()):
        det_id = int(row["id"])
        flux_t_raw = float(row["flux_t"])
        bmaj_t_raw = float(row["b_maj_t"])
        bmin_t_raw = float(row["b_min_t"])
        flux_t = max(flux_t_raw, 1.0e-12)
        bmaj_t = max(bmaj_t_raw, 1.0e-3)
        bmin_t = max(bmin_t_raw, 1.0e-3)
        pa_err = float(le90_diff_deg(row["pa"], row["pa_t"]))
        out["det_id"][i] = det_id
        out["truth_id"][i] = int(row["id_t"])
        out["ra_deg"][i] = float(row["ra_cent"])
        out["dec_deg"][i] = float(row["dec_cent"])
        out["truth_ra_deg"][i] = float(row["ra_cent_t"])
        out["truth_dec_deg"][i] = float(row["dec_cent_t"])
        out["flux"][i] = float(row["flux"])
        out["truth_flux"][i] = flux_t_raw
        out["flux_rel_err"][i] = (float(row["flux"]) - flux_t) / flux_t
        out["bmaj"][i] = float(row["b_maj"])
        out["truth_bmaj"][i] = bmaj_t_raw
        out["bmaj_rel_err"][i] = (float(row["b_maj"]) - bmaj_t) / bmaj_t
        out["bmin"][i] = float(row["b_min"])
        out["truth_bmin"][i] = bmin_t_raw
        out["bmin_rel_err"][i] = (float(row["b_min"]) - bmin_t) / bmin_t
        out["pa"][i] = float(row["pa"])
        out["truth_pa"][i] = float(row["pa_t"])
        out["pa_err_deg"][i] = pa_err
        out["pa_abs_err_deg"][i] = abs(pa_err)
        out["truth_aspect"][i] = bmaj_t / bmin_t
        out["truth_size_arcsec"][i] = np.sqrt(bmaj_t * bmin_t)
        out["truth_flux_log10"][i] = np.log10(flux_t)
        out["multi_d_err"][i] = float(row["multi_d_err"])
        score_row = score_by_id.get(det_id)
        for col in SCORE_COLUMNS:
            out["score_" + col][i] = float(score_row[col]) if score_row is not None and col in score_row else np.nan
    return out


def save_structured_csv(path: Path, data: np.ndarray) -> None:
    if data.dtype.names is None:
        raise ValueError("Expected a structured array.")
    header = ",".join(data.dtype.names)
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in data:
            values = []
            for name in data.dtype.names:
                value = row[name]
                if np.issubdtype(data.dtype[name], np.floating):
                    values.append("%.10g" % float(value))
                elif np.issubdtype(data.dtype[name], np.integer):
                    values.append("%d" % int(value))
                else:
                    values.append(str(value))
            f.write(",".join(values) + "\n")


def _quantile_labels(values, names):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    labels = np.full(values.shape, "", dtype=object)
    if np.sum(finite) == 0:
        return labels
    q1, q2 = np.nanquantile(values[finite], [1.0 / 3.0, 2.0 / 3.0])
    labels[values <= q1] = names[0]
    labels[(values > q1) & (values <= q2)] = names[1]
    labels[values > q2] = names[2]
    return labels


def grouped_error_table(errors: np.ndarray) -> np.ndarray:
    dtype = [
        ("axis", "U16"),
        ("group", "U32"),
        ("n", "i8"),
        ("pa_mae_deg", "f8"),
        ("pa_p50_abs_deg", "f8"),
        ("pa_p90_abs_deg", "f8"),
        ("flux_mape", "f8"),
        ("bmaj_mape", "f8"),
        ("bmin_mape", "f8"),
        ("position_score_mean", "f8"),
        ("pa_score_mean", "f8"),
    ]
    if errors.size == 0:
        return np.zeros((0,), dtype=dtype)

    groupings = {
        "size": _quantile_labels(errors["truth_size_arcsec"], ("small", "medium", "large")),
        "flux": _quantile_labels(errors["truth_flux"], ("faint", "medium", "bright")),
        "aspect": np.where(
            errors["truth_aspect"] < 1.5,
            "near_square",
            np.where(errors["truth_aspect"] < 3.0, "elongated", "very_elongated"),
        ),
    }

    rows = []
    for axis, labels in groupings.items():
        for label in sorted(set(labels.tolist())):
            if not label:
                continue
            mask = labels == label
            if not np.any(mask):
                continue
            abs_pa = errors["pa_abs_err_deg"][mask]
            rows.append(
                (
                    axis,
                    label,
                    int(np.sum(mask)),
                    float(np.nanmean(abs_pa)),
                    float(np.nanpercentile(abs_pa, 50)),
                    float(np.nanpercentile(abs_pa, 90)),
                    float(np.nanmean(np.abs(errors["flux_rel_err"][mask]))),
                    float(np.nanmean(np.abs(errors["bmaj_rel_err"][mask]))),
                    float(np.nanmean(np.abs(errors["bmin_rel_err"][mask]))),
                    float(np.nanmean(errors["score_position"][mask])),
                    float(np.nanmean(errors["score_pa"][mask])),
                )
            )
    out = np.zeros((len(rows),), dtype=dtype)
    for i, row in enumerate(rows):
        out[i] = row
    return out
