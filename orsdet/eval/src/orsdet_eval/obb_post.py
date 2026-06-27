"""V4a OBB post-processing summary helpers for V5."""

from __future__ import annotations

from pathlib import Path
import shutil

from .score import (
    ScoreResult,
    grouped_error_table,
    make_matched_errors,
    result_from_scorer,
    save_structured_csv,
    score_catalog,
    update_best_aliases,
    write_score_epoch,
    write_score_history_csv,
    write_score_summary,
)


CATALOG_DIR = "catalogs"
PRED_OBB_DIR = "pred_obb"
SCORE_DIR = "scores"
MATCHED_ERROR_DIR = "matched_errors"
ERROR_GROUP_DIR = "error_by_group"


def _safe_alias(src: Path, alias: Path) -> None:
    alias.unlink(missing_ok=True)
    try:
        target = src.relative_to(alias.parent)
    except ValueError:
        target = src
    try:
        alias.symlink_to(target)
    except OSError:
        shutil.copy2(src, alias)


def ensure_obb_layout(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CATALOG_DIR).mkdir(exist_ok=True)
    (out_dir / PRED_OBB_DIR).mkdir(exist_ok=True)
    (out_dir / SCORE_DIR).mkdir(exist_ok=True)
    (out_dir / MATCHED_ERROR_DIR).mkdir(exist_ok=True)
    (out_dir / ERROR_GROUP_DIR).mkdir(exist_ok=True)


def catalog_path(out_dir: Path, epoch: int) -> Path:
    return out_dir / CATALOG_DIR / ("catalog_sdc1_%04d.txt" % epoch)


def pred_obb_path(out_dir: Path, epoch: int) -> Path:
    return out_dir / PRED_OBB_DIR / ("pred_obb_%04d.csv" % epoch)


def score_path(out_dir: Path, epoch: int) -> Path:
    return out_dir / SCORE_DIR / ("score_epoch_%04d.txt" % epoch)


def matched_errors_path(out_dir: Path, epoch: int) -> Path:
    return out_dir / MATCHED_ERROR_DIR / ("matched_errors_epoch_%04d.csv" % epoch)


def error_group_path(out_dir: Path, epoch: int) -> Path:
    return out_dir / ERROR_GROUP_DIR / ("error_by_group_epoch_%04d.csv" % epoch)


def matched_errors_latest_path(out_dir: Path) -> Path:
    return out_dir / MATCHED_ERROR_DIR / "matched_errors.csv"


def error_group_latest_path(out_dir: Path) -> Path:
    return out_dir / ERROR_GROUP_DIR / "error_by_group.csv"


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def find_catalog_path(out_dir: Path, epoch: int) -> Path:
    return _first_existing(catalog_path(out_dir, epoch), out_dir / ("catalog_sdc1_%04d.txt" % epoch))


def find_pred_obb_path(out_dir: Path, epoch: int) -> Path:
    return _first_existing(pred_obb_path(out_dir, epoch), out_dir / ("pred_obb_%04d.csv" % epoch))


def find_score_path(out_dir: Path, epoch: int) -> Path:
    return _first_existing(score_path(out_dir, epoch), out_dir / ("score_epoch_%04d.txt" % epoch))


def find_matched_errors_path(out_dir: Path, epoch: int) -> Path:
    return _first_existing(matched_errors_path(out_dir, epoch), out_dir / ("matched_errors_epoch_%04d.csv" % epoch))


def find_error_group_path(out_dir: Path, epoch: int) -> Path:
    return _first_existing(error_group_path(out_dir, epoch), out_dir / ("error_by_group_epoch_%04d.csv" % epoch))


def _move_if_needed(src: Path, dst: Path) -> Path:
    if dst.is_file():
        if src.is_file() and src.resolve() != dst.resolve():
            src.unlink()
        return dst
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return dst


def organize_obb_epoch(out_dir: Path, epoch: int) -> None:
    """Move V4a flat epoch files into the V5 OBB output layout."""
    ensure_obb_layout(out_dir)
    c_path = _move_if_needed(out_dir / ("catalog_sdc1_%04d.txt" % epoch), catalog_path(out_dir, epoch))
    p_path = _move_if_needed(out_dir / ("pred_obb_%04d.csv" % epoch), pred_obb_path(out_dir, epoch))
    s_path = _move_if_needed(out_dir / ("score_epoch_%04d.txt" % epoch), score_path(out_dir, epoch))

    if c_path.is_file():
        _safe_alias(c_path, out_dir / "catalog_sdc1.txt")
    if p_path.is_file():
        _safe_alias(p_path, out_dir / "pred_obb.csv")
    if s_path.is_file():
        _safe_alias(s_path, out_dir / "score_epoch.txt")
    (out_dir / "score_history.txt").unlink(missing_ok=True)


def organize_obb_diagnostics(out_dir: Path, epoch: int | None = None) -> None:
    """Move flat diagnostic CSV files into their dedicated folders."""
    ensure_obb_layout(out_dir)
    epochs = []
    if epoch is not None:
        epochs = [int(epoch)]
    else:
        for path in out_dir.glob("matched_errors_epoch_*.csv"):
            text = path.stem.split("_")[-1]
            if text.isdigit():
                epochs.append(int(text))
        for path in out_dir.glob("error_by_group_epoch_*.csv"):
            text = path.stem.split("_")[-1]
            if text.isdigit():
                epochs.append(int(text))
        epochs = sorted(set(epochs))

    for item in epochs:
        _move_if_needed(out_dir / ("matched_errors_epoch_%04d.csv" % item), matched_errors_path(out_dir, item))
        _move_if_needed(out_dir / ("error_by_group_epoch_%04d.csv" % item), error_group_path(out_dir, item))

    _move_if_needed(out_dir / "matched_errors.csv", matched_errors_latest_path(out_dir))
    _move_if_needed(out_dir / "error_by_group.csv", error_group_latest_path(out_dir))


def score_obb_epoch(out_dir: Path, epoch: int, truth_path: str, train_score: bool = False):
    c_path = find_catalog_path(out_dir, epoch)
    if not c_path.is_file():
        raise FileNotFoundError(c_path)
    s_path = score_path(out_dir, epoch)
    scorer = score_catalog(c_path, truth_path, train=train_score)
    result = result_from_scorer(epoch, scorer, c_path, s_path)
    write_score_epoch(s_path, result)
    _safe_alias(s_path, out_dir / "score_epoch.txt")
    return result, scorer


def write_obb_summary(
    out_dir: Path,
    results: list[ScoreResult],
    scorer_by_epoch: dict[int, object],
    diagnostic_epoch: int | None = None,
    include_errors: bool = True,
) -> ScoreResult:
    if not results:
        raise ValueError("No OBB score results to summarize.")

    write_score_history_csv(out_dir / "score_history.csv", results)
    best = write_score_summary(out_dir / "score_summary.txt", results)
    update_best_aliases(out_dir, best)

    pred_best = find_pred_obb_path(out_dir, best.epoch)
    if pred_best.is_file():
        _safe_alias(pred_best, out_dir / "best_pred_obb.csv")

    if not include_errors:
        return best

    diag_epoch = int(diagnostic_epoch) if diagnostic_epoch is not None else best.epoch
    if diag_epoch not in scorer_by_epoch:
        if best.epoch not in scorer_by_epoch:
            return best
        diag_epoch = best.epoch
    diag_scorer = scorer_by_epoch[diag_epoch]
    errors = make_matched_errors(diag_scorer)
    me_path = matched_errors_path(out_dir, diag_epoch)
    save_structured_csv(me_path, errors)
    _safe_alias(me_path, matched_errors_latest_path(out_dir))
    grouped = grouped_error_table(errors)
    eg_path = error_group_path(out_dir, diag_epoch)
    save_structured_csv(eg_path, grouped)
    _safe_alias(eg_path, error_group_latest_path(out_dir))
    (out_dir / "diagnostic_epoch.txt").write_text(
        "epoch: %04d\nmatched_errors: %s\nerror_by_group: %s\n"
        % (
            diag_epoch,
            me_path,
            eg_path,
        ),
        encoding="utf-8",
    )
    organize_obb_diagnostics(out_dir, diag_epoch)
    return best


def summarize_obb_outputs(
    out_dir: Path,
    epochs: list[int],
    truth_path: str,
    train_score: bool = False,
    diagnostic_epoch: int | None = None,
):
    """Rescore V4a OBB catalogs and write V5-style summary files."""
    ensure_obb_layout(out_dir)
    results = []
    scorer_by_epoch = {}

    for epoch in epochs:
        organize_obb_epoch(out_dir, epoch)
        result, scorer = score_obb_epoch(out_dir, epoch, truth_path, train_score=train_score)
        results.append(result)
        scorer_by_epoch[epoch] = scorer

    return write_obb_summary(out_dir, results, scorer_by_epoch, diagnostic_epoch=diagnostic_epoch)
