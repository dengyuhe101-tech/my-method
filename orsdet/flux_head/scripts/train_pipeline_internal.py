#!/usr/bin/env python3
"""Run the V4m single-GPU sequential training pipeline.

V4m is trained as a checkpoint-integrated two-stage model:

1. Train the V4d-SA/V1a detector to the next checkpoint.
2. Freeze that checkpoint, build the Stage9 decoded-candidate flux head, and
   export detector + Stage9 as one formal .dat.
3. Continue detector training from that checkpoint and repeat.

The formal prediction/evaluation watcher remains a separate command so users
can decide when and where to score the packaged single-file .dat checkpoints.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
V4M_DIR = SCRIPT_DIR.parent
SKAO_DIR = V4M_DIR.parent
REPO_DIR = SKAO_DIR.parents[1]
V4D_POST_SCRIPT = SKAO_DIR / "detector" / "scripts" / "post_detector.py"
V4D_PRED_SCRIPT = SKAO_DIR / "detector" / "scripts" / "pred_detector.py"
V4D_TRAIN_SCRIPT = SKAO_DIR / "detector" / "scripts" / "train_detector.py"
STAGE9_HOOK_SCRIPT = SCRIPT_DIR / "build_flux_head.py"
FORMAL_METADATA_FILES = ("train_norm.txt", "train_cat_norm_lims.txt", "TrainingSet_perscut.txt")


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_DIR))
    except ValueError:
        return str(path)


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (V4M_DIR / path).resolve()


def cmd_to_text(cmd: list[str]) -> str:
    return " ".join("'%s'" % item if " " in item else item for item in cmd)


def run_logged(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[%s] cwd: %s" % (name, cwd), flush=True)
    print("[%s] log: %s" % (name, log_path), flush=True)
    print("[%s] cmd: %s" % (name, cmd_to_text(cmd)), flush=True)
    with log_path.open("ab") as log_file:
        subprocess.check_call(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def checkpoint_epochs(*, epoch_start: int, epoch_end: int, epoch_interv: int) -> list[int]:
    if epoch_start <= 0:
        raise ValueError("--epoch-start must be positive.")
    if epoch_end < epoch_start:
        raise ValueError("--epoch-end must be >= --epoch-start.")
    epochs = list(range(epoch_start, epoch_end + 1, epoch_interv))
    if not epochs or epochs[-1] != epoch_end:
        epochs.append(epoch_end)
    return epochs


def clean_detector_cache(detector_run: Path) -> None:
    """Remove raw detector artifacts after all formal V4m .dat files are exported."""
    for subdir_name in ("net_save", "fwd_res"):
        subdir = detector_run / subdir_name
        if not subdir.is_dir():
            continue
        for path in subdir.glob("net0*.dat"):
            if path.is_file():
                path.unlink()
                print("Removed detector cache:", path, flush=True)
    for pattern in ("angle_report_epoch_*",):
        for path in detector_run.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)


def detector_fwd_complete(path: Path, *, allow_training_roi: bool = False) -> bool:
    # V4d-SA/V1a has 9 boxes * (8 YOLO fields + 5 aux fields), 32x32 grid,
    # and 69x69 full-image tiles. Keep this local so interrupted checkpoint
    # predictions are detected before Stage9 post-processing starts.
    expected_floats = 69 * 69 * 9 * (8 + 5) * 32 * 32
    expected_bytes = expected_floats * 4
    if path.is_file() and path.stat().st_size == expected_bytes:
        return True
    if not allow_training_roi or not path.is_file():
        return False
    marker = path.with_name(path.stem + ".training_roi.npz")
    if not marker.is_file():
        return False
    try:
        import numpy as np

        meta = np.load(marker)
        tile_indices = np.asarray(meta["tile_indices"], dtype=np.int64)
        channels = int(np.asarray(meta["channels"]).reshape(-1)[0])
        yolo_nb_reg = int(np.asarray(meta["yolo_nb_reg"]).reshape(-1)[0])
        nb_area_h = int(np.asarray(meta["nb_area_h"]).reshape(-1)[0])
        nb_area_w = int(np.asarray(meta["nb_area_w"]).reshape(-1)[0])
    except Exception:
        return False
    if nb_area_h != 69 or nb_area_w != 69 or channels != 9 * (8 + 5) or yolo_nb_reg != 32:
        return False
    return path.stat().st_size == int(tile_indices.size * channels * yolo_nb_reg * yolo_nb_reg * 4)


def copy_formal_metadata(*, detector_run: Path, formal_run: Path, epoch: int) -> None:
    for name in FORMAL_METADATA_FILES:
        src = detector_run / name
        if src.is_file():
            shutil.copy2(src, formal_run / name)
    info = {
        "model": "V4m Stage9 decoded-candidate flux head",
        "profile": "v4m_stage9_v4d_sa_v1a_obb_phys",
        "epoch_latest_exported": int(epoch),
        "source_run_dir": str(detector_run.resolve()),
        "note": "Formal run stores one deliverable .dat per checkpoint: detector prefix + Stage9 trailer payload.",
    }
    (formal_run / "run_info.txt").write_text(
        "\n".join("%s=%s" % (key, value) for key, value in info.items()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/V4m_SA_V1a_train"),
        help="Formal V4m run directory. Relative paths are resolved under flux_head/.",
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--control-interv", type=int, default=10)
    parser.add_argument("--epoch-start", type=int, default=100)
    parser.add_argument("--epoch-end", type=int, default=None)
    parser.add_argument("--epoch-interv", type=int, default=None)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES used by both detector training and Stage9 build.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--stage9-roi-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use training-region ROI forward for Stage9 checkpoint builds. Formal eval evaluation remains full-image.",
    )
    parser.add_argument("--stage9-roi-halo-tiles", type=int, default=2)
    parser.add_argument(
        "--keep-source-checkpoint",
        action="store_true",
        help="Keep raw detector checkpoints in .cache/detector after formal .dat export.",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Pass --new-run to the first detector training segment.",
    )
    args = parser.parse_args()

    formal_run = resolve_output_path(args.run_dir)
    cache_root = formal_run / ".cache"
    detector_run = cache_root / "detector"
    train_post_root = cache_root / "stage9_train_post"
    hook_root = cache_root / "stage9_hook"
    log_dir = cache_root / "logs"
    epoch_end = int(args.epoch_end if args.epoch_end is not None else args.epochs)
    epoch_interv = int(args.epoch_interv if args.epoch_interv is not None else args.save_every)
    epochs = checkpoint_epochs(epoch_start=int(args.epoch_start), epoch_end=epoch_end, epoch_interv=epoch_interv)

    for path in (
        detector_run / "net_save",
        detector_run / "fwd_res",
        formal_run / "net_save",
        formal_run / "fwd_res",
        train_post_root,
        hook_root,
        log_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("V4m sequential training pipeline", flush=True)
    print("  formal run : %s" % rel_or_abs(formal_run), flush=True)
    print("  detector   : %s" % rel_or_abs(detector_run), flush=True)
    print("  train post : %s" % rel_or_abs(train_post_root), flush=True)
    print("  stage9 hook: %s" % rel_or_abs(hook_root), flush=True)
    print("  logs       : %s" % rel_or_abs(log_dir), flush=True)
    print("  epochs     : %s" % " ".join(str(epoch) for epoch in epochs), flush=True)
    if args.gpu is not None:
        print("  gpu        : %s" % args.gpu, flush=True)

    previous_epoch = 0
    for checkpoint_epoch in epochs:
        segment_epochs = checkpoint_epoch - previous_epoch
        if segment_epochs <= 0:
            continue
        checkpoint_path = detector_run / "net_save" / ("net0_s%04d.dat" % checkpoint_epoch)
        formal_path = formal_run / "net_save" / ("net0_s%04d.dat" % checkpoint_epoch)

        if not checkpoint_path.is_file():
            train_cmd = [
                args.python,
                str(V4D_TRAIN_SCRIPT),
                "--slim-mode",
                "v4d_sa",
                "--target-source",
                "v1a",
                "--run-dir",
                str(detector_run),
                "--epochs",
                str(segment_epochs),
                "--control-interv",
                str(args.control_interv),
                "--save-every",
                str(args.save_every),
            ]
            if previous_epoch > 0:
                train_cmd.append(str(previous_epoch))
            elif args.new_run:
                train_cmd.append("--new-run")
            if checkpoint_epoch != epoch_end:
                train_cmd.append("--skip-valid-angle-report")
            run_logged(
                name="detector_%04d" % checkpoint_epoch,
                cmd=train_cmd,
                cwd=REPO_DIR,
                env=env,
                log_path=log_dir / ("detector_to_%04d.log" % checkpoint_epoch),
            )
        else:
            print("Detector checkpoint exists, skipping training segment: %s" % checkpoint_path, flush=True)

        if formal_path.is_file():
            print("Formal V4m .dat exists, skipping Stage9 build: %s" % formal_path, flush=True)
        else:
            fwd_path = detector_run / "fwd_res" / ("net0_%04d.dat" % checkpoint_epoch)
            if fwd_path.is_file() and not detector_fwd_complete(
                fwd_path,
                allow_training_roi=bool(args.stage9_roi_forward),
            ):
                print("Removing incomplete detector fwd_res:", fwd_path, flush=True)
                fwd_path.unlink()
            if not detector_fwd_complete(
                fwd_path,
                allow_training_roi=bool(args.stage9_roi_forward),
            ):
                pred_cmd = [
                    args.python,
                    str(V4D_PRED_SCRIPT),
                    str(checkpoint_epoch),
                    "--run-dir",
                    str(detector_run),
                    "--batch-size",
                    "8",
                    "--slim-mode",
                    "v4d_sa",
                ]
                if args.stage9_roi_forward:
                    pred_cmd.extend(
                        [
                            "--training-roi-only",
                            "--roi-halo-tiles",
                            str(args.stage9_roi_halo_tiles),
                        ]
                    )
                run_logged(
                    name="pred_%04d" % checkpoint_epoch,
                    cmd=pred_cmd,
                    cwd=REPO_DIR,
                    env=env,
                    log_path=log_dir / ("pred_%04d.log" % checkpoint_epoch),
                )
            else:
                print("Detector fwd_res exists, skipping prediction: %s" % fwd_path, flush=True)

            train_post_dir = train_post_root / ("epoch_%04d" % checkpoint_epoch)
            train_catalog = train_post_dir / ("catalog_sdc1_%04d.txt" % checkpoint_epoch)
            train_pred_obb = train_post_dir / ("pred_obb_%04d.csv" % checkpoint_epoch)
            if not (train_catalog.is_file() and train_pred_obb.is_file()):
                if train_post_dir.exists():
                    shutil.rmtree(train_post_dir)
                train_post_cmd = [
                    args.python,
                    str(V4D_POST_SCRIPT),
                    str(checkpoint_epoch),
                    "--src-run-dir",
                    str(detector_run),
                    "--slim-mode",
                    "v4d_sa",
                    "--out-dir",
                    str(train_post_dir),
                    "--opt-rounds",
                    "1",
                    "--training-only",
                ]
                run_logged(
                    name="train_post_%04d" % checkpoint_epoch,
                    cmd=train_post_cmd,
                    cwd=REPO_DIR,
                    env=env,
                    log_path=log_dir / ("train_post_%04d.log" % checkpoint_epoch),
                )
            else:
                print("Training-region post exists, skipping: %s" % train_post_dir, flush=True)

            hook_out = hook_root / ("epoch_%04d" % checkpoint_epoch)
            if hook_out.exists():
                shutil.rmtree(hook_out)
            hook_env = env.copy()
            hook_env.pop("PYTHONNOUSERSITE", None)
            hook_cmd = [
                args.python,
                str(STAGE9_HOOK_SCRIPT),
                "--display",
                "V4m_SA_V1a@%04d" % checkpoint_epoch,
                "--epoch",
                str(checkpoint_epoch),
                "--detector",
                str(checkpoint_path),
                "--train-catalog",
                str(train_catalog),
                "--train-pred-obb",
                str(train_pred_obb),
                "--apply-catalog",
                str(train_catalog),
                "--apply-pred-obb",
                str(train_pred_obb),
                "--out-dir",
                str(hook_out),
                "--out-dat",
                str(formal_path),
                "--verify-dat",
                "--force",
            ]
            run_logged(
                name="stage9_%04d" % checkpoint_epoch,
                cmd=hook_cmd,
                cwd=REPO_DIR,
                env=hook_env,
                log_path=log_dir / ("stage9_%04d.log" % checkpoint_epoch),
            )
            copy_formal_metadata(detector_run=detector_run, formal_run=formal_run, epoch=checkpoint_epoch)
        previous_epoch = checkpoint_epoch

    if not args.keep_source_checkpoint:
        clean_detector_cache(detector_run)

    print("V4m sequential training pipeline finished successfully.", flush=True)
    print("Formal single .dat directory: %s" % rel_or_abs(formal_run / "net_save"), flush=True)


if __name__ == "__main__":
    main()
