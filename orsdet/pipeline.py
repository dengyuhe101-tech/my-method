#!/usr/bin/env python3
"""Public train/test orchestration for the ORSDet release."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ORSDET = ROOT / "orsdet"
CIANNA = ROOT / "src"
TRAIN_INTERNAL = ORSDET / "flux_head" / "scripts" / "train_pipeline_internal.py"
EVAL_INTERNAL = ORSDET / "eval" / "scripts" / "evaluate.py"
PROFILE = "flux_head_shared_angle_target_source_obb_phys"
DEFAULT_CHECKPOINT = ROOT / "weights" / "net0_s2700.dat"
RAW_FILES = (
    "sdc1_560MHz_1000h.fits",
    "PrimaryBeam_560MHz.fits",
    "TrainingSet_560MHz.txt",
    "True_560MHz.txt",
)


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.check_call([str(part) for part in cmd], cwd=ROOT, env=env)


def raw_data_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_value = os.environ.get("SDC1_RAW_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (ROOT / "external_data" / "560Mhz-1kh").resolve()


def require_raw_data(path: Path) -> None:
    missing = [name for name in RAW_FILES if not (path / name).is_file()]
    if missing:
        raise SystemExit(
            "Missing official SDC1 raw files in %s:\n  %s\n"
            "Set SDC1_RAW_DATA_DIR or pass --raw-data-dir."
            % (path, "\n  ".join(missing))
        )


def runtime_env(raw_dir: Path | None = None, gpu: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    if raw_dir is not None:
        env["SDC1_RAW_DATA_DIR"] = str(raw_dir)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def materialize_checkpoint(checkpoint: Path, run_dir: Path, epoch: int) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    target = run_dir / "net_save" / ("net0_s%04d.dat" % int(epoch))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(checkpoint)
    except OSError:
        shutil.copy2(checkpoint, target)
    return target


def score_catalog(catalog: Path, truth: Path, *, train: bool = False) -> None:
    from ska_sdc import Sdc1Scorer

    catalog = catalog.expanduser().resolve()
    truth = truth.expanduser().resolve()
    if not catalog.is_file():
        raise SystemExit(f"Catalog not found: {catalog}")
    if not truth.is_file():
        raise SystemExit(f"Truth catalog not found: {truth}")

    scorer = Sdc1Scorer.from_txt(str(catalog), str(truth), freq=560, sub_skiprows=0, truth_skiprows=0)
    scorer.run(mode=0, train=train, detail=True)
    score = scorer.score
    purity = float(score.n_match / score.n_det) if int(score.n_det) else float("nan")
    print("score: %.10f" % float(score.value))
    print("n_det: %d" % int(score.n_det))
    print("n_match: %d" % int(score.n_match))
    print("n_bad: %d" % int(score.n_bad))
    print("n_false: %d" % int(score.n_false))
    print("acc: %.10f" % float(score.acc_pc))
    print("purity: %.10f" % purity)


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train ORSDet and package flux head checkpoints.")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "train")
    parser.add_argument("--epochs", type=int, default=2700)
    parser.add_argument("--epoch-start", type=int, default=100)
    parser.add_argument("--epoch-interv", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--control-interv", type=int, default=10)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--new-run", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = raw_data_dir(args.raw_data_dir)
    require_raw_data(raw_dir)
    run_dir = args.run_dir.expanduser().resolve()
    cmd = [
        sys.executable,
        TRAIN_INTERNAL,
        "--run-dir",
        run_dir,
        "--epochs",
        args.epochs,
        "--epoch-start",
        args.epoch_start,
        "--epoch-end",
        args.epochs,
        "--save-every",
        args.save_every,
        "--control-interv",
        args.control_interv,
    ]
    if args.epoch_interv is not None:
        cmd.extend(["--epoch-interv", args.epoch_interv])
    if args.gpu is not None:
        cmd.extend(["--gpu", args.gpu])
    if args.new_run:
        cmd.append("--new-run")
    run(cmd, env=runtime_env(raw_dir, args.gpu))


def test_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run ORSDet inference/evaluation or score an existing catalog.")
    parser.add_argument("--raw-data-dir", default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--truth", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "test_run")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "test_eval")
    parser.add_argument("--epoch", type=int, default=2700)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--no-run-pred", action="store_true")
    parser.add_argument("--train-score", action="store_true")
    args = parser.parse_args(argv)

    raw_dir = raw_data_dir(args.raw_data_dir)
    if args.catalog is not None:
        truth = args.truth or (raw_dir / "True_560MHz.txt")
        score_catalog(args.catalog, truth, train=args.train_score)
        return

    require_raw_data(raw_dir)
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    materialize_checkpoint(args.checkpoint, run_dir, args.epoch)

    cmd = [
        sys.executable,
        EVAL_INTERNAL,
        args.epoch,
        "--profile",
        PROFILE,
        "--run-dir",
        run_dir,
        "--out-dir",
        out_dir,
        "--batch-size",
        args.batch_size,
        "--flux-head-force",
    ]
    if not args.no_run_pred:
        cmd.append("--run-pred")
    if args.gpu is not None:
        cmd.extend(["--device", args.gpu])
    if args.train_score:
        cmd.append("--train-score")
    run(cmd, env=runtime_env(raw_dir, args.gpu))
