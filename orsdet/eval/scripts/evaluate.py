#!/usr/bin/env python3
"""Post-process one checkpoint at a time, score it, and build eval summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
SKAO_DIR = EVAL_DIR.parent
FLUX_HEAD_HOOK_SCRIPT = SKAO_DIR / "flux_head" / "scripts" / "build_flux_head.py"
FLUX_HEAD_APPLY_SCRIPT = SKAO_DIR / "flux_head" / "scripts" / "apply_flux_head.py"
sys.path.insert(0, str(EVAL_DIR / "src"))

from orsdet_eval.runtime import (  # noqa: E402
    ALL_PROFILES,
    OBB_PROFILES,
    ORSDET_PROFILE,
    available_epochs,
    configure_paths,
    default_out_dir,
    default_run_dir,
    drop_user_site,
    ensure_dirs,
    expected_fwd_bytes,
    fwd_file_complete,
    install_numba_fallback_if_needed,
    is_flux_head_profile,
    detector_mode_for_profile,
)


drop_user_site()

from orsdet_eval.hbb_post import finish_outputs, postprocess_epoch  # noqa: E402
from orsdet_eval.obb_post import (  # noqa: E402
    find_catalog_path,
    find_pred_obb_path,
    organize_obb_epoch,
    score_obb_epoch,
    write_obb_summary,
)
from orsdet_eval.score import load_score_history_csv  # noqa: E402


FLUX_HEAD_FORMAL_METADATA_FILES = (
    "train_norm.txt",
    "train_cat_norm_lims.txt",
    "TrainingSet_perscut.txt",
    "run_info.txt",
)
FLUX_HEAD_FORMAL_PROFILE_BY_SOURCE = {
    "shared_angle_obb_phys": "flux_head_shared_angle_target_source_obb_phys",
}


def default_flux_head_python() -> str:
    env_value = os.environ.get("FLUX_HEAD_PYTHON")
    if env_value:
        return env_value
    ciana4090 = Path.home() / ".conda" / "envs" / "CIANNA4090" / "bin" / "python"
    if ciana4090.is_file():
        return str(ciana4090)
    return sys.executable


def select_epochs(run_dir: Path, args, expected_bytes: int) -> list[int]:
    if args.epoch is not None:
        return [args.epoch]
    if args.latest:
        if args.run_pred:
            net_epochs = available_epochs(run_dir, "net_save", "net0_s")
            if net_epochs:
                return [net_epochs[-1]]
            raise FileNotFoundError("No checkpoint found in %s" % (run_dir / "net_save"))
        fwd_epochs = available_epochs(run_dir, "fwd_res", "net0_")
        fwd_epochs = [epoch for epoch in fwd_epochs if fwd_file_complete(run_dir / "fwd_res" / ("net0_%04d.dat" % epoch), expected_bytes)]
        if fwd_epochs:
            return [fwd_epochs[-1]]
        net_epochs = available_epochs(run_dir, "net_save", "net0_s")
        if net_epochs:
            return [net_epochs[-1]]
        raise FileNotFoundError("No fwd_res or checkpoint found in %s" % run_dir)
    source_subdir = "net_save" if args.run_pred else "fwd_res"
    prefix = "net0_s" if args.run_pred else "net0_"
    epochs = available_epochs(run_dir, source_subdir, prefix, args.epoch_start, args.epoch_end, args.epoch_interv)
    if not epochs:
        raise FileNotFoundError("No epochs found in %s/%s" % (run_dir, source_subdir))
    if not args.run_pred:
        epochs = [
            epoch
            for epoch in epochs
            if fwd_file_complete(run_dir / "fwd_res" / ("net0_%04d.dat" % epoch), expected_bytes)
        ]
    if not epochs:
        raise FileNotFoundError("No complete fwd_res files found. Run predict.py first or pass --run-pred.")
    return epochs


def expected_epoch_list(args) -> list[int] | None:
    if args.epoch is not None:
        return [int(args.epoch)]
    if args.epoch_start is None or args.epoch_end is None:
        return None
    step = max(1, int(args.epoch_interv))
    return list(range(int(args.epoch_start), int(args.epoch_end) + 1, step))


def candidate_epochs(run_dir: Path, args, expected_bytes: int) -> list[int]:
    if args.epoch is not None:
        epoch = int(args.epoch)
        if args.run_pred:
            return [epoch] if (run_dir / "net_save" / ("net0_s%04d.dat" % epoch)).is_file() else []
        fwd_path = run_dir / "fwd_res" / ("net0_%04d.dat" % epoch)
        return [epoch] if fwd_file_complete(fwd_path, expected_bytes) else []
    source_subdir = "net_save" if args.run_pred else "fwd_res"
    prefix = "net0_s" if args.run_pred else "net0_"
    epochs = available_epochs(run_dir, source_subdir, prefix, args.epoch_start, args.epoch_end, args.epoch_interv)
    if not args.run_pred:
        epochs = [
            epoch
            for epoch in epochs
            if fwd_file_complete(run_dir / "fwd_res" / ("net0_%04d.dat" % epoch), expected_bytes)
        ]
    return epochs


def checkpoint_stable(path: Path, stable_seconds: float) -> bool:
    if not path.is_file():
        return False
    try:
        first = path.stat()
    except OSError:
        return False
    if first.st_size <= 0:
        return False
    if stable_seconds <= 0:
        return True
    time.sleep(stable_seconds)
    try:
        second = path.stat()
    except OSError:
        return False
    return first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns


def prepare_obb_stage(out_dir: Path, epoch: int) -> Path:
    stage_dir = out_dir / "_staging" / ("epoch_%04d" % epoch)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir


def promote_obb_stage(stage_dir: Path, out_dir: Path, epoch: int) -> None:
    required = [
        "catalog_sdc1_%04d.txt" % epoch,
        "pred_obb_%04d.csv" % epoch,
    ]
    missing = [name for name in required if not (stage_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Post-processing stage for epoch %04d is incomplete; missing %s in %s"
            % (epoch, ", ".join(missing), stage_dir)
        )

    for name in required + ["score_epoch_%04d.txt" % epoch]:
        src = stage_dir / name
        if not src.is_file():
            continue
        dst = out_dir / name
        dst.unlink(missing_ok=True)
        shutil.move(str(src), str(dst))
    organize_obb_epoch(out_dir, epoch)
    shutil.rmtree(stage_dir, ignore_errors=True)


def append_obb_profile_args(command: list[str], args) -> None:
    detector_mode = detector_mode_for_profile(args.profile)
    if detector_mode is not None:
        command.extend(["--slim-mode", detector_mode])


def run_flux_head_checkpoint_hook(post_script: Path, run_dir: Path, out_dir: Path, epoch: int, args, env, truth_path: Path) -> None:
    if not args.flux_head_hook:
        return
    if not FLUX_HEAD_HOOK_SCRIPT.is_file():
        raise FileNotFoundError(FLUX_HEAD_HOOK_SCRIPT)
    if post_script.name not in {"post_detector.py"}:
        raise RuntimeError("ORSDet flux head hook currently supports detector post scripts only, got %s." % post_script)

    detector = run_dir / "net_save" / ("net0_s%04d.dat" % epoch)
    if not detector.is_file():
        raise FileNotFoundError("ORSDet flux head hook requires detector checkpoint: %s" % detector)

    flux_head_root = (args.flux_head_out_dir or (out_dir / "flux_head_checkpoint_hook")).resolve()
    train_stage_dir = flux_head_root / "_train_region_post" / ("epoch_%04d" % epoch)
    train_catalog = train_stage_dir / ("catalog_sdc1_%04d.txt" % epoch)
    train_pred_obb = train_stage_dir / ("pred_obb_%04d.csv" % epoch)
    if args.flux_head_force and train_stage_dir.exists():
        shutil.rmtree(train_stage_dir)
    if not (train_catalog.is_file() and train_pred_obb.is_file()):
        train_stage_dir.mkdir(parents=True, exist_ok=True)
        train_post_cmd = [sys.executable, str(post_script), str(epoch)]
        train_post_cmd.extend(["--src-run-dir", str(run_dir)])
        append_obb_profile_args(train_post_cmd, args)
        train_post_cmd.extend(["--out-dir", str(train_stage_dir), "--opt-rounds", "1", "--training-only"])
        print("Running ORSDet flux head training-region post:", " ".join(train_post_cmd), flush=True)
        subprocess.check_call(train_post_cmd, env=env)

    apply_catalog = find_catalog_path(out_dir, epoch)
    apply_pred_obb = find_pred_obb_path(out_dir, epoch)
    require = [apply_catalog, apply_pred_obb, train_catalog, train_pred_obb]
    missing = [str(path) for path in require if not path.is_file()]
    if missing:
        raise FileNotFoundError("ORSDet flux head hook missing inputs: %s" % ", ".join(missing))

    hook_out = flux_head_root / ("epoch_%04d" % epoch)
    hook_cmd = [
        str(args.flux_head_python),
        str(FLUX_HEAD_HOOK_SCRIPT),
        "--display",
        "%s@%04d" % (args.profile, epoch),
        "--epoch",
        str(epoch),
        "--detector",
        str(detector),
        "--train-catalog",
        str(train_catalog),
        "--train-pred-obb",
        str(train_pred_obb),
        "--apply-catalog",
        str(apply_catalog),
        "--apply-pred-obb",
        str(apply_pred_obb),
        "--truth-catalog",
        str(truth_path),
        "--out-dir",
        str(hook_out),
        "--device",
        str(args.flux_head_device),
        "--score",
        "--verify-dat",
    ]
    if args.flux_head_force:
        hook_cmd.append("--force")
    hook_env = dict(os.environ if env is None else env)
    hook_env.pop("PYTHONNOUSERSITE", None)
    print("Running ORSDet flux head checkpoint hook:", " ".join(hook_cmd), flush=True)
    subprocess.check_call(hook_cmd, env=hook_env)
    formal_dat = export_flux_head_formal_run(run_dir, hook_out, epoch, args)
    if args.flux_head_clean_source_checkpoint and formal_dat is not None and formal_dat.is_file():
        cleanup_flux_head_source_checkpoint(run_dir, epoch)


def cleanup_flux_head_source_checkpoint(run_dir: Path, epoch: int) -> None:
    """Drop raw detector artifacts after the formal ORSDet flux head single .dat is exported."""
    candidates = [
        run_dir / "net_save" / ("net0_s%04d.dat" % epoch),
        run_dir / "fwd_res" / ("net0_%04d.dat" % epoch),
    ]
    for path in candidates:
        if path.is_file():
            path.unlink()
            print("Removed ORSDet flux head source cache:", path, flush=True)


def export_flux_head_formal_run(src_run_dir: Path, hook_out: Path, epoch: int, args) -> Path | None:
    if args.flux_head_formal_run_dir is None:
        return None
    formal_profile = args.flux_head_formal_profile or FLUX_HEAD_FORMAL_PROFILE_BY_SOURCE.get(
        args.profile, ORSDET_PROFILE
    )
    report_path = hook_out / "flux_head_checkpoint_hook_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("ORSDet flux head formal export missing hook report: %s" % report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    dat_value = report.get("dat_path_abs") or report.get("dat_path")
    if not dat_value:
        raise RuntimeError("ORSDet flux head hook report has no dat_path: %s" % report_path)
    hook_dat = Path(str(dat_value))
    if not hook_dat.is_absolute():
        hook_dat = (hook_out / hook_dat).resolve()
    if not hook_dat.is_file():
        raise FileNotFoundError("ORSDet flux head hook dat not found: %s" % hook_dat)

    formal_run = args.flux_head_formal_run_dir.resolve()
    net_dir = formal_run / "net_save"
    net_dir.mkdir(parents=True, exist_ok=True)
    formal_dat = net_dir / ("net0_s%04d.dat" % epoch)
    if formal_dat.exists() and not args.flux_head_force:
        raise FileExistsError("ORSDet flux head formal .dat already exists: %s" % formal_dat)
    shutil.copy2(hook_dat, formal_dat)

    for name in FLUX_HEAD_FORMAL_METADATA_FILES:
        src = src_run_dir / name
        if src.is_file():
            shutil.copy2(src, formal_run / name)

    info = {
        "model": "ORSDet flux head decoded-candidate flux head",
        "profile": formal_profile,
        "epoch": int(epoch),
        "source_run_dir": str(src_run_dir.resolve()),
        "hook_out": str(hook_out.resolve()),
        "source_flux_head_dat": str(hook_dat.resolve()),
        "formal_flux_head_dat": str(formal_dat.resolve()),
        "note": "Formal run stores the deliverable one .dat per checkpoint: detector prefix + flux head trailer payload.",
    }
    (formal_run / "run_info.txt").write_text(
        "\n".join("%s=%s" % (key, value) for key, value in info.items()) + "\n",
        encoding="utf-8",
    )
    print("Exported ORSDet flux head formal single .dat:", formal_dat, flush=True)
    return formal_dat


def apply_flux_head_profile(run_dir: Path, out_dir: Path, epoch: int, args, env) -> None:
    if not is_flux_head_profile(args.profile):
        return
    if not FLUX_HEAD_APPLY_SCRIPT.is_file():
        raise FileNotFoundError(FLUX_HEAD_APPLY_SCRIPT)

    dat_package = run_dir / "net_save" / ("net0_s%04d.dat" % epoch)
    if not dat_package.is_file():
        raise FileNotFoundError("Missing ORSDet flux head .dat: %s" % dat_package)

    catalog = find_catalog_path(out_dir, epoch)
    pred_obb = find_pred_obb_path(out_dir, epoch)
    missing = [str(path) for path in (catalog, pred_obb) if not path.is_file()]
    if missing:
        raise FileNotFoundError("ORSDet flux head apply missing inputs: %s" % ", ".join(missing))

    stage_dir = out_dir / "flux_head" / ("epoch_%04d" % epoch)
    apply_dir = stage_dir / "apply"
    marker = stage_dir / "applied.json"
    corrected = apply_dir / "catalog_flux_head_post_head.txt"
    base_catalog = stage_dir / ("base_catalog_sdc1_%04d.txt" % epoch)

    if marker.is_file() and corrected.is_file() and not args.flux_head_force:
        print("ORSDet flux head already applied for epoch %04d." % epoch, flush=True)
        return

    stage_dir.mkdir(parents=True, exist_ok=True)
    if not base_catalog.is_file():
        shutil.copy2(catalog, base_catalog)
    if apply_dir.exists() and args.flux_head_force:
        shutil.rmtree(apply_dir)
    apply_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(FLUX_HEAD_APPLY_SCRIPT),
        "--flux-head-dat",
        str(dat_package),
        "--catalog",
        str(base_catalog),
        "--pred-obb",
        str(pred_obb),
        "--out-dir",
        str(apply_dir),
    ]
    print("Applying ORSDet flux head payload:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env=env)
    if not corrected.is_file():
        raise FileNotFoundError(corrected)
    shutil.copy2(corrected, catalog)
    organize_obb_epoch(out_dir, epoch)
    marker.write_text(
        json.dumps(
            {
                "epoch": int(epoch),
                "dat_package": str(dat_package.resolve()),
                "base_catalog": str(base_catalog.resolve()),
                "corrected_catalog": str(corrected.resolve()),
                "final_catalog": str(catalog.resolve()),
                "note": "eval formal ORSDet flux head profile applies the single .dat payload before scoring.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def predict_epoch_if_needed(run_dir: Path, args, expected_bytes: int, epoch: int, wait_mode: bool) -> bool:
    fwd_path = run_dir / "fwd_res" / ("net0_%04d.dat" % epoch)
    if fwd_file_complete(fwd_path, expected_bytes) and not args.overwrite_pred:
        print("fwd_res already complete: %s" % fwd_path, flush=True)
        return True

    if not args.run_pred:
        if wait_mode:
            print("Waiting for complete fwd_res for epoch %04d." % epoch, flush=True)
            return False
        raise FileNotFoundError("Missing complete fwd_res file: %s" % fwd_path)

    model_path = run_dir / "net_save" / ("net0_s%04d.dat" % epoch)
    if not checkpoint_stable(model_path, args.checkpoint_stable_seconds):
        if wait_mode:
            print("Checkpoint is not stable yet: %s" % model_path, flush=True)
            return False
        raise RuntimeError("Checkpoint is missing or still being written: %s" % model_path)

    pred_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "predict.py"),
        str(epoch),
        "--run-dir",
        str(run_dir),
        "--profile",
        args.profile,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.device is not None:
        pred_cmd.extend(["--device", str(args.device)])
    if args.overwrite_pred:
        pred_cmd.append("--overwrite")
    print("Running:", " ".join(pred_cmd), flush=True)
    subprocess.check_call(pred_cmd)
    if not fwd_file_complete(fwd_path, expected_bytes):
        raise RuntimeError("%s was not written completely." % fwd_path)
    return True


def run_best_diagnostics(
    out_dir: Path,
    args,
    best_epoch: int,
    refresh_plots: bool = False,
    refresh_regions: bool = False,
) -> None:
    """Refresh live diagnostics for the current best epoch.

    Plot/region scripts are intentionally called without --epoch here. That
    makes them use best_pred_obb.csv plus the latest diagnostic CSV aliases and
    overwrite the previous best visualizations instead of creating one file per
    scored checkpoint.
    """

    if args.live_plots or refresh_plots:
        plot_cmd = [sys.executable, str(SCRIPT_DIR / "plot_eval.py"), "--eval-dir", str(out_dir)]
        print("Refreshing best-epoch plots for %04d" % best_epoch, flush=True)
        print("Running:", " ".join(plot_cmd), flush=True)
        subprocess.check_call(plot_cmd)
    if args.live_regions and refresh_regions:
        vis_cmd = [sys.executable, str(SCRIPT_DIR / "vis_eval.py"), "--eval-dir", str(out_dir)]
        print("Refreshing best-epoch region visualizations for %04d" % best_epoch, flush=True)
        print("Running:", " ".join(vis_cmd), flush=True)
        subprocess.check_call(vis_cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epoch", nargs="?", type=int)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--profile",
        choices=ALL_PROFILES,
        default=ORSDET_PROFILE,
    )
    parser.add_argument("--run-pred", action="store_true", help="Run predict.py before scoring selected epochs.")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--epoch-start", type=int)
    parser.add_argument("--epoch-end", type=int)
    parser.add_argument("--epoch-interv", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8, help="Passed to predict.py when --run-pred is used.")
    parser.add_argument("--opt-rounds", type=int, default=4)
    parser.add_argument("--device", type=int, default=None, help="Passed to predict.py when --run-pred is used.")
    parser.add_argument("--overwrite-pred", action="store_true", help="Regenerate fwd_res even if a complete file exists.")
    parser.add_argument("--watch", action="store_true", help="Keep polling net_save/fwd_res and process new epochs as they appear.")
    parser.add_argument("--poll-seconds", type=float, default=60.0, help="Polling interval used by --watch.")
    parser.add_argument(
        "--checkpoint-stable-seconds",
        type=float,
        default=5.0,
        help="A net_save file must keep the same size and mtime for this many seconds before prediction.",
    )
    parser.add_argument(
        "--max-idle-polls",
        type=int,
        default=0,
        help="With --watch, stop after this many empty polls. 0 means wait until the requested epoch range is complete.",
    )
    parser.add_argument("--train-score", action="store_true", help="Score the training region instead of the test region.")
    parser.add_argument(
        "--auto-plot-min-epochs",
        type=int,
        default=5,
        help="For OBB profiles, refresh plots once the number of scored epochs is greater than this value.",
    )
    parser.add_argument(
        "--auto-plot-every",
        type=int,
        default=5,
        help="For OBB profiles, refresh intermediate plots every N scored epochs after --auto-plot-min-epochs.",
    )
    parser.add_argument("--no-auto-plots", action="store_true", help="Disable intermediate plot refresh for OBB profiles.")
    parser.add_argument("--live-plots", action="store_true", help="Refresh best-epoch plot_eval.py outputs during live scoring.")
    parser.add_argument("--live-regions", action="store_true", help="Refresh best-epoch vis_eval.py outputs when the best epoch changes.")
    parser.add_argument("--flux-head-hook", action="store_true", help="After each OBB epoch, train/package the ORSDet flux head checkpoint hook.")
    parser.add_argument("--flux-head-out-dir", type=Path, default=None, help="Output root for ORSDet flux head checkpoint hook artifacts.")
    parser.add_argument(
        "--flux-head-formal-run-dir",
        type=Path,
        default=None,
        help="Optional run directory receiving hook-packaged single .dat files as net_save/net0_sXXXX.dat.",
    )
    parser.add_argument(
        "--flux-head-formal-profile",
        choices=ALL_PROFILES,
        default=None,
        help="ORSDet flux head profile written into formal-run metadata; useful when exporting target-source or custom formal runs.",
    )
    parser.add_argument("--flux-head-python", default=default_flux_head_python(), help="Python executable used for ORSDet flux head hook training.")
    parser.add_argument(
        "--flux-head-device",
        default="cpu",
        help="Device passed to the flux head builder; default keeps the small flux head on CPU.",
    )
    parser.add_argument("--flux-head-force", action="store_true", help="Overwrite existing ORSDet flux head hook artifacts for an epoch.")
    parser.add_argument(
        "--flux-head-clean-source-checkpoint",
        action="store_true",
        help="After exporting the formal single .dat, remove the raw detector checkpoint and detector fwd cache for that epoch.",
    )
    args = parser.parse_args()
    if args.flux_head_formal_profile is not None and not is_flux_head_profile(args.flux_head_formal_profile):
        parser.error("--flux-head-formal-profile must be a ORSDet flux head profile.")

    configure_paths()
    install_numba_fallback_if_needed()

    import aux_fct as aux

    run_dir = (args.run_dir or default_run_dir(args.profile)).resolve()
    out_dir = (args.out_dir or default_out_dir(args.profile)).resolve()
    ensure_dirs(run_dir, out_dir)
    expected_bytes = expected_fwd_bytes(aux, args.profile)
    epochs = [] if args.watch else select_epochs(run_dir, args, expected_bytes)

    if args.profile in OBB_PROFILES:
        post_script = SKAO_DIR / "detector" / "scripts" / "post_detector.py"
        if not post_script.is_file():
            raise FileNotFoundError(post_script)

        env = None
        if args.device is not None:
            env = dict(**os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(args.device)
        print("run_dir:", run_dir)
        print("out_dir:", out_dir)
        print("profile:", args.profile)
        if args.watch:
            print("watch: enabled")
        else:
            print("epochs:", " ".join("%04d" % epoch for epoch in epochs))

        results_by_epoch = {
            result.epoch: result
            for result in load_score_history_csv(out_dir / "score_history.csv", out_dir)
        }
        results = list(results_by_epoch.values())
        scorer_by_epoch = {}
        best = max(results, key=lambda item: item.score) if results else None
        processed_epochs: set[int] = set() if args.overwrite_pred else set(results_by_epoch)
        idle_polls = 0
        expected_epochs = expected_epoch_list(args)

        while True:
            if args.watch:
                batch_epochs = [epoch for epoch in candidate_epochs(run_dir, args, expected_bytes) if epoch not in processed_epochs]
                is_last_known_epoch = False
                if not batch_epochs:
                    if expected_epochs is not None and set(expected_epochs).issubset(processed_epochs):
                        break
                    idle_polls += 1
                    if args.max_idle_polls > 0 and idle_polls >= args.max_idle_polls:
                        print("No new epochs after %d polls; stopping watch." % idle_polls, flush=True)
                        break
                    print("No new epoch. Sleeping %.1f seconds." % args.poll_seconds, flush=True)
                    time.sleep(max(0.1, args.poll_seconds))
                    continue
                idle_polls = 0
            else:
                batch_epochs = [epoch for epoch in epochs if epoch not in processed_epochs]
                is_last_known_epoch = True
                if not batch_epochs:
                    break

            for epoch in batch_epochs:
                if not predict_epoch_if_needed(run_dir, args, expected_bytes, epoch, wait_mode=args.watch):
                    continue
                print("Post-processing epoch %04d" % epoch, flush=True)
                processed_epochs.add(epoch)

                organize_obb_epoch(out_dir, epoch)
                if not (find_catalog_path(out_dir, epoch).is_file() and find_pred_obb_path(out_dir, epoch).is_file()):
                    stage_dir = prepare_obb_stage(out_dir, epoch)
                    post_cmd = [sys.executable, str(post_script), str(epoch)]
                    post_cmd.extend(["--src-run-dir", str(run_dir)])
                    append_obb_profile_args(post_cmd, args)
                    post_cmd.extend(["--out-dir", str(stage_dir), "--opt-rounds", str(args.opt_rounds)])
                    subprocess.check_call(post_cmd, env=env)
                    promote_obb_stage(stage_dir, out_dir, epoch)

                apply_flux_head_profile(run_dir, out_dir, epoch, args, env)
                result, scorer = score_obb_epoch(out_dir, epoch, aux.TRUTH_CATALOG_PATH, train_score=args.train_score)
                results_by_epoch[epoch] = result
                results = list(results_by_epoch.values())
                scorer_by_epoch[epoch] = scorer
                auto_plot_every = max(1, int(args.auto_plot_every))
                crossed_plot_threshold = len(results) == args.auto_plot_min_epochs + 1
                is_final_epoch = (not args.watch) and is_last_known_epoch and epoch == epochs[-1]
                should_refresh_plots = (
                    (not args.watch)
                    and (not args.no_auto_plots)
                    and len(results) > args.auto_plot_min_epochs
                    and (crossed_plot_threshold or len(results) % auto_plot_every == 0 or is_final_epoch)
                )
                current_best = max(results, key=lambda item: item.score)
                previous_best_epoch = best.epoch if best is not None else None
                best_changed = previous_best_epoch != current_best.epoch
                include_errors = args.epoch is not None or best_changed or should_refresh_plots or is_final_epoch
                diagnostic_epoch = current_best.epoch if (args.watch or args.live_plots or args.live_regions) else args.epoch
                best = write_obb_summary(
                    out_dir,
                    results,
                    scorer_by_epoch,
                    diagnostic_epoch=diagnostic_epoch,
                    include_errors=include_errors,
                )

                if best_changed:
                    print("best epoch updated: %04d score: %.10f" % (best.epoch, best.score), flush=True)
                run_flux_head_checkpoint_hook(post_script, run_dir, out_dir, epoch, args, env, Path(aux.TRUTH_CATALOG_PATH))
                if best_changed or should_refresh_plots:
                    run_best_diagnostics(
                        out_dir,
                        args,
                        best.epoch,
                        refresh_plots=should_refresh_plots,
                        refresh_regions=best_changed,
                    )

        print(out_dir / "score_summary.txt")
        if best is not None:
            print("best epoch: %04d score: %.10f" % (best.epoch, best.score))
        return

    print("run_dir:", run_dir)
    print("out_dir:", out_dir)
    print("profile:", args.profile)
    if args.watch:
        print("watch: enabled")
    else:
        print("epochs:", " ".join("%04d" % epoch for epoch in epochs))

    results = []
    scorer_by_epoch = {}
    processed_epochs: set[int] = set()
    idle_polls = 0
    expected_epochs = expected_epoch_list(args)
    while True:
        if args.watch:
            batch_epochs = [epoch for epoch in candidate_epochs(run_dir, args, expected_bytes) if epoch not in processed_epochs]
            if not batch_epochs:
                if expected_epochs is not None and set(expected_epochs).issubset(processed_epochs):
                    break
                idle_polls += 1
                if args.max_idle_polls > 0 and idle_polls >= args.max_idle_polls:
                    print("No new epochs after %d polls; stopping watch." % idle_polls, flush=True)
                    break
                print("No new epoch. Sleeping %.1f seconds." % args.poll_seconds, flush=True)
                time.sleep(max(0.1, args.poll_seconds))
                continue
            idle_polls = 0
        else:
            batch_epochs = [epoch for epoch in epochs if epoch not in processed_epochs]
            if not batch_epochs:
                break

        for epoch in batch_epochs:
            if not predict_epoch_if_needed(run_dir, args, expected_bytes, epoch, wait_mode=args.watch):
                continue
            processed_epochs.add(epoch)
            print("Post-processing epoch %04d" % epoch, flush=True)
            result, scorer = postprocess_epoch(
                epoch=epoch,
                run_dir=run_dir,
                out_dir=out_dir,
                aux=aux,
                truth_path=aux.TRUTH_CATALOG_PATH,
                opt_rounds=args.opt_rounds,
                train_score=args.train_score,
            )
            results.append((result, scorer))
            scorer_by_epoch[result.epoch] = scorer

    if not results:
        raise RuntimeError("No epoch was processed.")
    best_result = max((item[0] for item in results), key=lambda item: item.score)
    finish_outputs(out_dir, results, scorer_by_epoch[best_result.epoch])
    print(out_dir / "score_summary.txt")


if __name__ == "__main__":
    main()
