#!/usr/bin/env python3
"""Generate fwd_res files from CIANNA checkpoints for V5 evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
V5_DIR = SCRIPT_DIR.parent
SKAO_DIR = V5_DIR.parent
sys.path.insert(0, str(V5_DIR / "src"))

from orsdet_eval.runtime import (  # noqa: E402
    ALL_PROFILES,
    DEFAULT_RUN_DIR,
    OBB_PROFILES,
    available_epochs,
    configure_paths,
    default_run_dir,
    drop_user_site,
    ensure_dirs,
    expected_fwd_bytes,
    fwd_file_complete,
    set_cuda_device,
    v4d_mode_for_profile,
    v4e_mode_for_profile,
    v4f_mode_for_profile,
    v4g_mode_for_profile,
    v4h_mode_for_profile,
    v4i_base_for_profile,
    v4i_mode_for_profile,
)


drop_user_site()

import numpy as np  # noqa: E402


def i_ar(values):
    return np.asarray(values, dtype="int")


def checkpoint_epochs(run_dir: Path, epoch_start=None, epoch_end=None, epoch_interv: int = 1) -> list[int]:
    epochs = available_epochs(run_dir, "net_save", "net0_s", epoch_start, epoch_end, epoch_interv)
    if not epochs:
        raise FileNotFoundError("No checkpoint found in %s" % (run_dir / "net_save"))
    return epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epoch", nargs="?", type=int)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--profile",
        choices=ALL_PROFILES,
        default="v4a_obb",
    )
    parser.add_argument("--latest", action="store_true", help="Only predict the latest checkpoint.")
    parser.add_argument("--epoch-start", type=int)
    parser.add_argument("--epoch-end", type=int)
    parser.add_argument("--epoch-interv", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=int, default=None, help="Sets CUDA_VISIBLE_DEVICES before importing CIANNA.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    set_cuda_device(args.device)
    configure_paths()

    import aux_fct as aux

    run_dir = (args.run_dir or default_run_dir(args.profile)).resolve()
    ensure_dirs(run_dir)
    if args.epoch is not None:
        epochs = [args.epoch]
    elif args.latest:
        epochs = [checkpoint_epochs(run_dir)[-1]]
    else:
        epochs = checkpoint_epochs(run_dir, args.epoch_start, args.epoch_end, args.epoch_interv)

    expected_bytes = expected_fwd_bytes(aux, args.profile)
    missing = []
    for epoch in epochs:
        fwd_path = run_dir / "fwd_res" / ("net0_%04d.dat" % epoch)
        if args.overwrite or not fwd_file_complete(fwd_path, expected_bytes):
            missing.append(epoch)

    print("run_dir:", run_dir)
    print("profile:", args.profile)
    print("epochs:", " ".join("%04d" % epoch for epoch in epochs))
    if args.device is not None:
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not missing:
        print("All requested fwd_res files already exist and have the expected size.")
        return
    print("epochs_to_predict:", " ".join("%04d" % epoch for epoch in missing))
    if args.dry_run:
        return

    if args.profile in OBB_PROFILES:
        if args.profile == "v4a_obb":
            pred_script = SKAO_DIR / "detector" / "scripts" / "pred_detector.py"
        elif args.profile == "v4b_obb_phys":
            pred_script = SKAO_DIR / "detector" / "scripts" / "pred_detector.py"
        elif v4f_mode_for_profile(args.profile) is not None:
            pred_script = SKAO_DIR / "detector" / "scripts" / "pred_detector.py"
        elif v4g_mode_for_profile(args.profile) is not None:
            pred_script = SKAO_DIR / "detector" / "scripts" / "pred_detector.py"
        elif v4h_mode_for_profile(args.profile) is not None:
            pred_script = SKAO_DIR / "detector" / "scripts" / "pred_detector.py"
        elif v4i_base_for_profile(args.profile) == "ecs":
            pred_script = SKAO_DIR / "detector" / "scripts" / "pred_detector.py"
        elif v4i_base_for_profile(args.profile) == "dsa":
            pred_script = SKAO_DIR / "detector" / "scripts" / "pred_detector.py"
        elif v4e_mode_for_profile(args.profile) is not None:
            pred_script = SKAO_DIR / "detector" / "scripts" / "pred_detector.py"
        else:
            pred_script = SKAO_DIR / "detector" / "scripts" / "pred_detector.py"
        if not pred_script.is_file():
            raise FileNotFoundError(pred_script)
        for epoch in missing:
            cmd = [
                sys.executable,
                str(pred_script),
                str(epoch),
                "--run-dir",
                str(run_dir),
                "--batch-size",
                str(args.batch_size),
            ]
            v4d_mode = v4d_mode_for_profile(args.profile)
            if v4d_mode is not None:
                cmd.extend(["--slim-mode", v4d_mode])
            v4e_mode = v4e_mode_for_profile(args.profile)
            if v4e_mode is not None:
                cmd.extend(["--slim-mode", v4e_mode])
            v4f_mode = v4f_mode_for_profile(args.profile)
            if v4f_mode is not None:
                cmd.extend(["--slim-mode", v4f_mode])
            v4g_mode = v4g_mode_for_profile(args.profile)
            if v4g_mode is not None:
                cmd.extend(["--slim-mode", v4g_mode])
            v4h_mode = v4h_mode_for_profile(args.profile)
            if v4h_mode is not None:
                cmd.extend(["--slim-mode", v4h_mode])
            v4i_mode = v4i_mode_for_profile(args.profile)
            if v4i_mode is not None:
                cmd.extend(["--slim-mode", v4i_mode])
            print(" ".join(cmd), flush=True)
            subprocess.check_call(cmd)
        return

    import CIANNA as cnn
    import data_gen as dg

    os.chdir(run_dir)
    dg.dataset_perscut(dg.TRAINING_CATALOG_PATH, "TrainingSet_perscut.txt", 18)
    cnn.init(
        in_dim=i_ar([dg.fwd_image_size, dg.fwd_image_size]),
        in_nb_ch=1,
        out_dim=1 + dg.max_nb_obj_per_image * (7 + dg.nb_param),
        bias=0.1,
        b_size=args.batch_size,
        comp_meth="C_CUDA",
        dynamic_load=1,
        mixed_precision="FP32C_FP32A",
        adv_size=30,
        inference_only=1,
    )
    dg.init_data_gen()
    input_data = dg.create_full_pred()
    targets = np.zeros((input_data.shape[0], 1 + dg.max_nb_obj_per_image * (7 + dg.nb_param)), dtype="float32")
    cnn.create_dataset("TEST", input_data.shape[0], input_data[:, :], targets[:, :])
    cnn.set_yolo_params(raw_output=0)

    for epoch in missing:
        model_path = run_dir / "net_save" / ("net0_s%04d.dat" % epoch)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        print("Predicting epoch %04d" % epoch, flush=True)
        cnn.load(str(model_path), epoch, bin=1)
        cnn.forward(no_error=1, saving=2, repeat=1, drop_mode="AVG_MODEL")
        fwd_path = run_dir / "fwd_res" / ("net0_%04d.dat" % epoch)
        if not fwd_file_complete(fwd_path, expected_bytes):
            raise RuntimeError("%s was not written completely." % fwd_path)
        print(fwd_path, flush=True)


if __name__ == "__main__":
    main()
