#!/usr/bin/env python3
"""Post-process detector predictions with OBB NMS and physical SDC1 shape output."""

from __future__ import annotations

import argparse
import importlib
import os
import site
import subprocess
import sys
import types
from pathlib import Path


def drop_user_site():
    os.environ["PYTHONNOUSERSITE"] = "1"
    try:
        user_paths = site.getusersitepackages()
    except Exception:
        user_paths = []
    if isinstance(user_paths, str):
        user_paths = [user_paths]
    if os.environ.get("PYTHONUSERBASE"):
        user_paths.append(str(Path(os.environ["PYTHONUSERBASE"]).expanduser()))

    resolved_user_paths = []
    for path in user_paths:
        if not path:
            continue
        try:
            resolved_user_paths.append(Path(path).expanduser().resolve())
        except OSError:
            continue

    if not resolved_user_paths:
        return

    filtered = []
    for path_entry in sys.path:
        if not path_entry:
            filtered.append(path_entry)
            continue
        try:
            resolved = Path(path_entry).expanduser().resolve()
        except OSError:
            filtered.append(path_entry)
            continue
        if any(resolved == user_path or user_path in resolved.parents for user_path in resolved_user_paths):
            continue
        filtered.append(path_entry)
    sys.path[:] = filtered


drop_user_site()

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

    stub = types.ModuleType("tqdm")
    stub.tqdm = tqdm
    sys.modules["tqdm"] = stub

try:
    from numba import jit
except ImportError:
    def jit(*jit_args, **jit_kwargs):
        if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
            return jit_args[0]

        def decorator(func):
            return func

        return decorator


SCRIPT_DIR = Path(__file__).resolve().parent
DETECTOR_DIR = SCRIPT_DIR.parent
SKAO_DIR = DETECTOR_DIR.parent
NMS_DIR = SKAO_DIR / "nms"
CANDIDATE_DIR = SKAO_DIR / "candidates"
sys.path.insert(0, str(DETECTOR_DIR / "src"))
sys.path.insert(0, str(NMS_DIR / "src"))
sys.path.insert(0, str(CANDIDATE_DIR / "src"))
sys.path.insert(0, str(SKAO_DIR / "angle" / "src"))
sys.path.insert(0, str(SKAO_DIR / "target_source" / "src"))
sys.path.insert(0, str(SKAO_DIR / "geometry" / "src"))
sys.path.insert(0, str(SKAO_DIR))

from orsdet_detector import DEFAULT_RUN_DIR, DETECTOR_NB_PARAM, DETECTOR_TOTAL_AUX, configure_paths as configure_detector_paths
from orsdet_detector import decode_rows_obb, normalize_slim_mode, detector_catalog_arrays, detector_layout


DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "post"
CURRENT_TOTAL_AUX = DETECTOR_TOTAL_AUX
CURRENT_NB_PARAM = DETECTOR_NB_PARAM
CANDIDATE_ORIGIN_COLUMNS = (
    "tile_y",
    "tile_x",
    "cell_y",
    "cell_x",
    "grid_index",
    "local_candidate_id",
)
PRED_OBB_ORIGIN_COLUMNS = CANDIDATE_ORIGIN_COLUMNS + (
    "fwd_base_offset",
    "fwd_delta_param_index",
    "fwd_delta_channel",
    "fwd_delta_offset",
)


def install_numba_fallback_if_needed():
    try:
        import numba  # noqa: F401
        return
    except ImportError:
        pass

    def jit(*jit_args, **jit_kwargs):
        if jit_args and callable(jit_args[0]) and len(jit_args) == 1 and not jit_kwargs:
            return jit_args[0]

        def decorator(func):
            return func

        return decorator

    stub = types.ModuleType("numba")
    stub.jit = jit
    sys.modules["numba"] = stub


def expected_fwd_floats() -> int:
    return int(nb_area_h * nb_area_w * nb_box * (8 + CURRENT_TOTAL_AUX) * yolo_nb_reg * yolo_nb_reg)


def roi_marker_path(path: Path) -> Path:
    return path.with_name(path.stem + ".training_roi.npz")


def roi_fwd_file_complete(path: Path) -> bool:
    marker = roi_marker_path(path)
    if not (path.is_file() and marker.is_file()):
        return False
    try:
        meta = np.load(marker)
        tile_indices = np.asarray(meta["tile_indices"], dtype=np.int64)
        marker_channels = int(np.asarray(meta["channels"]).reshape(-1)[0])
        marker_yolo_nb_reg = int(np.asarray(meta["yolo_nb_reg"]).reshape(-1)[0])
        marker_nb_area_h = int(np.asarray(meta["nb_area_h"]).reshape(-1)[0])
        marker_nb_area_w = int(np.asarray(meta["nb_area_w"]).reshape(-1)[0])
    except Exception:
        return False
    if marker_nb_area_h != nb_area_h or marker_nb_area_w != nb_area_w:
        return False
    if marker_channels != nb_box * (8 + CURRENT_TOTAL_AUX) or marker_yolo_nb_reg != yolo_nb_reg:
        return False
    expected_bytes = int(tile_indices.size * marker_channels * yolo_nb_reg * yolo_nb_reg * 4)
    return path.stat().st_size == expected_bytes


def fwd_file_complete(path: Path, *, allow_roi: bool = False) -> bool:
    try:
        full_ok = path.is_file() and path.stat().st_size == expected_fwd_floats() * 4
        return full_ok or (allow_roi and roi_fwd_file_complete(path))
    except NameError:
        return path.is_file()


def read_run_slim_mode(run_dir: Path) -> str | None:
    info = run_dir / "run_info.txt"
    if not info.is_file():
        return None
    for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("slim_mode="):
            return line.split("=", 1)[1].split()[0]
    return None


@jit(nopython=True, fastmath=False)
def tile_filter_candidates(c_pred, c_box, c_tile, nb_box, nb_aux, prob_obj_cases, patch, val_med_lim, val_med_obj, hist_count):
    c_nb_box = 0
    for i in range(0, yolo_nb_reg):
        for j in range(0, yolo_nb_reg):
            kept_count = 0
            for k in range(0, nb_box):
                offset = int(k * (8 + nb_aux))
                c_box[4] = c_pred[offset + 6, i, j]
                c_box[5] = c_pred[offset + 7, i, j]
                if j == 0 or j == yolo_nb_reg - 1 or i == 0 or i == yolo_nb_reg - 1:
                    c_box[4] = max(0.03, c_box[4] - 0.05)
                    c_box[5] = max(0.03, c_box[5] - 0.05)

                if c_box[5] >= prob_obj_cases[k]:
                    bx = (c_pred[offset + 0, i, j] + c_pred[offset + 3, i, j]) * 0.5
                    by = (c_pred[offset + 1, i, j] + c_pred[offset + 4, i, j]) * 0.5
                    bw = max(5.0, c_pred[offset + 3, i, j] - c_pred[offset + 0, i, j])
                    bh = max(5.0, c_pred[offset + 4, i, j] - c_pred[offset + 1, i, j])

                    c_box[0] = bx - bw * 0.5
                    c_box[1] = by - bh * 0.5
                    c_box[2] = bx + bw * 0.5
                    c_box[3] = by + bh * 0.5

                    xmin = max(0, int(c_box[0] - 5))
                    xmax = min(fwd_image_size, int(c_box[2] + 5))
                    ymin = max(0, int(c_box[1] - 5))
                    ymax = min(fwd_image_size, int(c_box[3] + 5))

                    med_val_box = np.median(patch[ymin:ymax, xmin:xmax])
                    if (
                        (med_val_box > val_med_lim[0] and c_box[5] < val_med_obj[0])
                        or (med_val_box > val_med_lim[1] and c_box[5] < val_med_obj[1])
                        or (med_val_box > val_med_lim[2] and c_box[5] < val_med_obj[2])
                    ):
                        continue

                    c_box[6] = k
                    c_box[7 : 7 + nb_aux] = c_pred[offset + 8 : offset + 8 + nb_aux, i, j]
                    c_box[-1] = i * yolo_nb_reg + j
                    c_tile[c_nb_box, :] = c_box[:]
                    c_nb_box += 1
                    kept_count += 1

            hist_count[kept_count] += 1

    return c_nb_box


def latest_epoch(run_dir: Path, subdir: str, prefix: str, *, allow_roi: bool = False):
    epochs = []
    for path in (run_dir / subdir).glob("%s*.dat" % prefix):
        try:
            epoch = int(path.stem.split(prefix)[-1])
        except ValueError:
            continue
        if subdir == "fwd_res" and not fwd_file_complete(path, allow_roi=allow_roi):
            continue
        epochs.append(epoch)
    return max(epochs) if epochs else None


def available_epochs(
    run_dir: Path,
    subdir: str,
    prefix: str,
    epoch_start=None,
    epoch_end=None,
    epoch_interv: int = 1,
    *,
    allow_roi: bool = False,
):
    epochs = []
    for path in (run_dir / subdir).glob("%s*.dat" % prefix):
        try:
            epoch = int(path.stem.split(prefix)[-1])
        except ValueError:
            continue
        if subdir == "fwd_res" and not fwd_file_complete(path, allow_roi=allow_roi):
            continue
        if epoch_start is not None and epoch < epoch_start:
            continue
        if epoch_end is not None and epoch > epoch_end:
            continue
        if epoch_start is not None and epoch_interv > 1 and (epoch - epoch_start) % epoch_interv != 0:
            continue
        epochs.append(epoch)
    return sorted(set(epochs))


def select_epochs(src_run_dir: Path, args):
    if args.epoch is not None:
        return [args.epoch]
    if args.latest:
        epoch = latest_epoch(src_run_dir, "fwd_res", "net0_", allow_roi=bool(args.training_only))
        if epoch is None:
            epoch = latest_epoch(src_run_dir, "net_save", "net0_s")
        if epoch is None:
            raise FileNotFoundError("No complete fwd_res/net0_*.dat or net_save/net0_s*.dat found in %s" % src_run_dir)
        return [epoch]
    if args.run_pred:
        epochs = available_epochs(src_run_dir, "net_save", "net0_s", args.epoch_start, args.epoch_end, args.epoch_interv)
        if not epochs:
            raise FileNotFoundError("No checkpoint found in %s" % (src_run_dir / "net_save"))
        return epochs
    epochs = available_epochs(
        src_run_dir,
        "fwd_res",
        "net0_",
        args.epoch_start,
        args.epoch_end,
        args.epoch_interv,
        allow_roi=bool(args.training_only),
    )
    if not epochs:
        raise FileNotFoundError(
            "No complete fwd_res/net0_*.dat found in %s. Run pred_detector.py first or pass --run-pred." % src_run_dir
        )
    return epochs


def update_latest_alias(src: Path, alias: Path):
    alias.unlink(missing_ok=True)
    try:
        os.symlink(src.name, alias)
    except OSError:
        pass


def catalog_path(out_dir: Path, load_epoch: int):
    return out_dir / ("catalog_sdc1_%04d.txt" % load_epoch)


def pred_obb_path(out_dir: Path, load_epoch: int):
    return out_dir / ("pred_obb_%04d.csv" % load_epoch)


def update_score_history(out_dir: Path, load_epoch: int, score: float):
    rows = {}
    history_path = out_dir / "score_history.txt"
    if history_path.is_file():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows[int(float(parts[0]))] = float(parts[1])
            except ValueError:
                continue

    for path in out_dir.glob("score_epoch_*.txt"):
        epoch_text = path.stem.split("_")[-1]
        if not epoch_text.isdigit():
            continue
        try:
            epoch = int(epoch_text)
        except ValueError:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("score:"):
                try:
                    rows[epoch] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
                break

    rows[int(load_epoch)] = float(score)
    with history_path.open("w", encoding="utf-8") as f:
        for epoch in sorted(rows):
            f.write("%d %.10g\n" % (epoch, rows[epoch]))


def pred_obb_header() -> str:
    return (
        "det_id,ra_deg,dec_deg,cx_pix,cy_pix,obb_w_pix,obb_h_pix,theta_le90_deg,"
        "hbb_xmin,hbb_ymin,hbb_xmax,hbb_ymax,objectness,probability,prior_id,"
        "flux_jy,phys_bmaj_arcsec,phys_bmin_arcsec,phys_pa_deg,"
        "obb_bmaj_arcsec,obb_bmin_arcsec,aspect_ratio,"
        "c0_x,c0_y,c1_x,c1_y,c2_x,c2_y,c3_x,c3_y,"
        + ",".join(PRED_OBB_ORIGIN_COLUMNS)
    )


def fwd_flat_offsets(tile_y, tile_x, channel, cell_y, cell_x) -> np.ndarray:
    tile_y = np.asarray(tile_y, dtype=np.int64)
    tile_x = np.asarray(tile_x, dtype=np.int64)
    channel = np.asarray(channel, dtype=np.int64)
    cell_y = np.asarray(cell_y, dtype=np.int64)
    cell_x = np.asarray(cell_x, dtype=np.int64)
    out = np.full(tile_y.shape[0], -1, dtype=np.int64)
    valid = (
        (tile_y >= 0)
        & (tile_y < nb_area_h)
        & (tile_x >= 0)
        & (tile_x < nb_area_w)
        & (cell_y >= 0)
        & (cell_y < yolo_nb_reg)
        & (cell_x >= 0)
        & (cell_x < yolo_nb_reg)
        & (channel >= 0)
        & (channel < nb_box * (8 + CURRENT_TOTAL_AUX))
    )
    out[valid] = (
        (
            (
                (tile_y[valid] * nb_area_w + tile_x[valid])
                * (nb_box * (8 + CURRENT_TOTAL_AUX))
                + channel[valid]
            )
            * yolo_nb_reg
            + cell_y[valid]
        )
        * yolo_nb_reg
        + cell_x[valid]
    )
    return out


def candidate_origin_arrays(flat_kept_scaled: np.ndarray) -> dict[str, np.ndarray]:
    n_rows = flat_kept_scaled.shape[0]
    origin_start = 7 + CURRENT_TOTAL_AUX + 1
    origin_width = len(CANDIDATE_ORIGIN_COLUMNS)
    origin = np.full((n_rows, origin_width), -1, dtype=np.int64)
    if flat_kept_scaled.shape[1] >= origin_start + origin_width:
        origin = np.rint(flat_kept_scaled[:, origin_start : origin_start + origin_width]).astype(np.int64)

    out = {name: origin[:, idx] for idx, name in enumerate(CANDIDATE_ORIGIN_COLUMNS)}
    prior_id = np.rint(flat_kept_scaled[:, 6]).astype(np.int64)
    channels = 8 + CURRENT_TOTAL_AUX
    base_channel = prior_id * channels
    out["fwd_base_offset"] = fwd_flat_offsets(
        out["tile_y"],
        out["tile_x"],
        base_channel,
        out["cell_y"],
        out["cell_x"],
    )

    delta_param_index = 3 if CURRENT_NB_PARAM > 3 else -1
    out["fwd_delta_param_index"] = np.full(n_rows, delta_param_index, dtype=np.int64)
    delta_channel = np.full(n_rows, -1, dtype=np.int64)
    if delta_param_index >= 0:
        delta_channel = prior_id * channels + 8 + delta_param_index
    out["fwd_delta_channel"] = delta_channel
    out["fwd_delta_offset"] = fwd_flat_offsets(
        out["tile_y"],
        out["tile_x"],
        delta_channel,
        out["cell_y"],
        out["cell_x"],
    )
    return out


def write_pred_obb_csv(path: Path, obb_info, ra, dec, flat_kept_scaled):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = pred_obb_header()
    obb = obb_info["obb"]
    hbb = obb_info["hbb"]
    corners = obb_info["corners"]
    origin = candidate_origin_arrays(flat_kept_scaled)
    rows = np.column_stack(
        [
            np.arange(obb.shape[0], dtype=np.float64),
            ra,
            dec,
            obb[:, 0],
            obb[:, 1],
            obb[:, 2],
            obb[:, 3],
            obb[:, 4],
            hbb[:, 0],
            hbb[:, 1],
            hbb[:, 2],
            hbb[:, 3],
            flat_kept_scaled[:, 5],
            flat_kept_scaled[:, 4],
            flat_kept_scaled[:, 6],
            obb_info["flux_jy"],
            obb_info["bmaj_arcsec"],
            obb_info["bmin_arcsec"],
            obb_info["pa_deg"],
            obb_info["obb_bmaj_arcsec"],
            obb_info["obb_bmin_arcsec"],
            obb_info["aspect_ratio"],
            corners[:, 0, 0],
            corners[:, 0, 1],
            corners[:, 1, 0],
            corners[:, 1, 1],
            corners[:, 2, 0],
            corners[:, 2, 1],
            corners[:, 3, 0],
            corners[:, 3, 1],
            origin["tile_y"],
            origin["tile_x"],
            origin["cell_y"],
            origin["cell_x"],
            origin["grid_index"],
            origin["local_candidate_id"],
            origin["fwd_base_offset"],
            origin["fwd_delta_param_index"],
            origin["fwd_delta_channel"],
            origin["fwd_delta_offset"],
        ]
    )
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.10g")


def write_empty_outputs(out_dir: Path, load_epoch: int):
    catalog = catalog_path(out_dir, load_epoch)
    pred_obb = pred_obb_path(out_dir, load_epoch)
    score_epoch_path = out_dir / ("score_epoch_%04d.txt" % load_epoch)

    catalog.write_text("", encoding="utf-8")
    pred_obb.write_text(pred_obb_header() + "\n", encoding="utf-8")
    score_epoch_path.write_text(
        "epoch: %d\nscore: 0.0000000000\ncatalog: %s\nobb_catalog: %s\n"
        % (load_epoch, catalog, pred_obb),
        encoding="utf-8",
    )
    update_score_history(out_dir, load_epoch, 0.0)
    update_latest_alias(catalog, out_dir / "catalog_sdc1.txt")
    update_latest_alias(pred_obb, out_dir / "pred_obb.csv")


def build_candidate_cache(
    predict_getter,
    lims,
    full_data_norm,
    cache_min_obj: float,
    base_row_width: int,
    out_range: int,
    decode_rows_obb_fn,
    slim_mode: str,
    tile_mask=None,
):
    cache_rows_grid = np.empty((nb_area_h, nb_area_w), dtype="object")
    cache_boxes_grid = np.empty((nb_area_h, nb_area_w), dtype="object")
    cache_prob_obj_cases = np.full((nb_box,), float(cache_min_obj), dtype=np.float64)

    max_candidates = yolo_nb_reg * yolo_nb_reg * nb_box
    c_tile = np.zeros((max_candidates, base_row_width), dtype="float32")
    c_box = np.zeros((base_row_width,), dtype="float32")
    patch = np.zeros((fwd_image_size, fwd_image_size), dtype="float32")
    box_count_per_reg_hist = np.zeros((nb_box + 1), dtype="int")
    total_candidates = 0

    for ph in tqdm(range(nb_area_h), desc="Caching pre-NMS candidates"):
        row_line = []
        box_line = []
        for pw in range(nb_area_w):
            xmin = pw * patch_shift - orig_offset
            xmax = pw * patch_shift + fwd_image_size - orig_offset
            ymin = ph * patch_shift - orig_offset
            ymax = ph * patch_shift + fwd_image_size - orig_offset

            if tile_mask is not None and not tile_mask[ph, pw]:
                row_line.append(np.zeros((0, base_row_width + len(CANDIDATE_ORIGIN_COLUMNS)), dtype=np.float64))
                box_line.append(np.zeros((0, 5), dtype=np.float64))
                continue

            if ph < out_range or ph >= nb_area_h - out_range or pw < out_range or pw >= nb_area_w - out_range:
                row_line.append(np.zeros((0, base_row_width + len(CANDIDATE_ORIGIN_COLUMNS)), dtype=np.float64))
                box_line.append(np.zeros((0, 5), dtype=np.float64))
                continue

            c_tile[:, :] = 0.0
            c_box[:] = 0.0
            patch[:, :] = full_data_norm[ymin:ymax, xmin:xmax]
            c_pred = predict_getter(ph, pw)
            c_nb_box = tile_filter_candidates(
                c_pred,
                c_box,
                c_tile,
                nb_box,
                CURRENT_TOTAL_AUX,
                cache_prob_obj_cases,
                patch,
                val_med_lims,
                val_med_obj,
                box_count_per_reg_hist,
            )
            if c_nb_box == 0:
                row_line.append(np.zeros((0, base_row_width + len(CANDIDATE_ORIGIN_COLUMNS)), dtype=np.float64))
                box_line.append(np.zeros((0, 5), dtype=np.float64))
                continue

            rows = c_tile[:c_nb_box].astype(np.float32, copy=True)
            grid_index = np.rint(rows[:, base_row_width - 1]).astype(np.int64)
            cell_y = grid_index // yolo_nb_reg
            cell_x = grid_index % yolo_nb_reg
            origin = np.column_stack(
                [
                    np.full(c_nb_box, ph, dtype=np.float32),
                    np.full(c_nb_box, pw, dtype=np.float32),
                    cell_y.astype(np.float32),
                    cell_x.astype(np.float32),
                    grid_index.astype(np.float32),
                    np.arange(c_nb_box, dtype=np.float32),
                ]
            )
            rows = np.concatenate((rows, origin), axis=1)
            boxes = decode_rows_obb_fn(rows, lims, slim_mode)
            row_line.append(rows)
            box_line.append(boxes)
            total_candidates += c_nb_box

        cache_rows_grid[ph] = row_line
        cache_boxes_grid[ph] = box_line

    return cache_rows_grid, cache_boxes_grid, total_candidates


def load_prediction_tensor(path: Path, *, layout, training_only: bool):
    full_expected = expected_fwd_floats()
    marker = roi_marker_path(path)
    pred_data = np.fromfile(path, dtype="float32")
    channels = nb_box * (8 + layout.total_aux)

    if pred_data.size == full_expected:
        predict = np.reshape(pred_data, (nb_area_h, nb_area_w, channels, yolo_nb_reg, yolo_nb_reg))

        def get_full(ph, pw):
            return predict[ph, pw, :, :, :]

        return get_full, "full", None

    if marker.is_file():
        if not training_only:
            raise ValueError(
                "%s is a compact training-region ROI fwd_res. Re-run pred_detector.py without "
                "--training-roi-only for full-image post-processing." % path
            )
        meta = np.load(marker)
        tile_indices = np.asarray(meta["tile_indices"], dtype=np.int64)
        marker_channels = int(np.asarray(meta["channels"]).reshape(-1)[0])
        marker_yolo_nb_reg = int(np.asarray(meta["yolo_nb_reg"]).reshape(-1)[0])
        marker_nb_area_h = int(np.asarray(meta["nb_area_h"]).reshape(-1)[0])
        marker_nb_area_w = int(np.asarray(meta["nb_area_w"]).reshape(-1)[0])
        if marker_nb_area_h != nb_area_h or marker_nb_area_w != nb_area_w:
            raise ValueError("%s ROI marker has incompatible tile grid." % marker)
        if marker_channels != channels or marker_yolo_nb_reg != yolo_nb_reg:
            raise ValueError("%s ROI marker has incompatible output layout." % marker)
        roi_expected = int(tile_indices.size * channels * yolo_nb_reg * yolo_nb_reg)
        if pred_data.size != roi_expected:
            raise ValueError(
                "%s has %d ROI float32 values, expected %d."
                % (path, pred_data.size, roi_expected)
            )
        compact = np.reshape(pred_data, (int(tile_indices.size), channels, yolo_nb_reg, yolo_nb_reg))
        flat_to_compact = {int(tile): idx for idx, tile in enumerate(tile_indices)}
        zero = np.zeros((channels, yolo_nb_reg, yolo_nb_reg), dtype=np.float32)

        def get_roi(ph, pw):
            idx = flat_to_compact.get(int(ph) * nb_area_w + int(pw))
            if idx is None:
                return zero
            return compact[idx, :, :, :]

        return get_roi, "training_roi", tile_indices

    raise ValueError(
        "%s has %d float32 values, expected %d for full fwd_res. This looks like a partial or corrupt fwd_res file."
        % (path, pred_data.size, full_expected)
    )


def training_tile_mask(*, halo_tiles: int = 2) -> np.ndarray:
    """Tiles needed to reproduce training-region post-NMS with local context."""
    mask = np.zeros((nb_area_h, nb_area_w), dtype=bool)
    train_xmin = 16383.0
    train_xmax = 19853.0
    train_ymin = 16730.0
    train_ymax = 19921.0
    margin = float(max(0, halo_tiles) * patch_shift + fwd_image_size)
    for ph in range(nb_area_h):
        tile_ymin = ph * patch_shift - orig_offset
        tile_ymax = tile_ymin + fwd_image_size
        for pw in range(nb_area_w):
            tile_xmin = pw * patch_shift - orig_offset
            tile_xmax = tile_xmin + fwd_image_size
            intersects = (
                tile_xmax >= train_xmin - margin
                and tile_xmin <= train_xmax + margin
                and tile_ymax >= train_ymin - margin
                and tile_ymin <= train_ymax + margin
            )
            if intersects:
                mask[ph, pw] = True
    return mask


def main():
    from orsdet_nms import (
        FIRST_IOU_THRESHOLDS,
        FIRST_OBJ_THRESHOLDS,
        SECOND_IOU_THRESHOLD,
        local_nms,
        merge_nms,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("epoch", nargs="?", type=int)
    parser.add_argument("--src-run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--slim-mode",
        default=None,
        choices=(
            "size",
            "shared_angle",
            "size-angle",
            "size_angle",
            "flux_refine",
            "shared_angle_flux_refine",
            "native_flux_head",
            "flux_calib_gate",
            "shared_angle_flux_calib_gate",
        ),
        help="detector branch. Defaults to run_info.txt slim_mode, then size.",
    )
    parser.add_argument(
        "--flux-decode-mode",
        choices=(
            "base",
            "flux_base",
            "final",
            "final_gate1",
            "gate1",
            "final_noclip",
            "final_gate1_noclip",
            "gate1_noclip",
            "native_delta",
            "bright_gate",
            "final_bright_gate",
            "bright_protected",
            "learned_gate",
            "final_learned_gate",
            "calib_gate",
        ),
        default="base",
    )
    parser.add_argument("--flux-delta-norm-scale", type=float, default=0.25)
    parser.add_argument("--run-pred", action="store_true")
    parser.add_argument("--training-only", action="store_true", help="Write and score only detections inside the SDC1 training region.")
    parser.add_argument("--latest", action="store_true", help="Only post-process the latest available epoch.")
    parser.add_argument("--epoch-start", type=int)
    parser.add_argument("--epoch-end", type=int)
    parser.add_argument("--epoch-interv", type=int, default=1)
    parser.add_argument("--opt-rounds", type=int, default=4)
    parser.add_argument(
        "--no-early-stop",
        dest="early_stop_on_threshold_convergence",
        action="store_false",
        help="Disable early stop when an opt round computes the same thresholds as the current round.",
    )
    parser.add_argument(
        "--cache-min-objectness",
        type=float,
        default=float(np.logspace(-1.5, 0, num=60)[0]),
        help="Lower bound used when caching pre-NMS candidates. Results stay exact as long as later rounds do not use a lower threshold.",
    )
    pred_obb_group = parser.add_mutually_exclusive_group()
    pred_obb_group.add_argument(
        "--skip-intermediate-pred-obb",
        dest="skip_intermediate_pred_obb",
        action="store_true",
        help="Default behavior: only write pred_obb CSV on the final opt round. Catalog files are still written every round for scoring.",
    )
    pred_obb_group.add_argument(
        "--write-intermediate-pred-obb",
        dest="skip_intermediate_pred_obb",
        action="store_false",
        help="Write pred_obb CSV on every opt round instead of only the final one.",
    )
    parser.set_defaults(skip_intermediate_pred_obb=True, early_stop_on_threshold_convergence=True)
    args = parser.parse_args()

    configure_detector_paths()
    install_numba_fallback_if_needed()

    import data_gen as dg
    from ska_sdc import Sdc1Scorer

    src_run_dir = args.src_run_dir.resolve()
    out_dir = args.out_dir.resolve()
    slim_mode = normalize_slim_mode(args.slim_mode or read_run_slim_mode(src_run_dir) or "size")
    layout = detector_layout(slim_mode)
    global CURRENT_NB_PARAM, CURRENT_TOTAL_AUX
    CURRENT_TOTAL_AUX = layout.total_aux
    CURRENT_NB_PARAM = layout.nb_param
    src_run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fwd_res").mkdir(exist_ok=True)
    print("Using source run directory:", src_run_dir)
    print("Using output directory:", out_dir)
    print("Using detector slim mode:", layout.mode)
    print("Using flux decode mode:", args.flux_decode_mode)

    os.chdir(src_run_dir)
    af = importlib.import_module("aux_fct")
    globals().update({name: getattr(af, name) for name in dir(af) if not name.startswith("_")})

    lims = np.loadtxt(src_run_dir / "train_norm.txt")
    epochs = select_epochs(src_run_dir, args)
    training_only = bool(args.training_only)
    print("Training-region output only:", training_only)
    full_data_norm = np.clip(full_img, min_pix, max_pix)
    full_data_norm = (full_data_norm - min_pix) / (max_pix - min_pix)
    full_data_norm = np.tanh(3.0 * full_data_norm)
    initial_prob_obj_cases = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1], dtype=np.float64)
    print("Initial prob_obj_cases:", " ".join("%.4f" % x for x in initial_prob_obj_cases))
    out_range = 2
    base_row_width = 6 + 1 + layout.total_aux + 1
    cached_row_width = base_row_width + len(CANDIDATE_ORIGIN_COLUMNS)
    cache_floor_default = float(args.cache_min_objectness)
    candidate_rows_grid = None
    candidate_boxes_grid = None
    candidate_cache_floor = None
    candidate_cache_total = 0
    roi_tile_mask = training_tile_mask(halo_tiles=2) if training_only else None
    if roi_tile_mask is not None:
        print("Training-region tile ROI:", int(roi_tile_mask.sum()), "/", int(nb_area_h * nb_area_w), "tiles")

    def ensure_candidate_cache(predict_getter, min_obj_threshold: float):
        nonlocal candidate_rows_grid, candidate_boxes_grid, candidate_cache_floor, candidate_cache_total
        target_floor = float(min_obj_threshold)
        if candidate_rows_grid is not None and target_floor >= candidate_cache_floor - 1e-12:
            return

        build_floor = min(cache_floor_default, target_floor) if candidate_rows_grid is None else target_floor
        candidate_rows_grid, candidate_boxes_grid, candidate_cache_total = build_candidate_cache(
            predict_getter,
            lims,
            full_data_norm,
            build_floor,
            base_row_width,
            out_range,
            decode_rows_obb,
            layout.mode,
            tile_mask=roi_tile_mask,
        )
        candidate_cache_floor = build_floor
        print(
            "Cached %d pre-NMS candidates at objectness floor %.6f"
            % (candidate_cache_total, candidate_cache_floor)
        )

    def run_round(candidate_rows_grid, candidate_boxes_grid, prob_obj_cases, load_epoch, write_pred_obb: bool = True):
        final_rows = []
        final_boxes = []
        c_tile = np.zeros((yolo_nb_reg * yolo_nb_reg * nb_box, cached_row_width), dtype="float32")
        c_tile_kept = np.zeros((yolo_nb_reg * yolo_nb_reg * nb_box, cached_row_width), dtype="float32")
        c_box = np.zeros((cached_row_width,), dtype="float32")
        patch = np.zeros((fwd_image_size, fwd_image_size), dtype="float32")
        box_count_per_reg_hist = np.zeros((nb_box + 1), dtype="int")

        for ph in tqdm(range(nb_area_h)):
            row_line = []
            box_line = []
            for pw in range(nb_area_w):
                if roi_tile_mask is not None and not roi_tile_mask[ph, pw]:
                    row_line.append(np.zeros((0, cached_row_width), dtype=np.float64))
                    box_line.append(np.zeros((0, 5), dtype=np.float64))
                    continue

                if ph < out_range or ph >= nb_area_h - out_range or pw < out_range or pw >= nb_area_w - out_range:
                    row_line.append(np.zeros((0, cached_row_width), dtype=np.float64))
                    box_line.append(np.zeros((0, 5), dtype=np.float64))
                    continue

                cached_rows = candidate_rows_grid[ph, pw]
                cached_boxes = candidate_boxes_grid[ph, pw]
                if cached_rows.shape[0] == 0:
                    row_line.append(np.zeros((0, cached_row_width), dtype=np.float64))
                    box_line.append(np.zeros((0, 5), dtype=np.float64))
                    continue

                prior_ids = np.rint(cached_rows[:, 6]).astype(np.int64)
                keep_mask = cached_rows[:, 5] >= prob_obj_cases[prior_ids]
                if not np.any(keep_mask):
                    row_line.append(np.zeros((0, cached_row_width), dtype=np.float64))
                    box_line.append(np.zeros((0, 5), dtype=np.float64))
                    continue

                selected_rows = cached_rows[keep_mask]
                c_nb_box = selected_rows.shape[0]
                c_tile[:, :] = 0.0
                c_tile_kept[:, :] = 0.0
                c_tile[:c_nb_box, :] = selected_rows
                c_nb_box_final = first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, first_nms_thresholds, first_nms_obj_thresholds)
                if c_nb_box_final == 0:
                    row_line.append(np.zeros((0, cached_row_width), dtype=np.float64))
                    box_line.append(np.zeros((0, 5), dtype=np.float64))
                    continue

                rows = c_tile_kept[:c_nb_box_final].astype(np.float64, copy=True)
                kept_ids = np.rint(rows[:, -1]).astype(np.int64)
                boxes = cached_boxes[kept_ids].astype(np.float64, copy=True)
                rows, boxes = local_nms(rows, boxes, FIRST_IOU_THRESHOLDS, FIRST_OBJ_THRESHOLDS)
                row_line.append(rows)
                box_line.append(boxes)

            final_rows.append(row_line)
            final_boxes.append(box_line)

        final_rows_grid = np.empty((nb_area_h, nb_area_w), dtype="object")
        final_boxes_grid = np.empty((nb_area_h, nb_area_w), dtype="object")
        for ph in range(nb_area_h):
            for pw in range(nb_area_w):
                final_rows_grid[ph, pw] = final_rows[ph][pw]
                final_boxes_grid[ph, pw] = final_boxes[ph][pw]

        dir_array = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1], [-1, +1], [+1, +1], [-1, -1], [+1, -1]])
        for ph in tqdm(range(nb_area_h)):
            for pw in range(nb_area_w):
                if roi_tile_mask is not None and not roi_tile_mask[ph, pw]:
                    continue
                rows = np.copy(final_rows_grid[ph, pw])
                boxes = np.copy(final_boxes_grid[ph, pw])
                for l in range(8):
                    nh = ph + dir_array[l, 1]
                    nw = pw + dir_array[l, 0]
                    if 0 <= nh <= nb_area_h - 1 and 0 <= nw <= nb_area_w - 1:
                        comp_rows = np.copy(final_rows_grid[nh, nw])
                        comp_boxes = np.copy(final_boxes_grid[nh, nw])
                        rows, boxes = merge_nms(rows, boxes, comp_rows, comp_boxes, dir_array[l], SECOND_IOU_THRESHOLD, patch_shift, overlap)
                final_rows_grid[ph, pw] = np.copy(rows)
                final_boxes_grid[ph, pw] = np.copy(boxes)

        empty_rows = np.zeros((0, cached_row_width), dtype=np.float64)
        empty_boxes = np.zeros((0, 5), dtype=np.float64)
        for pw in range(nb_area_w):
            final_rows_grid[0, pw] = np.copy(empty_rows)
            final_rows_grid[nb_area_h - 1, pw] = np.copy(empty_rows)
            final_boxes_grid[0, pw] = np.copy(empty_boxes)
            final_boxes_grid[nb_area_h - 1, pw] = np.copy(empty_boxes)
        for ph in range(nb_area_h):
            final_rows_grid[ph, 0] = np.copy(empty_rows)
            final_rows_grid[ph, nb_area_w - 1] = np.copy(empty_rows)
            final_boxes_grid[ph, 0] = np.copy(empty_boxes)
            final_boxes_grid[ph, nb_area_w - 1] = np.copy(empty_boxes)

        flat_rows = []
        flat_boxes = []
        for ph in range(nb_area_h):
            box_h_offset = ph * patch_shift - orig_offset
            for pw in range(nb_area_w):
                box_w_offset = pw * patch_shift - orig_offset
                rows = np.copy(final_rows_grid[ph, pw])
                boxes = np.copy(final_boxes_grid[ph, pw])
                if rows.shape[0] > 0:
                    rows[:, 0] += box_w_offset
                    rows[:, 2] += box_w_offset
                    rows[:, 1] += box_h_offset
                    rows[:, 3] += box_h_offset
                    boxes[:, 0] += box_w_offset
                    boxes[:, 1] += box_h_offset
                flat_rows.append(rows)
                flat_boxes.append(boxes)

        if all(rows.shape[0] == 0 for rows in flat_rows):
            return {"empty": True}

        flat_kept_scaled = np.vstack(flat_rows)
        flat_kept_scaled = flat_kept_scaled[flat_kept_scaled[:, 5].argsort(), :][::-1]

        x_y_flat_kept = np.copy(flat_kept_scaled[:, 0:2])
        x_y_flat_kept[:, 0] = (flat_kept_scaled[:, 0] + flat_kept_scaled[:, 2]) * 0.5 - 0.5
        x_y_flat_kept[:, 1] = (flat_kept_scaled[:, 1] + flat_kept_scaled[:, 3]) * 0.5 - 0.5

        if training_only:
            training_area_id = np.where(
                (x_y_flat_kept[:, 0] > 16383)
                & (x_y_flat_kept[:, 0] < 19853)
                & (x_y_flat_kept[:, 1] > 16730)
                & (x_y_flat_kept[:, 1] < 19921)
            )[0]
            flat_kept_scaled = flat_kept_scaled[training_area_id]
            x_y_flat_kept = x_y_flat_kept[training_area_id]

        cls = utils.pixel_to_skycoord(x_y_flat_kept[:, 0], x_y_flat_kept[:, 1], wcs_img)
        ra_dec_coords = np.array([cls.ra.deg, cls.dec.deg])
        obb_info = detector_catalog_arrays(
            flat_kept_scaled,
            lims,
            pixel_size,
            layout.mode,
            args.flux_decode_mode,
            args.flux_delta_norm_scale,
        )

        catalog_size = np.shape(flat_kept_scaled)[0]
        box_catalog = np.zeros((catalog_size, 10), dtype="float32")
        box_catalog[:, [0, 1]] = ra_dec_coords.T
        box_catalog[:, [2, 3]] = obb_info["obb"][:, 2:4]
        box_catalog[:, 4] = flat_kept_scaled[:, 4]
        box_catalog[:, 5] = flat_kept_scaled[:, 5]
        box_catalog[:, 6] = obb_info["flux_jy"]
        box_catalog[:, 7] = obb_info["bmaj_arcsec"]
        box_catalog[:, 8] = obb_info["bmin_arcsec"]
        box_catalog[:, 9] = obb_info["pa_deg"]

        coords = SkyCoord(box_catalog[:, 0] * u.deg, box_catalog[:, 1] * u.deg)
        index = np.where(coords.ra.deg > 90.0)
        ra = coords.ra.deg
        dec = coords.dec.deg
        ra[index[0]] -= 360.0

        index_train = np.where((ra[:] < -0.0) & (ra[:] > -0.6723) & (dec[:] < -29.4061) & (dec[:] > -29.9400))[0]
        xbeam, ybeam = utils.skycoord_to_pixel(coords, wcs_beam)
        new_data_beam = np.nan_to_num(data_beam)
        beamval = interpn(
            (np.arange(0, np.shape(data_beam)[0]), np.arange(0, np.shape(data_beam)[1])),
            new_data_beam,
            (ybeam, xbeam),
            method="splinef2d",
        )
        flux_jy = box_catalog[:, 6] / beamval

        empty = np.zeros((np.shape(coords.ra.deg)[0]))
        scoring_table = np.vstack(
            (
                np.arange(0, np.shape(box_catalog)[0]),
                ra,
                dec,
                ra,
                dec,
                flux_jy,
                empty + 0.0375,
                box_catalog[:, 7],
                box_catalog[:, 8],
                box_catalog[:, 9],
                empty + 2.0,
                empty + 3.0,
            )
        )

        c_path = catalog_path(out_dir, load_epoch)
        o_path = pred_obb_path(out_dir, load_epoch)
        np.savetxt(c_path, scoring_table.T, fmt="%d %1.8f %2.8f %1.8f %2.8f %g %0.8f %f %f %f %d %d")
        if write_pred_obb:
            write_pred_obb_csv(o_path, obb_info, ra, dec, flat_kept_scaled)
            update_latest_alias(o_path, out_dir / "pred_obb.csv")
        update_latest_alias(c_path, out_dir / "catalog_sdc1.txt")

        scorer = Sdc1Scorer.from_txt(str(c_path), TRUTH_CATALOG_PATH, freq=560, sub_skiprows=0, truth_skiprows=0)
        scorer.run(mode=0, train=training_only, detail=True)

        return {
            "empty": False,
            "score": float(scorer.score.value),
            "scorer": scorer,
            "flat_kept_scaled": flat_kept_scaled,
            "box_catalog": box_catalog,
            "obb_info": obb_info,
            "ra": ra,
            "dec": dec,
            "index_train": index_train,
        }

    def update_prob_thresholds(scorer, flat_kept_scaled, box_catalog, index_train, search_idx):
        matched = scorer.score.match_df
        id_match = matched.id[:]
        match_array = np.zeros((np.shape(box_catalog)[0]))
        match_array[id_match] = 1

        scores_df = scorer.score.scores_df
        score_array = np.zeros((np.shape(box_catalog)[0]))
        score_array[id_match] = scores_df.to_numpy()[:, 1:].sum(axis=1) / 7.0
        box_id = flat_kept_scaled[:, 6]

        test_catalog = np.delete(box_catalog, index_train, axis=0)
        test_match_array = np.delete(match_array, index_train, axis=0)
        test_score_array = np.delete(score_array, index_train, axis=0)
        test_box_id = np.delete(box_id, index_train, axis=0)

        opt_sampling = 60
        bins = np.logspace(-1.5, 0, num=opt_sampling)
        dig_index = np.digitize(test_catalog[:, 5], bins=bins, right=True)
        opt_array = np.zeros((nb_box, opt_sampling, 4))
        opt_thresholds = np.zeros((nb_box))

        for k in range(nb_box):
            for i in range(opt_sampling - 1):
                bin_object_id = np.where((dig_index[:] == i) & (test_box_id[:] == k))[0]
                nb_tot_bin = int(np.shape(bin_object_id)[0])
                if nb_tot_bin == 0:
                    continue
                nb_match = np.sum(test_match_array[bin_object_id])
                avg_score = 0.0
                l_purity = 0.0
                if nb_match > 0:
                    avg_score = np.sum(test_score_array[bin_object_id] * test_match_array[bin_object_id]) / nb_match
                if nb_tot_bin > 0:
                    l_purity = nb_match / nb_tot_bin
                add_score = np.sum(test_score_array[bin_object_id] * test_match_array[bin_object_id]) - (nb_tot_bin - nb_match)
                opt_array[k, i, :] = np.array([nb_tot_bin, l_purity, avg_score, add_score])

            for i in range(opt_sampling - 1):
                if opt_array[k, i, 1] <= 0.630:
                    opt_array[k, i, 3] = 0.0

            id_opt = opt_sampling - 1
            for i in range(opt_sampling - 1):
                if np.all(np.cumsum(opt_array[k, i:, 3]) > 0) and opt_array[k, i, 0] >= 10:
                    id_opt = i
                    break
            opt_thresholds[k] = bins[id_opt - 2] if search_idx < 1 else bins[id_opt - 1]

        return opt_thresholds

    for epoch_idx, load_epoch in enumerate(epochs):
        if args.run_pred:
            subprocess.check_call(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "pred_detector.py"),
                    str(load_epoch),
                    "--run-dir",
                    str(src_run_dir),
                    "--slim-mode",
                    layout.mode,
                ]
            )

        fwd_path = src_run_dir / "fwd_res" / ("net0_%04d.dat" % load_epoch)
        if not fwd_path.is_file():
            raise FileNotFoundError(fwd_path)

        predict_getter, predict_kind, roi_tile_indices = load_prediction_tensor(
            fwd_path,
            layout=layout,
            training_only=training_only,
        )
        if predict_kind == "training_roi":
            print(
                "Prediction tensor kind: training_roi compact (%d / %d tiles)"
                % (int(roi_tile_indices.size), int(nb_area_h * nb_area_w))
            )
        else:
            print("Prediction tensor kind: full")
        score = 0.0
        scorer = None
        candidate_rows_grid = None
        candidate_boxes_grid = None
        candidate_cache_floor = None
        candidate_cache_total = 0

        ensure_candidate_cache(predict_getter, float(np.min(initial_prob_obj_cases)))

        max_opt_rounds = 1 if training_only else max(1, args.opt_rounds)

        for opt_search in range(max_opt_rounds):
            prob_obj_cases = initial_prob_obj_cases if opt_search == 0 else opt_thresholds
            current_prob_obj_cases = np.asarray(prob_obj_cases, dtype=np.float64).copy()
            write_pred_obb = (not args.skip_intermediate_pred_obb) or (opt_search == max_opt_rounds - 1)
            ensure_candidate_cache(predict_getter, float(np.min(prob_obj_cases)))
            result = run_round(candidate_rows_grid, candidate_boxes_grid, prob_obj_cases, load_epoch, write_pred_obb=write_pred_obb)
            if result["empty"]:
                print("Warning: no detections kept after rotated NMS NMS for epoch %d" % load_epoch)
                write_empty_outputs(out_dir, load_epoch)
                score = 0.0
                scorer = None
                break

            scorer = result["scorer"]
            score = result["score"]
            if opt_search < max_opt_rounds - 1:
                next_opt_thresholds = update_prob_thresholds(
                    scorer,
                    result["flat_kept_scaled"],
                    result["box_catalog"],
                    result["index_train"],
                    opt_search,
                )
                print("Opt thresholds:", " ".join("%.4f" % x for x in next_opt_thresholds))
                thresholds_converged = (
                    args.early_stop_on_threshold_convergence
                    and opt_search >= 1
                    and np.allclose(next_opt_thresholds, current_prob_obj_cases, rtol=0.0, atol=1e-12)
                )
                opt_thresholds = next_opt_thresholds
                if thresholds_converged:
                    if args.skip_intermediate_pred_obb and not write_pred_obb:
                        o_path = pred_obb_path(out_dir, load_epoch)
                        write_pred_obb_csv(
                            o_path,
                            result["obb_info"],
                            result["ra"],
                            result["dec"],
                            result["flat_kept_scaled"],
                        )
                        update_latest_alias(o_path, out_dir / "pred_obb.csv")
                    print(
                        "Opt thresholds converged at round %d/%d; skipping remaining opt rounds."
                        % (opt_search + 1, max_opt_rounds)
                    )
                    break

        if scorer is None:
            continue

        with (out_dir / ("score_epoch_%04d.txt" % load_epoch)).open("w", encoding="utf-8") as f:
            f.write("epoch: %d\n" % load_epoch)
            f.write("score: %.10f\n" % score)
            f.write("catalog: %s\n" % catalog_path(out_dir, load_epoch))
            f.write("obb_catalog: %s\n" % pred_obb_path(out_dir, load_epoch))
        update_score_history(out_dir, load_epoch, score)


if __name__ == "__main__":
    main()
