"""CIANNA HBB prediction post-processing for V5."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .score import (
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


INITIAL_THRESHOLDS = np.asarray([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05], dtype=np.float64)


def _safe_symlink(src: Path, alias: Path) -> None:
    alias.unlink(missing_ok=True)
    try:
        alias.symlink_to(src.name)
    except OSError:
        import shutil

        shutil.copy2(src, alias)


def expected_fwd_floats(aux) -> int:
    return int(aux.nb_area_h * aux.nb_area_w * aux.nb_box * (8 + aux.nb_param) * aux.yolo_nb_reg * aux.yolo_nb_reg)


def decode_catalog_from_boxes(flat_kept_scaled, aux, lims):
    x_y = np.copy(flat_kept_scaled[:, 0:2])
    x_y[:, 0] = (flat_kept_scaled[:, 0] + flat_kept_scaled[:, 2]) * 0.5 - 0.5
    x_y[:, 1] = (flat_kept_scaled[:, 1] + flat_kept_scaled[:, 3]) * 0.5 - 0.5

    cls = aux.utils.pixel_to_skycoord(x_y[:, 0], x_y[:, 1], aux.wcs_img)
    coords = aux.SkyCoord(cls.ra.deg * aux.u.deg, cls.dec.deg * aux.u.deg)
    ra = coords.ra.deg.copy()
    dec = coords.dec.deg.copy()
    ra[np.where(ra > 90.0)[0]] -= 360.0

    w = flat_kept_scaled[:, 2] - flat_kept_scaled[:, 0]
    h = flat_kept_scaled[:, 3] - flat_kept_scaled[:, 1]
    apparent_flux = np.exp(flat_kept_scaled[:, 7] * (lims[0, 0] - lims[0, 1]) + lims[0, 1])
    bmaj = np.exp(flat_kept_scaled[:, 8] * (lims[1, 0] - lims[1, 1]) + lims[1, 1])
    bmin = np.exp(flat_kept_scaled[:, 9] * (lims[2, 0] - lims[2, 1]) + lims[2, 1])
    pa = np.clip(
        np.arctan2(np.clip(flat_kept_scaled[:, 11], 0.0, 1.0) * 2.0 - 1.0, np.clip(flat_kept_scaled[:, 10], 0.0, 1.0))
        * 180.0
        / np.pi,
        -90.0,
        90.0,
    )

    xbeam, ybeam = aux.utils.skycoord_to_pixel(coords, aux.wcs_beam)
    beamval = aux.interpn(
        (np.arange(0, np.shape(aux.data_beam)[0]), np.arange(0, np.shape(aux.data_beam)[1])),
        np.nan_to_num(aux.data_beam),
        (ybeam, xbeam),
        method="splinef2d",
    )
    flux = apparent_flux / beamval

    return {
        "ra": ra,
        "dec": dec,
        "x": x_y[:, 0],
        "y": x_y[:, 1],
        "w": w,
        "h": h,
        "apparent_flux": apparent_flux,
        "flux": flux,
        "bmaj": bmaj,
        "bmin": bmin,
        "pa": pa,
    }


def write_catalog(path: Path, decoded) -> None:
    empty = np.zeros(decoded["ra"].shape[0], dtype=np.float64)
    table = np.vstack(
        (
            np.arange(decoded["ra"].shape[0]),
            decoded["ra"],
            decoded["dec"],
            decoded["ra"],
            decoded["dec"],
            decoded["flux"],
            empty + 0.0375,
            decoded["bmaj"],
            decoded["bmin"],
            decoded["pa"],
            empty + 2.0,
            empty + 3.0,
        )
    )
    np.savetxt(path, table.T, fmt="%d %1.8f %2.8f %1.8f %2.8f %g %0.8f %f %f %f %d %d")


def write_pred_csv(path: Path, flat_kept_scaled, decoded, matched_ids=None) -> None:
    n = flat_kept_scaled.shape[0]
    matched = np.zeros(n, dtype=np.int64)
    if matched_ids is not None:
        ids = np.asarray(matched_ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < n)]
        matched[ids] = 1
    aspect = decoded["bmaj"] / np.maximum(decoded["bmin"], 1.0e-30)
    rows = np.column_stack(
        [
            np.arange(n),
            decoded["ra"],
            decoded["dec"],
            decoded["x"],
            decoded["y"],
            flat_kept_scaled[:, 0],
            flat_kept_scaled[:, 1],
            flat_kept_scaled[:, 2],
            flat_kept_scaled[:, 3],
            decoded["w"],
            decoded["h"],
            flat_kept_scaled[:, 5],
            flat_kept_scaled[:, 4],
            flat_kept_scaled[:, 6],
            decoded["flux"],
            decoded["apparent_flux"],
            decoded["bmaj"],
            decoded["bmin"],
            decoded["pa"],
            aspect,
            matched,
        ]
    )
    header = (
        "det_id,ra_deg,dec_deg,cx_pix,cy_pix,hbb_xmin,hbb_ymin,hbb_xmax,hbb_ymax,"
        "hbb_w_pix,hbb_h_pix,objectness,probability,prior_id,flux_jy,apparent_flux_jy,"
        "bmaj_arcsec,bmin_arcsec,pa_deg,aspect_ratio,matched"
    )
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%.10g")


def compute_thresholds(scorer, flat_kept_scaled, decoded, score_type: str = "score", search_idx: int = 0):
    matched = scorer.score.match_df
    scores_df = scorer.score.scores_df
    id_match = matched.id[:] if matched is not None and len(matched) > 0 else []
    match_array = np.zeros(decoded["ra"].shape[0])
    match_array[id_match] = 1
    score_array = np.zeros(decoded["ra"].shape[0])
    if scores_df is not None and len(scores_df) > 0:
        score_array[id_match] = scores_df.to_numpy()[:, 1:].sum(axis=1) / 7.0

    index_train = np.where(
        (decoded["ra"] < -0.0)
        & (decoded["ra"] > -0.6723)
        & (decoded["dec"] < -29.4061)
        & (decoded["dec"] > -29.9400)
    )[0]
    test_objectness = np.delete(flat_kept_scaled[:, 5], index_train, axis=0)
    test_match_array = np.delete(match_array, index_train, axis=0)
    test_score_array = np.delete(score_array, index_train, axis=0)
    test_box_id = np.delete(flat_kept_scaled[:, 6], index_train, axis=0)

    opt_sampling = 60
    bins = np.logspace(-1.5, 0, num=opt_sampling) if score_type == "score" else np.linspace(0, 1, num=opt_sampling)
    dig_index = np.digitize(test_objectness, bins=bins, right=True)
    opt_array = np.zeros((9, opt_sampling, 4))
    opt_thresholds = np.zeros((9))

    for k in range(9):
        for i in range(opt_sampling - 1):
            bin_object_id = np.where((dig_index[:] == i) & (test_box_id[:] == k))[0]
            nb_tot_bin = int(np.shape(bin_object_id)[0])
            nb_match = np.sum(test_match_array[bin_object_id])
            avg_score = 0.0
            l_purity = 0.0
            if nb_match > 0:
                avg_score = np.sum(test_score_array[bin_object_id] * test_match_array[bin_object_id]) / nb_match
            if nb_tot_bin > 0:
                l_purity = nb_match / nb_tot_bin
            add_score = np.sum(test_score_array[bin_object_id] * test_match_array[bin_object_id]) - (nb_tot_bin - nb_match)
            opt_array[k, i, :] = np.array([nb_tot_bin, l_purity, avg_score, add_score])

        if score_type == "score":
            for i in range(opt_sampling - 1):
                if opt_array[k, i, 1] <= 0.630:
                    opt_array[k, i, 3] = 0.0
            id_opt = opt_sampling - 1
            for i in range(opt_sampling - 1):
                if np.all(np.cumsum(opt_array[k, i:, 3]) > 0) and opt_array[k, i, 0] >= 10:
                    id_opt = i
                    break
            opt_thresholds[k] = bins[id_opt - 2] if search_idx < 1 else bins[id_opt - 1]
        else:
            opt_thresholds[k] = INITIAL_THRESHOLDS[k]
    return opt_thresholds


def run_hbb_round(predict, prob_obj_cases, aux, lims):
    final_boxes = []
    c_tile = np.zeros((aux.yolo_nb_reg * aux.yolo_nb_reg * aux.nb_box, (6 + 1 + aux.nb_param + 1)), dtype="float32")
    c_tile_kept = np.zeros_like(c_tile)
    c_box = np.zeros((6 + 1 + aux.nb_param + 1), dtype="float32")
    patch = np.zeros((aux.fwd_image_size, aux.fwd_image_size), dtype="float32")
    hist = np.zeros((aux.nb_box + 1), dtype="int")

    full_data_norm = np.clip(aux.full_img, aux.min_pix, aux.max_pix)
    full_data_norm = (full_data_norm - aux.min_pix) / (aux.max_pix - aux.min_pix)
    full_data_norm = np.tanh(3.0 * full_data_norm)

    for ph in range(aux.nb_area_h):
        for pw in range(aux.nb_area_w):
            c_tile[:, :] = 0.0
            c_tile_kept[:, :] = 0.0
            xmin = pw * aux.patch_shift - aux.orig_offset
            xmax = pw * aux.patch_shift + aux.fwd_image_size - aux.orig_offset
            ymin = ph * aux.patch_shift - aux.orig_offset
            ymax = ph * aux.patch_shift + aux.fwd_image_size - aux.orig_offset
            if ph == 0 or ph == aux.nb_area_h - 1 or pw == 0 or pw == aux.nb_area_w - 1:
                patch[:, :] = 0.0
            else:
                patch[:, :] = full_data_norm[ymin:ymax, xmin:xmax]
            c_pred = predict[ph, pw, :, :, :]
            c_nb_box = aux.tile_filter(c_pred, c_box, c_tile, aux.nb_box, prob_obj_cases, patch, aux.val_med_lims, aux.val_med_obj, hist)
            c_nb_box_final = aux.first_NMS(c_tile, c_tile_kept, c_box, c_nb_box, aux.first_nms_thresholds, aux.first_nms_obj_thresholds)
            if ph < 2 or ph >= aux.nb_area_h - 2 or pw < 2 or pw >= aux.nb_area_w - 2:
                c_nb_box_final = 0
            final_boxes.append(np.copy(c_tile_kept[0:c_nb_box_final]))

    final_grid = np.empty((aux.nb_area_h, aux.nb_area_w), dtype="object")
    for idx, boxes in enumerate(final_boxes):
        final_grid.flat[idx] = boxes

    c_tile = np.zeros((aux.yolo_nb_reg * aux.yolo_nb_reg * aux.nb_box, (6 + 1 + aux.nb_param + 1)), dtype="float32")
    dir_array = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1], [-1, +1], [+1, +1], [-1, -1], [+1, -1]])
    for ph in range(aux.nb_area_h):
        for pw in range(aux.nb_area_w):
            boxes = np.copy(final_grid[ph, pw])
            for dxy in dir_array:
                nh, nw = ph + dxy[1], pw + dxy[0]
                if 0 <= nh <= aux.nb_area_h - 1 and 0 <= nw <= aux.nb_area_w - 1:
                    comp_boxes = np.copy(final_grid[nh, nw])
                    c_nb_box = aux.second_NMS_local(boxes, comp_boxes, c_tile, dxy, aux.second_nms_threshold)
                    boxes = np.copy(c_tile[0:c_nb_box, :])
            final_grid[ph, pw] = np.copy(boxes)

    empty = np.copy(c_tile[0:0, :])
    for pw in range(aux.nb_area_w):
        final_grid[0, pw] = np.copy(empty)
        final_grid[aux.nb_area_h - 1, pw] = np.copy(empty)
    for ph in range(aux.nb_area_h):
        final_grid[ph, 0] = np.copy(empty)
        final_grid[ph, aux.nb_area_w - 1] = np.copy(empty)

    scaled = np.copy(final_grid)
    for ph in range(aux.nb_area_h):
        yoff = ph * aux.patch_shift - aux.orig_offset
        for pw in range(aux.nb_area_w):
            xoff = pw * aux.patch_shift - aux.orig_offset
            scaled[ph, pw][:, 0] += xoff
            scaled[ph, pw][:, 2] += xoff
            scaled[ph, pw][:, 1] += yoff
            scaled[ph, pw][:, 3] += yoff

    if all(boxes.shape[0] == 0 for boxes in scaled.flat):
        return np.zeros((0, 6 + 1 + aux.nb_param + 1), dtype=np.float32)
    flat = np.vstack(scaled.flatten())
    return flat[flat[:, 5].argsort(), :][::-1]


def postprocess_epoch(epoch: int, run_dir: Path, out_dir: Path, aux, truth_path: str, opt_rounds: int, train_score: bool):
    fwd_path = run_dir / "fwd_res" / ("net0_%04d.dat" % epoch)
    pred_data = np.fromfile(fwd_path, dtype="float32")
    expected = expected_fwd_floats(aux)
    if pred_data.size != expected:
        raise ValueError("%s has %d float32 values, expected %d" % (fwd_path, pred_data.size, expected))
    predict = np.reshape(pred_data, (aux.nb_area_h, aux.nb_area_w, aux.nb_box * (8 + aux.nb_param), aux.yolo_nb_reg, aux.yolo_nb_reg))
    lims = np.loadtxt(run_dir / "train_cat_norm_lims.txt")
    prob_obj_cases = INITIAL_THRESHOLDS.copy()
    final_result = None

    for round_idx in range(max(1, opt_rounds)):
        flat = run_hbb_round(predict, prob_obj_cases, aux, lims)
        if flat.shape[0] == 0:
            raise RuntimeError("No detections kept for epoch %d" % epoch)
        decoded = decode_catalog_from_boxes(flat, aux, lims)
        catalog_path = out_dir / ("catalog_sdc1_%04d.txt" % epoch)
        pred_path = out_dir / ("pred_hbb_%04d.csv" % epoch)
        write_catalog(catalog_path, decoded)
        scorer = score_catalog(catalog_path, truth_path, train=train_score)
        if round_idx < max(1, opt_rounds) - 1:
            prob_obj_cases = compute_thresholds(scorer, flat, decoded, search_idx=round_idx)
            continue
        matched_ids = scorer.score.match_df.id[:] if scorer.score.match_df is not None and len(scorer.score.match_df) > 0 else []
        write_pred_csv(pred_path, flat, decoded, matched_ids=matched_ids)
        _safe_symlink(catalog_path, out_dir / "catalog_sdc1.txt")
        _safe_symlink(pred_path, out_dir / "pred_hbb.csv")
        score_path = out_dir / ("score_epoch_%04d.txt" % epoch)
        result = result_from_scorer(epoch, scorer, catalog_path, score_path)
        write_score_epoch(score_path, result)
        final_result = (result, scorer)
    if final_result is None:
        raise RuntimeError("Postprocess did not produce a result for epoch %d" % epoch)
    return final_result


def finish_outputs(out_dir: Path, results: list, best_scorer) -> None:
    score_results = [item[0] for item in results]
    write_score_history_csv(out_dir / "score_history.csv", score_results)
    best = write_score_summary(out_dir / "score_summary.txt", score_results)
    update_best_aliases(out_dir, best)

    scorer = best_scorer
    errors = make_matched_errors(scorer)
    save_structured_csv(out_dir / "matched_errors.csv", errors)
    grouped = grouped_error_table(errors)
    save_structured_csv(out_dir / "error_by_group.csv", grouped)
