"""OBB-driven target construction for V2.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orsdet_geometry.geometry import canonicalize_le90, hbb_from_obb, xywhtheta_to_corners
from orsdet_angle.angle_codec import encode_theta_le90
from orsdet_angle.angle_loss import AspectWeightConfig, angle_weight_from_aspect
from orsdet_angle.angle_weight_variants import SmallElongatedBoostConfig, boost_small_elongated_weights
from orsdet_angle.tables import DEFAULT_V1_TABLE, load_v1_rotated_table

from .decode import V25_NB_ANGLE, V25_NB_PARAM, V25_TARGET_STRIDE, v25_target_dim


EPS = 1.0e-8


@dataclass
class PatchMeta:
    source_id: np.ndarray
    obb: np.ndarray
    hbb: np.ndarray


def _normalize_log_values(values, max_min_row):
    values = np.asarray(values, dtype=np.float64)
    row = np.asarray(max_min_row, dtype=np.float64)
    denom = max(float(row[0] - row[1]), EPS)
    return (values - row[1]) / denom


def _apply_point_transform(points, image_size: int, rot90: int, flip_w: int, flip_h: int):
    points = np.asarray(points, dtype=np.float64).copy()
    if rot90 == -1:
        x_old = points[..., 0].copy()
        y_old = points[..., 1].copy()
        points[..., 0] = image_size - y_old
        points[..., 1] = x_old
    elif rot90 == 1:
        x_old = points[..., 0].copy()
        y_old = points[..., 1].copy()
        points[..., 0] = y_old
        points[..., 1] = image_size - x_old

    if flip_w:
        points[..., 0] = image_size - points[..., 0]
    if flip_h:
        points[..., 1] = image_size - points[..., 1]
    return points


def _transform_boxes(local_boxes, image_size: int, rot90: int, flip_w: int, flip_h: int):
    if local_boxes.shape[0] == 0:
        return local_boxes.copy()
    boxes = np.asarray(local_boxes, dtype=np.float64).copy()
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    theta = boxes[:, 4]

    if rot90 == -1:
        cx_new = image_size - cy
        cy_new = cx
        theta = theta + 90.0
    elif rot90 == 1:
        cx_new = cy
        cy_new = image_size - cx
        theta = theta + 90.0
    else:
        cx_new = cx
        cy_new = cy

    if flip_w:
        cx_new = image_size - cx_new
        theta = -theta
    if flip_h:
        cy_new = image_size - cy_new
        theta = -theta

    cx_new, cy_new, w, h, theta = canonicalize_le90(cx_new, cy_new, w, h, theta)
    return np.stack([cx_new, cy_new, w, h, theta], axis=-1)


def _obb_to_hbb_fast(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float64)
    theta = np.deg2rad(boxes[:, 4])
    c = np.abs(np.cos(theta))
    s = np.abs(np.sin(theta))
    half_w = 0.5 * boxes[:, 2]
    half_h = 0.5 * boxes[:, 3]
    dx = half_w * c + half_h * s
    dy = half_w * s + half_h * c
    return np.column_stack([boxes[:, 0] - dx, boxes[:, 1] - dy, boxes[:, 0] + dx, boxes[:, 1] + dy])


class V25DataBuilder:
    def __init__(
        self,
        dg,
        *,
        v1_table_path: Path = DEFAULT_V1_TABLE,
        weight_config: AspectWeightConfig | None = None,
        small_elongated_boost: SmallElongatedBoostConfig | None = None,
    ):
        self.dg = dg
        self.image_size = int(dg.image_size)
        self.max_objects = int(dg.max_nb_obj_per_image)
        self.weight_config = weight_config or AspectWeightConfig()
        self.small_elongated_boost = small_elongated_boost
        self.target_dim = v25_target_dim(self.max_objects)

        train_list = np.loadtxt(dg.training_selection_path())
        self.source_ids = train_list[:, 0].astype(np.int64)

        v1_table = load_v1_rotated_table(v1_table_path)
        row_map = {int(source_id): idx for idx, source_id in enumerate(v1_table["source_id"].astype(np.int64))}
        try:
            row_idx = np.asarray([row_map[int(source_id)] for source_id in self.source_ids], dtype=np.int64)
        except KeyError as exc:
            raise KeyError("Missing source id in V1 OBB table: %s" % (exc.args[0],)) from exc
        self.v1_table = v1_table[row_idx]

        self.boxes_global = np.column_stack(
            [
                self.v1_table["cx_pix"],
                self.v1_table["cy_pix"],
                self.v1_table["w_pix"],
                self.v1_table["h_pix"],
                self.v1_table["theta_le90_deg"],
            ]
        ).astype(np.float64)
        self.global_hbb = hbb_from_obb(self.boxes_global)

        self.flux_norm = np.asarray(dg.flux_list, dtype=np.float64)
        self.angle_weights = angle_weight_from_aspect(
            np.asarray(self.v1_table["aspect_ratio"], dtype=np.float64),
            self.weight_config,
        )
        small_angle = np.asarray(self.v1_table["bmaj_arcsec"], dtype=np.float64) <= float(dg.pa_res_lim)
        self.angle_weights[small_angle] = np.minimum(self.angle_weights[small_angle], self.weight_config.min_weight)
        self.small_elongated_boost_mask = np.zeros_like(self.angle_weights, dtype=bool)
        if small_elongated_boost is not None and small_elongated_boost.boost_factor > 1.0:
            sqrt_area = np.sqrt(
                np.maximum(
                    np.asarray(self.v1_table["w_pix"], dtype=np.float64)
                    * np.asarray(self.v1_table["h_pix"], dtype=np.float64),
                    0.0,
                )
            )
            self.angle_weights, self.small_elongated_boost_mask = boost_small_elongated_weights(
                self.angle_weights,
                np.asarray(self.v1_table["aspect_ratio"], dtype=np.float64),
                sqrt_area,
                small_elongated_boost,
            )

        self.norm_lims = self._build_norm_lims()
        width_log = np.log(np.maximum(np.asarray(self.v1_table["w_pix"], dtype=np.float64), EPS))
        height_log = np.log(np.maximum(np.asarray(self.v1_table["h_pix"], dtype=np.float64), EPS))
        self.width_norm = _normalize_log_values(width_log, self.norm_lims[1])
        self.height_norm = _normalize_log_values(height_log, self.norm_lims[2])

    def _build_norm_lims(self):
        flux_lims = np.asarray(self.dg.lims[0], dtype=np.float64)
        width_log = np.log(np.maximum(np.asarray(self.v1_table["w_pix"], dtype=np.float64), EPS))
        height_log = np.log(np.maximum(np.asarray(self.v1_table["h_pix"], dtype=np.float64), EPS))
        width_row = np.array([np.max(width_log), np.min(width_log)], dtype=np.float64)
        height_row = np.array([np.max(height_log), np.min(height_log)], dtype=np.float64)
        return np.vstack([flux_lims, width_row, height_row])

    def save_norm_lims(self, path: Path) -> None:
        np.savetxt(path, self.norm_lims)

    def _patch_indices(self, x0: float, y0: float):
        hbb = self.global_hbb
        mask = (
            (hbb[:, 0] >= x0)
            & (hbb[:, 1] >= y0)
            & (hbb[:, 2] < x0 + self.image_size)
            & (hbb[:, 3] < y0 + self.image_size)
        )
        return np.flatnonzero(mask)

    def _make_patch_target(self, x0: float, y0: float, rot90: int, flip_w: int, flip_h: int, return_meta: bool):
        target = np.zeros(self.target_dim, dtype=np.float32)
        idx = self._patch_indices(x0, y0)
        if idx.size == 0:
            if return_meta:
                meta = PatchMeta(
                    source_id=np.zeros((0,), dtype=np.int64),
                    obb=np.zeros((0, 5), dtype=np.float64),
                    hbb=np.zeros((0, 4), dtype=np.float64),
                )
                return target, meta
            return target, None

        local_boxes = self.boxes_global[idx].copy()
        local_boxes[:, 0] -= x0 - 0.5
        local_boxes[:, 1] -= y0 - 0.5
        local_boxes = _transform_boxes(local_boxes, self.image_size, rot90, flip_w, flip_h)
        hbb = _obb_to_hbb_fast(local_boxes)

        n_obj = min(self.max_objects, idx.size)
        target[0] = n_obj
        for out_idx, src_idx in enumerate(idx[:n_obj]):
            angle_vec = np.asarray(encode_theta_le90(local_boxes[out_idx, 4]), dtype=np.float64).reshape(-1)
            row = np.array(
                [
                    1.0,
                    hbb[out_idx, 0],
                    hbb[out_idx, 1],
                    0.0,
                    hbb[out_idx, 2],
                    hbb[out_idx, 3],
                    1.0,
                    self.flux_norm[src_idx],
                    self.width_norm[src_idx],
                    self.height_norm[src_idx],
                    angle_vec[0],
                    angle_vec[1],
                    self.angle_weights[src_idx],
                ],
                dtype=np.float32,
            )
            start = 1 + out_idx * V25_TARGET_STRIDE
            target[start : start + V25_TARGET_STRIDE] = row

        if return_meta:
            meta = PatchMeta(
                source_id=self.source_ids[idx[:n_obj]],
                obb=local_boxes[:n_obj].copy(),
                hbb=hbb[:n_obj].copy(),
            )
            return target, meta
        return target, None

    def create_train_batch(self, return_meta: bool = False):
        input_data = np.zeros((self.dg.nb_images_iter, self.image_size * self.image_size), dtype=np.float32)
        targets = np.zeros((self.dg.nb_images_iter, self.target_dim), dtype=np.float32)
        meta_list = [] if return_meta else None

        for i in range(self.dg.nb_images_iter):
            if np.random.rand() > self.dg.add_noise_prop:
                p_x = np.random.randint(0, self.dg.area_width - self.image_size)
                p_y = np.random.randint(0, self.dg.area_height - self.image_size)
                patch = np.copy(self.dg.norm_data[p_y : p_y + self.image_size, p_x : p_x + self.image_size])

                flip_w = 0
                flip_h = 0
                rot90 = 0
                rot_rand = np.random.random()
                if self.dg.rotate_flag and rot_rand < 0.33:
                    rot90 = -1
                    patch = np.rot90(patch, k=-1, axes=(0, 1))
                elif self.dg.rotate_flag and rot_rand < 0.66:
                    rot90 = 1
                    patch = np.rot90(patch, k=1, axes=(0, 1))

                if np.random.random() < self.dg.flip_hor:
                    flip_w = 1
                    patch = np.flip(patch, axis=1)
                if np.random.random() < self.dg.flip_vert:
                    flip_h = 1
                    patch = np.flip(patch, axis=0)

                input_data[i, :] = patch.flatten("C")
                targets[i, :], meta = self._make_patch_target(
                    self.dg.min_ra_train_pix + p_x,
                    self.dg.min_dec_train_pix + p_y,
                    rot90,
                    flip_w,
                    flip_h,
                    return_meta,
                )
            else:
                p_x = np.random.randint(0, self.dg.noise_size[0] - self.image_size)
                p_y = np.random.randint(0, self.dg.noise_size[1] - self.image_size)
                if np.random.rand() > 0.5:
                    patch = np.flip(
                        np.copy(self.dg.norm_data_noise_1[p_y : p_y + self.image_size, p_x : p_x + self.image_size]),
                        axis=0,
                    )
                else:
                    patch = np.flip(
                        np.copy(self.dg.norm_data_noise_2[p_y : p_y + self.image_size, p_x : p_x + self.image_size]),
                        axis=0,
                    )

                rot_rand = np.random.random()
                if self.dg.rotate_flag and rot_rand < 0.33:
                    patch = np.rot90(patch, k=-1, axes=(0, 1))
                elif self.dg.rotate_flag and rot_rand < 0.66:
                    patch = np.rot90(patch, k=1, axes=(0, 1))
                if np.random.random() < self.dg.flip_hor:
                    patch = np.flip(patch, axis=1)
                if np.random.random() < self.dg.flip_vert:
                    patch = np.flip(patch, axis=0)

                input_data[i, :] = patch.flatten("C")
                meta = PatchMeta(
                    source_id=np.zeros((0,), dtype=np.int64),
                    obb=np.zeros((0, 5), dtype=np.float64),
                    hbb=np.zeros((0, 4), dtype=np.float64),
                )

            if return_meta:
                meta_list.append(meta)

        if return_meta:
            return input_data, targets, meta_list
        return input_data, targets

    def create_valid_batch(self, return_meta: bool = False):
        input_valid = np.zeros((self.dg.nb_valid, self.image_size * self.image_size), dtype=np.float32)
        targets_valid = np.zeros((self.dg.nb_valid, self.target_dim), dtype=np.float32)
        meta_list = [] if return_meta else None
        patch_shift = self.image_size

        for i in range(self.dg.nb_valid):
            p_x = int(i / int(self.dg.area_height / self.image_size)) * patch_shift
            p_y = int(i % int(self.dg.area_height / self.image_size)) * patch_shift
            patch = np.copy(self.dg.norm_data[p_y : p_y + self.image_size, p_x : p_x + self.image_size])
            input_valid[i, :] = patch.flatten("C")
            targets_valid[i, :], meta = self._make_patch_target(
                self.dg.min_ra_train_pix + p_x,
                self.dg.min_dec_train_pix + p_y,
                0,
                0,
                0,
                return_meta,
            )
            if return_meta:
                meta_list.append(meta)

        if return_meta:
            return input_valid, targets_valid, meta_list
        return input_valid, targets_valid
