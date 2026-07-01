"""detector targets with one native OBB size branch and physical shape parameters."""

from __future__ import annotations

import numpy as np

from orsdet_angle.angle_codec import encode_theta_le90, normalize_theta_le90
from orsdet_candidates.data import PatchMeta, CandidateDataBuilder, _obb_to_hbb_fast, _transform_boxes

from .decode import DETECTOR_MODE_SIZE, detector_layout, detector_target_dim
from .hard_group_sampler import HardGroupSampler, HardGroupSamplerConfig


def _transform_pa_le90(pa_deg, rot90: int, flip_w: int, flip_h: int):
    pa = np.asarray(pa_deg, dtype=np.float64).copy()
    if rot90 != 0:
        pa = normalize_theta_le90(pa + 90.0)
    if flip_w:
        pa = normalize_theta_le90(-pa)
    if flip_h:
        pa = normalize_theta_le90(-pa)
    return pa


class DetectorDataBuilder(CandidateDataBuilder):
    """Build detector targets for slim geometry/scoring separation.

    detector keeps OBB width/height only in the native YOLO box branch. The extra
    parameter heads regress SDC1 physical flux/Bmaj/Bmin. detector-S keeps separate
    OBB theta and physical PA angle pairs; shared-angle uses one shared angle pair.
    """

    def __init__(
        self,
        dg,
        slim_mode: str | None = None,
        hard_group_sampler: HardGroupSamplerConfig | None = None,
        **kwargs,
    ):
        self.layout = detector_layout(slim_mode)
        super().__init__(dg, **kwargs)
        self.target_dim = detector_target_dim(self.max_objects, self.layout.mode)
        self.phys_bmaj_norm = np.asarray(dg.bmaj_list, dtype=np.float64)
        self.phys_bmin_norm = np.asarray(dg.bmin_list, dtype=np.float64)
        self.phys_pa_deg = np.asarray(dg.pa_list, dtype=np.float64)
        self.phys_pa_known = np.asarray(self.source_table["bmaj_arcsec"], dtype=np.float64) > float(dg.pa_res_lim)
        self._center_x_order = np.argsort(self.boxes_global[:, 0])
        self._center_x_sorted = self.boxes_global[self._center_x_order, 0]
        self.hard_group_sampler = None
        if hard_group_sampler is not None and hard_group_sampler.enabled:
            self.hard_group_sampler = HardGroupSampler(self, hard_group_sampler)

    def _refresh_source_index(self):
        self._center_x_order = np.argsort(self.boxes_global[:, 0])
        self._center_x_sorted = self.boxes_global[self._center_x_order, 0]

    def reset_hard_group_sampler(self, config: HardGroupSamplerConfig | None):
        self.hard_group_sampler = None
        if config is not None and config.enabled:
            self.hard_group_sampler = HardGroupSampler(self, config)

    def _filter_sources(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self.source_ids.shape[0]:
            raise ValueError("Source filter length does not match source catalog.")
        if int(mask.sum()) <= 0:
            raise ValueError("Source filter removed every source.")

        self.source_ids = self.source_ids[mask]
        if hasattr(self.source_table, "iloc"):
            self.source_table = self.source_table.iloc[np.flatnonzero(mask)].reset_index(drop=True)
        else:
            self.source_table = self.source_table[mask]
        self.boxes_global = self.boxes_global[mask]
        self.global_hbb = self.global_hbb[mask]
        self.flux_norm = self.flux_norm[mask]
        self.phys_bmaj_norm = self.phys_bmaj_norm[mask]
        self.phys_bmin_norm = self.phys_bmin_norm[mask]
        self.phys_pa_deg = self.phys_pa_deg[mask]
        self.phys_pa_known = self.phys_pa_known[mask]
        self.angle_weights = self.angle_weights[mask]
        if hasattr(self, "small_elongated_boost_mask"):
            self.small_elongated_boost_mask = self.small_elongated_boost_mask[mask]
        self._refresh_source_index()

    def apply_flux_refine_target_table(
        self,
        target_table,
        *,
        source_id_col: str = "source_id",
        delta_adjust_col: str = "delta_log_adjust",
        target_only: bool = False,
    ) -> dict[str, object]:
        """Apply flux-refine catalog-derived flux targets to the training sources.

        The table supplies a log-flux adjustment relative to the original truth
        target. This keeps the ordinary detector target geometry intact while
        letting the frozen native delta channel learn the bright-gated gated target.
        """

        source_values = np.asarray(target_table[source_id_col], dtype=np.int64)
        adjust_values = np.asarray(target_table[delta_adjust_col], dtype=np.float64)
        valid = np.isfinite(adjust_values)
        source_values = source_values[valid]
        adjust_values = adjust_values[valid]
        if source_values.size == 0:
            raise ValueError("Flux refine target table has no finite target rows.")

        adjust_by_source: dict[int, float] = {}
        for source_id, adjust in zip(source_values, adjust_values):
            adjust_by_source[int(source_id)] = float(adjust)

        matched_mask = np.asarray([int(source_id) in adjust_by_source for source_id in self.source_ids], dtype=bool)
        n_before = int(self.source_ids.size)
        n_matched = int(matched_mask.sum())
        if n_matched <= 0:
            raise ValueError("Flux refine target table did not match any detector training source ids.")

        if target_only:
            self._filter_sources(matched_mask)
            source_adjust = np.asarray([adjust_by_source[int(source_id)] for source_id in self.source_ids], dtype=np.float64)
        else:
            source_adjust = np.zeros(self.source_ids.shape[0], dtype=np.float64)
            for idx, source_id in enumerate(self.source_ids):
                source_adjust[idx] = adjust_by_source.get(int(source_id), 0.0)

        flux_row = np.asarray(self.norm_lims[0], dtype=np.float64)
        flux_range = float(flux_row[0] - flux_row[1])
        if flux_range <= 1.0e-8:
            raise ValueError("Invalid flux normalization range.")
        log_flux = self.flux_norm * flux_range + float(flux_row[1])
        updated_norm = (log_flux + source_adjust - float(flux_row[1])) / flux_range
        self.flux_norm = np.clip(updated_norm, 0.0, 1.0)

        summary = {
            "n_sources_before": n_before,
            "n_sources_after": int(self.source_ids.size),
            "n_target_rows": int(source_values.size),
            "n_matched_before_filter": n_matched,
            "target_only": bool(target_only),
            "delta_adjust_col": str(delta_adjust_col),
            "delta_adjust_min": float(np.min(source_adjust)) if source_adjust.size else 0.0,
            "delta_adjust_max": float(np.max(source_adjust)) if source_adjust.size else 0.0,
            "delta_adjust_mean": float(np.mean(source_adjust)) if source_adjust.size else 0.0,
        }
        self.flux_refine_target_summary = summary
        return summary

    def _patch_indices(self, x0: float, y0: float):
        x1 = x0 + self.image_size
        y1 = y0 + self.image_size
        left = np.searchsorted(self._center_x_sorted, x0, side="left")
        right = np.searchsorted(self._center_x_sorted, x1, side="right")
        candidates = self._center_x_order[left:right]
        if candidates.size == 0:
            return candidates

        hbb = self.global_hbb[candidates]
        keep = (
            (hbb[:, 0] >= x0)
            & (hbb[:, 1] >= y0)
            & (hbb[:, 2] < x1)
            & (hbb[:, 3] < y1)
        )
        return np.sort(candidates[keep])

    def _build_norm_lims(self):
        flux_lims = np.asarray(self.dg.lims[0], dtype=np.float64)
        phys_bmaj_row = np.asarray(self.dg.lims[1], dtype=np.float64)
        phys_bmin_row = np.asarray(self.dg.lims[2], dtype=np.float64)
        return np.vstack([flux_lims, phys_bmaj_row, phys_bmin_row])

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
        phys_pa_local = _transform_pa_le90(self.phys_pa_deg[idx], rot90, flip_w, flip_h)
        phys_pa_local = np.where(self.phys_pa_known[idx], phys_pa_local, 0.0)

        n_obj = min(self.max_objects, idx.size)
        target[0] = n_obj
        for out_idx, src_idx in enumerate(idx[:n_obj]):
            cx, cy, w, h, theta = local_boxes[out_idx]
            obb_angle_vec = np.asarray(encode_theta_le90(theta), dtype=np.float64).reshape(-1)
            phys_pa_vec = np.asarray(encode_theta_le90(phys_pa_local[out_idx]), dtype=np.float64).reshape(-1)
            row_values = [
                1.0,
                cx - 0.5 * w,
                cy - 0.5 * h,
                0.0,
                cx + 0.5 * w,
                cy + 0.5 * h,
                1.0,
                self.flux_norm[src_idx],
                self.phys_bmaj_norm[src_idx],
                self.phys_bmin_norm[src_idx],
            ]
            for _ in range(max(0, self.layout.nb_param - 3)):
                row_values.append(0.0)
            if self.layout.mode == DETECTOR_MODE_SIZE:
                row_values.extend([obb_angle_vec[0], obb_angle_vec[1], phys_pa_vec[0], phys_pa_vec[1]])
            else:
                # shared-angle keeps the RotIoU geometry exact and lets catalog PA
                # share the same geometric angle at decode time.
                row_values.extend([obb_angle_vec[0], obb_angle_vec[1]])
            row_values.append(self.angle_weights[src_idx])
            row = np.asarray(row_values, dtype=np.float32)
            start = 1 + out_idx * self.layout.target_stride
            target[start : start + self.layout.target_stride] = row

        if return_meta:
            meta = PatchMeta(
                source_id=self.source_ids[idx[:n_obj]],
                obb=local_boxes[:n_obj].copy(),
                hbb=hbb[:n_obj].copy(),
            )
            return target, meta
        return target, None

    def hard_group_description(self) -> str:
        if self.hard_group_sampler is None:
            return "disabled"
        return self.hard_group_sampler.describe()

    def create_train_batch(self, return_meta: bool = False):
        if self.hard_group_sampler is None:
            return super().create_train_batch(return_meta=return_meta)

        input_data = np.zeros((self.dg.nb_images_iter, self.image_size * self.image_size), dtype=np.float32)
        targets = np.zeros((self.dg.nb_images_iter, self.target_dim), dtype=np.float32)
        meta_list = [] if return_meta else None

        for i in range(self.dg.nb_images_iter):
            use_noise = np.random.rand() <= self.dg.add_noise_prop
            use_hard = (not use_noise) and (np.random.rand() < float(self.hard_group_sampler.config.fraction))
            if use_hard:
                p_x, p_y, _, _ = self.hard_group_sampler.sample_patch_origin()
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
            elif not use_noise:
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
