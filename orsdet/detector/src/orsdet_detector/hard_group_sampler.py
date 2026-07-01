"""Hard-group source sampler for detector training.

The design follows the balanced-learning idea from Libra R-CNN: if the model
fails on distribution tails, make the sampler explicitly expose those tails
instead of hoping uniform random crops will cover them often enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HardGroupSamplerConfig:
    enabled: bool = False
    fraction: float = 0.35
    jitter: int = 48
    seed: int | None = None


class HardGroupSampler:
    """Choose source-centered training crops from flux/size/crowding tails."""

    def __init__(self, builder, config: HardGroupSamplerConfig):
        self.builder = builder
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.groups = self._build_groups()
        self.group_names = tuple(name for name, idx in self.groups.items() if idx.size > 0)
        if not self.group_names:
            raise ValueError("HardGroupSampler has no non-empty source groups.")

    def _build_groups(self) -> dict[str, np.ndarray]:
        b = self.builder
        flux = np.asarray(b.flux_norm, dtype=np.float64)
        size = np.sqrt(
            np.maximum(
                np.asarray(b.source_table["w_pix"], dtype=np.float64)
                * np.asarray(b.source_table["h_pix"], dtype=np.float64),
                0.0,
            )
        )
        aspect = np.asarray(b.source_table["aspect_ratio"], dtype=np.float64)
        x = np.asarray(b.boxes_global[:, 0], dtype=np.float64)
        y = np.asarray(b.boxes_global[:, 1], dtype=np.float64)
        q_flux = np.nanquantile(flux, [1.0 / 3.0, 2.0 / 3.0])
        q_size = np.nanquantile(size, [1.0 / 3.0, 2.0 / 3.0])

        groups = {
            "faint_small": np.flatnonzero((flux <= q_flux[0]) & (size <= q_size[0])),
            "bright_large": np.flatnonzero((flux >= q_flux[1]) & (size >= q_size[1])),
            "large": np.flatnonzero(size >= q_size[1]),
            "large_elongated": np.flatnonzero((size >= q_size[1]) & (aspect >= 2.0)),
            "faint": np.flatnonzero(flux <= q_flux[0]),
            "bright": np.flatnonzero(flux >= q_flux[1]),
        }
        if x.size > 1:
            coords = np.column_stack([x, y])
            nn = _nearest_neighbor_distance(coords)
            groups["crowded"] = np.flatnonzero(nn <= np.nanquantile(nn, 1.0 / 3.0))
        return groups

    def sample_patch_origin(self) -> tuple[int, int, str, int]:
        name = str(self.rng.choice(self.group_names))
        indices = self.groups[name]
        src_idx = int(self.rng.choice(indices))
        cx, cy = self.builder.boxes_global[src_idx, 0:2]
        jitter = int(max(0, self.config.jitter))
        if jitter > 0:
            cx += int(self.rng.integers(-jitter, jitter + 1))
            cy += int(self.rng.integers(-jitter, jitter + 1))
        # boxes_global is in full-image pixels, while dg.norm_data is the
        # training cutout. Convert the source-centered crop origin to cutout
        # coordinates before clipping.
        x0 = int(round(cx - self.builder.dg.min_ra_train_pix - 0.5 * self.builder.image_size))
        y0 = int(round(cy - self.builder.dg.min_dec_train_pix - 0.5 * self.builder.image_size))
        dg = self.builder.dg
        x0 = int(np.clip(x0, 0, max(0, int(dg.area_width - self.builder.image_size - 1))))
        y0 = int(np.clip(y0, 0, max(0, int(dg.area_height - self.builder.image_size - 1))))
        return x0, y0, name, src_idx

    def describe(self) -> str:
        rows = [
            "enabled=%s fraction=%.6g jitter=%d seed=%s"
            % (self.config.enabled, self.config.fraction, self.config.jitter, self.config.seed)
        ]
        for name in self.group_names:
            rows.append("%s=%d" % (name, self.groups[name].size))
        return " ".join(rows)


def _nearest_neighbor_distance(coords: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        dists, _ = cKDTree(coords).query(coords, k=2, workers=-1)
        return np.asarray(dists[:, 1], dtype=np.float64)
    except Exception:
        # O(n^2) fallback for environments without scipy; this is only used at
        # sampler construction and training catalogs are manageable.
        n = coords.shape[0]
        out = np.full(n, np.inf, dtype=np.float64)
        for i in range(n):
            diff = coords - coords[i]
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            dist[i] = np.inf
            out[i] = np.min(dist)
        return out
