from __future__ import annotations

import numpy as np

from .config import CurveFitConfig
from .types import PolynomialLane


def fit_lane_polynomial(bev_mask: np.ndarray, cfg: CurveFitConfig) -> PolynomialLane | None:
    ys, xs = np.where(bev_mask > 0)
    if len(xs) < cfg.min_points:
        return None

    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    bins = np.linspace(y_min, y_max + 1, cfg.bins + 1, dtype=np.int32)

    centers_y: list[float] = []
    centers_x: list[float] = []
    for i in range(cfg.bins):
        y0, y1 = bins[i], bins[i + 1]
        idx = (ys >= y0) & (ys < y1)
        if np.count_nonzero(idx) < 6:
            continue
        centers_y.append(float(np.mean(ys[idx])))
        centers_x.append(float(np.mean(xs[idx])))

    if len(centers_y) < max(6, cfg.min_points // 4):
        return None

    coeff = np.polyfit(np.asarray(centers_y), np.asarray(centers_x), 2)
    return PolynomialLane(a=float(coeff[0]), b=float(coeff[1]), c=float(coeff[2]))
