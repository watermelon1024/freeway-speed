from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from .config import CalibrationConfig
from .types import PolynomialLane


def arc_length(poly: PolynomialLane, y1: float, y2: float) -> float:
    if abs(y2 - y1) < 1e-6:
        return 0.0
    low, high = (y1, y2) if y1 <= y2 else (y2, y1)

    def integrand(y: float) -> float:
        d = poly.dx_dy(y)
        return math.sqrt(1.0 + d * d)

    val, _ = quad(integrand, low, high, limit=80)
    return float(val)


def estimate_scale_from_dashed_line(
    bev_mask: np.ndarray, poly: PolynomialLane, cfg: CalibrationConfig
) -> float | None:
    h = bev_mask.shape[0]
    sampled: list[bool] = []
    for y in range(h):
        x = int(round(float(poly.x(y))))
        x0 = max(0, x - cfg.search_band_px)
        x1 = min(bev_mask.shape[1], x + cfg.search_band_px + 1)
        if x0 >= x1:
            sampled.append(False)
            continue
        sampled.append(bool(np.any(bev_mask[y, x0:x1] > 0)))

    if not any(sampled):
        return None

    # Collect runs of white segments and black gaps along the fitted lane curve.
    runs: list[tuple[bool, int, int, int]] = []
    cur_val = sampled[0]
    start = 0
    for i in range(1, h):
        if sampled[i] != cur_val:
            runs.append((cur_val, start, i - 1, i - start))
            cur_val = sampled[i]
            start = i
    runs.append((cur_val, start, h - 1, h - start))

    candidate_lengths: list[float] = []
    for idx, (is_white, y0, y1, run_len) in enumerate(runs):
        if not is_white:
            continue
        if run_len < cfg.min_dash_pixels or run_len > cfg.max_dash_pixels:
            continue

        left_gap_ok = idx > 0 and (not runs[idx - 1][0]) and runs[idx - 1][3] >= cfg.min_gap_pixels
        right_gap_ok = (
            idx < len(runs) - 1 and (not runs[idx + 1][0]) and runs[idx + 1][3] >= cfg.min_gap_pixels
        )
        if not (left_gap_ok or right_gap_ok):
            continue

        candidate_lengths.append(arc_length(poly, float(y0), float(y1)))

    if len(candidate_lengths) < cfg.min_dash_runs:
        return None

    pixel_len = float(np.median(np.asarray(candidate_lengths, dtype=np.float32)))
    if pixel_len < 1e-6:
        return None

    scale = cfg.dash_length_m / pixel_len
    if scale < cfg.min_scale_m_per_px or scale > cfg.max_scale_m_per_px:
        return None
    return scale


def estimate_scale_from_lane_width(bev_mask: np.ndarray, cfg: CalibrationConfig) -> float | None:
    h, w = bev_mask.shape[:2]
    y0 = int(max(0, min(h - 1, round(h * cfg.lane_hist_roi_start_ratio))))
    roi = bev_mask[y0:h, :]
    if roi.size == 0:
        return None

    hist = np.sum(roi > 0, axis=0).astype(np.float32)
    if np.max(hist) < cfg.lane_peak_min_votes:
        return None

    kernel = np.ones((9,), dtype=np.float32) / 9.0
    smooth = np.convolve(hist, kernel, mode="same")

    peak_thresh = max(float(cfg.lane_peak_min_votes), float(np.percentile(smooth, 75)))
    peaks: list[int] = []
    for i in range(1, w - 1):
        if smooth[i] >= peak_thresh and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            peaks.append(i)

    if len(peaks) < 2:
        return None

    best_pair: tuple[int, int] | None = None
    best_score = -1.0
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            sep = peaks[j] - peaks[i]
            if sep < cfg.lane_width_min_px or sep > cfg.lane_width_max_px:
                continue
            score = float(smooth[peaks[i]] + smooth[peaks[j]])
            if score > best_score:
                best_score = score
                best_pair = (peaks[i], peaks[j])

    if best_pair is None:
        return None

    lane_width_px = float(best_pair[1] - best_pair[0])
    if lane_width_px <= 1e-6:
        return None

    scale = cfg.lane_width_m / lane_width_px
    if scale < cfg.min_scale_m_per_px or scale > cfg.max_scale_m_per_px:
        return None
    return scale


def distance_to_camera_m(
    poly: PolynomialLane,
    y_vehicle: float,
    bev_bottom_y: float,
    scale_m_per_px: float,
) -> float:
    px_dist = arc_length(poly, y_vehicle, bev_bottom_y)
    return px_dist * scale_m_per_px
