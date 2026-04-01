from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class TrafficConfig:
    direction_lookback_sec: float = 0.5
    direction_min_displacement_px: float = 10.0
    calibration_warmup_sec: float = 20.0
    calibration_min_samples: int = 80
    lane_hist_bins: int = 50
    lane_peak_prominence: float = 10.0
    lane_match_max_distance_px: float = 40.0
    lane_match_spacing_ratio: float = 0.85
    lane_match_span_ratio: float = 0.20
    flow_window_sec: float = 45.0
    flow_sample_interval_sec: float = 1.0
    dynamic_recalibration_interval_sec: float = 300.0
    dynamic_recalibration_lookback_sec: float = 60.0


@dataclass
class TrafficSnapshot:
    lane_centers: dict[str, tuple[float, ...]]
    lane_avg_speed_kmh: dict[str, float]
    direction_avg_speed_kmh: dict[str, float]


class TrafficAnalyzer:
    def __init__(self, cfg: TrafficConfig, bev_width: int):
        self.cfg = cfg
        self._bev_width = max(1, int(bev_width))
        self._track_hist: dict[int, deque[tuple[float, float, float]]] = {}
        self._moving_x_hist: dict[str, deque[tuple[float, float]]] = {
            "Departing": deque(),
            "Approaching": deque(),
        }
        self._warmup_x: dict[str, list[float]] = {
            "Departing": [],
            "Approaching": [],
        }
        self._lane_centers: dict[str, tuple[float, ...]] = {
            "Departing": tuple(),
            "Approaching": tuple(),
        }
        self._lane_bounds: dict[str, tuple[float, float]] = {
            "Departing": (0.0, 0.0),
            "Approaching": (0.0, 0.0),
        }
        self._flow_records: deque[tuple[float, int, str, str, float]] = deque()
        self._last_flow_sample_ts: dict[int, float] = {}
        self._start_ts: float | None = None
        self._last_recalibration_ts: float | None = None

    def update_vehicle(
        self,
        track_id: int,
        timestamp: float,
        bev_x: float,
        bev_y: float,
        speed_kmh: float | None,
    ) -> tuple[str, str]:
        if self._start_ts is None:
            self._start_ts = timestamp
        if self._last_recalibration_ts is None:
            self._last_recalibration_ts = timestamp

        direction, moving = self._infer_direction(track_id, timestamp, bev_x, bev_y)

        if moving and direction in self._moving_x_hist:
            if self._is_valid_bev_x(bev_x):
                self._moving_x_hist[direction].append((timestamp, bev_x))
                self._warmup_collect(timestamp, direction, bev_x)

        self._maybe_recalibrate(timestamp)
        lane = self._assign_lane(direction, bev_x)

        if (
            speed_kmh is not None
            and direction in ("Departing", "Approaching")
            and lane
            not in (
                "Unknown",
                "LaneChanging",
            )
        ):
            self._append_flow_sample(track_id, timestamp, direction, lane, speed_kmh)

        self._prune_flow(timestamp)
        return direction, lane

    def snapshot(self, timestamp: float) -> TrafficSnapshot:
        self._prune_flow(timestamp)

        lane_buckets: dict[tuple[str, str], list[float]] = {}
        direction_buckets: dict[str, list[float]] = {}
        for _, _, direction, lane, speed in self._flow_records:
            lane_key = (direction, lane)
            lane_buckets.setdefault(lane_key, []).append(speed)
            direction_buckets.setdefault(direction, []).append(speed)

        lane_avg_speed: dict[str, float] = {}
        for (direction, lane), speeds in lane_buckets.items():
            lane_avg_speed[f"{direction}:{lane}"] = float(sum(speeds) / len(speeds))

        direction_avg_speed: dict[str, float] = {}
        for direction, speeds in direction_buckets.items():
            direction_avg_speed[direction] = float(sum(speeds) / len(speeds))

        return TrafficSnapshot(
            lane_centers={
                "Departing": self._lane_centers["Departing"],
                "Approaching": self._lane_centers["Approaching"],
            },
            lane_avg_speed_kmh=lane_avg_speed,
            direction_avg_speed_kmh=direction_avg_speed,
        )

    def _infer_direction(
        self,
        track_id: int,
        timestamp: float,
        bev_x: float,
        bev_y: float,
    ) -> tuple[str, bool]:
        hist = self._track_hist.setdefault(track_id, deque(maxlen=120))
        hist.append((timestamp, bev_x, bev_y))

        if len(hist) < 2:
            return "Unknown", False

        now_t, _, now_y = hist[-1]
        prev_t, _, prev_y = hist[-2]
        for t, _, y in reversed(hist):
            if (now_t - t) >= self.cfg.direction_lookback_sec:
                prev_t = t
                prev_y = y
                break

        if (now_t - prev_t) <= 1e-6:
            return "Unknown", False

        dy = now_y - prev_y
        if abs(dy) < self.cfg.direction_min_displacement_px:
            return "Unknown", False
        if dy < 0:
            return "Departing", True
        return "Approaching", True

    def _warmup_collect(self, timestamp: float, direction: str, bev_x: float) -> None:
        _ = timestamp
        self._warmup_x[direction].append(bev_x)

    def _maybe_recalibrate(self, timestamp: float) -> None:
        if self._start_ts is None or self._last_recalibration_ts is None:
            return

        warmup_done = (timestamp - self._start_ts) >= self.cfg.calibration_warmup_sec
        sample_ready = (
            len(self._warmup_x["Departing"]) >= self.cfg.calibration_min_samples
            or len(self._warmup_x["Approaching"]) >= self.cfg.calibration_min_samples
        )
        if (warmup_done or sample_ready) and (
            not self._lane_centers["Departing"] or not self._lane_centers["Approaching"]
        ):
            self._rebuild_lane_centers(self._warmup_x)
            self._last_recalibration_ts = timestamp
            return

        if (timestamp - self._last_recalibration_ts) < self.cfg.dynamic_recalibration_interval_sec:
            return

        lookback_samples: dict[str, list[float]] = {
            "Departing": [],
            "Approaching": [],
        }
        min_t = timestamp - self.cfg.dynamic_recalibration_lookback_sec
        for direction in ("Departing", "Approaching"):
            dq = self._moving_x_hist[direction]
            while dq and dq[0][0] < min_t:
                dq.popleft()
            lookback_samples[direction] = [x for _, x in dq]

        self._rebuild_lane_centers(lookback_samples)
        self._last_recalibration_ts = timestamp

    def _rebuild_lane_centers(self, source: dict[str, list[float]]) -> None:
        for direction in ("Departing", "Approaching"):
            result = self._calibrate_lanes(source[direction])
            if result is None:
                continue
            centers, lo, hi = result
            if not centers:
                continue
            if direction == "Departing":
                self._lane_centers[direction] = tuple(sorted(centers))
            else:
                self._lane_centers[direction] = tuple(sorted(centers, reverse=True))
            self._lane_bounds[direction] = (float(lo), float(hi))

    def _calibrate_lanes(self, x_coordinates: list[float]) -> tuple[list[float], float, float] | None:
        x_arr = np.asarray(x_coordinates, dtype=np.float32)
        x_arr = x_arr[np.isfinite(x_arr)]
        if x_arr.size < 8:
            return None

        lo, hi = np.percentile(x_arr, [2.0, 98.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) <= 1e-3:
            return None
        x_arr = x_arr[(x_arr >= lo) & (x_arr <= hi)]
        if x_arr.size < 8:
            return None

        counts, bin_edges = np.histogram(x_arr, bins=self.cfg.lane_hist_bins, range=(float(lo), float(hi)))
        peaks, _ = find_peaks(counts, prominence=self.cfg.lane_peak_prominence)

        if len(peaks) == 0:
            top_bins = np.argsort(counts)[::-1]
            selected: list[int] = []
            min_gap = max(1, self.cfg.lane_hist_bins // 8)
            for b in top_bins:
                if counts[b] <= 0:
                    break
                if all(abs(int(b) - int(s)) >= min_gap for s in selected):
                    selected.append(int(b))
                if len(selected) >= 3:
                    break
            peaks = np.asarray(sorted(selected), dtype=int)

        lane_centers: list[float] = []
        for p in peaks:
            center_x = float((bin_edges[p] + bin_edges[p + 1]) * 0.5)
            lane_centers.append(center_x)
        return lane_centers, float(lo), float(hi)

    def _assign_lane(self, direction: str, bev_x: float) -> str:
        if direction not in ("Departing", "Approaching"):
            return "Unknown"
        if not self._is_valid_bev_x(bev_x):
            return "Unknown"

        centers = self._lane_centers[direction]
        if not centers:
            return "Unknown"

        dists = [abs(bev_x - c) for c in centers]
        closest_idx = int(np.argmin(dists))

        dynamic_threshold = self.cfg.lane_match_max_distance_px
        lo, hi = self._lane_bounds[direction]
        span = hi - lo
        if span > 1e-3:
            dynamic_threshold = max(dynamic_threshold, span * self.cfg.lane_match_span_ratio)
        if len(centers) >= 2:
            if closest_idx == 0:
                spacing = abs(centers[1] - centers[0])
            elif closest_idx == len(centers) - 1:
                spacing = abs(centers[-1] - centers[-2])
            else:
                spacing = min(
                    abs(centers[closest_idx] - centers[closest_idx - 1]),
                    abs(centers[closest_idx + 1] - centers[closest_idx]),
                )
            dynamic_threshold = max(dynamic_threshold, spacing * self.cfg.lane_match_spacing_ratio)

        if dists[closest_idx] > dynamic_threshold:
            return "LaneChanging"

        if len(centers) == 1:
            return "Middle"

        if closest_idx == 0:
            return "InnerFast"
        if closest_idx == (len(centers) - 1):
            return "OuterSlow"
        return "Middle"

    def _is_valid_bev_x(self, bev_x: float) -> bool:
        return bool(np.isfinite(bev_x)) and abs(bev_x) < 20000.0

    def _append_flow_sample(
        self,
        track_id: int,
        timestamp: float,
        direction: str,
        lane: str,
        speed_kmh: float,
    ) -> None:
        last_ts = self._last_flow_sample_ts.get(track_id)
        if last_ts is not None and (timestamp - last_ts) < self.cfg.flow_sample_interval_sec:
            return

        self._last_flow_sample_ts[track_id] = timestamp
        self._flow_records.append((timestamp, track_id, direction, lane, speed_kmh))

    def _prune_flow(self, timestamp: float) -> None:
        min_t = timestamp - self.cfg.flow_window_sec
        while self._flow_records and self._flow_records[0][0] < min_t:
            _, track_id, _, _, _ = self._flow_records.popleft()
            latest = self._last_flow_sample_ts.get(track_id)
            if latest is not None and latest < min_t:
                del self._last_flow_sample_ts[track_id]
