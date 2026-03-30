from __future__ import annotations

from collections import deque

from .config import TrackingConfig
from .geometry import arc_length
from .types import PolynomialLane


class SpeedEstimator:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self._history: dict[int, deque[tuple[float, float]]] = {}
        self._smooth: dict[int, deque[float]] = {}

    def update(
        self,
        track_id: int,
        bev_y: float,
        timestamp: float,
        poly: PolynomialLane,
        scale_m_per_px: float,
    ) -> float | None:
        hist = self._history.setdefault(track_id, deque(maxlen=40))
        smooth = self._smooth.setdefault(track_id, deque(maxlen=self.cfg.speed_smooth_window))
        hist.append((bev_y, timestamp))

        if len(hist) < 2:
            return None

        y_now, t_now = hist[-1]
        idx = len(hist) - 2
        y_prev, t_prev = hist[idx]
        while idx > 0 and (t_now - t_prev) < self.cfg.min_dt_sec:
            idx -= 1
            y_prev, t_prev = hist[idx]

        dt = t_now - t_prev
        if dt <= 1e-6:
            return None

        travel_px = arc_length(poly, y_prev, y_now)
        travel_m = travel_px * scale_m_per_px
        speed_kmh = (travel_m / dt) * 3.6
        if speed_kmh > self.cfg.max_speed_kmh:
            if len(smooth) > 0:
                return float(sum(smooth) / len(smooth))
            return None
        smooth.append(speed_kmh)
        return float(sum(smooth) / len(smooth))
