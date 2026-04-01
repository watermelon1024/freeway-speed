from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    class_name: str

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, y2)


@dataclass
class PolynomialLane:
    a: float
    b: float
    c: float

    def x(self, y: float | np.ndarray) -> float | np.ndarray:
        return self.a * y * y + self.b * y + self.c

    def dx_dy(self, y: float | np.ndarray) -> float | np.ndarray:
        return 2.0 * self.a * y + self.b


@dataclass
class TrackObservation:
    timestamp: float
    bev_y: float
    speed_kmh: Optional[float] = None


@dataclass
class TrackedVehicle:
    track_id: int
    bbox: tuple[float, float, float, float]
    score: float
    class_name: str
    bev_point: Optional[tuple[float, float]] = None
    distance_m: Optional[float] = None
    speed_kmh: Optional[float] = None
    direction: Optional[str] = None
    lane: Optional[str] = None


@dataclass
class FrameState:
    homography: Optional[np.ndarray] = None
    lane_mask: Optional[np.ndarray] = None
    lane_mask_bev: Optional[np.ndarray] = None
    lane_poly: Optional[PolynomialLane] = None
    scale_m_per_px: Optional[float] = None
    scale_source: str = "default"
    tracked: list[TrackedVehicle] = field(default_factory=list)
    lane_centers_departing: tuple[float, ...] = field(default_factory=tuple)
    lane_centers_approaching: tuple[float, ...] = field(default_factory=tuple)
    lane_avg_speed_kmh: dict[str, float] = field(default_factory=dict)
    direction_avg_speed_kmh: dict[str, float] = field(default_factory=dict)
