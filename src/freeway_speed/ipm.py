from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import IPMConfig


@dataclass
class DynamicIPM:
    config: IPMConfig
    _h_ema: np.ndarray | None = None

    def _fit_lane_lines(self, roi_mask: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        lines = cv2.HoughLinesP(roi_mask, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=20)
        if lines is None:
            return None, None

        left: list[np.ndarray] = []
        right: list[np.ndarray] = []
        cx = roi_mask.shape[1] * 0.5
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line.astype(float)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1e-6:
                continue
            slope = dy / dx
            if abs(slope) < 0.2:
                continue
            mid_x = (x1 + x2) * 0.5
            vec = np.array([x1, y1, x2, y2], dtype=np.float64)
            if mid_x < cx and slope < 0:
                left.append(vec)
            elif mid_x >= cx and slope > 0:
                right.append(vec)

        left_line = np.mean(left, axis=0) if left else None
        right_line = np.mean(right, axis=0) if right else None
        return left_line, right_line

    @staticmethod
    def _line_intersection(l1: np.ndarray, l2: np.ndarray) -> tuple[float, float] | None:
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
        return float(px), float(py)

    def estimate_homography(self, lane_mask: np.ndarray) -> np.ndarray:
        h, w = lane_mask.shape[:2]
        roi_h = int(h * self.config.roi_ratio_from_bottom)
        y0 = h - roi_h
        roi = lane_mask[y0:h, :]

        left_line, right_line = self._fit_lane_lines(roi)
        if left_line is not None:
            left_line[[1, 3]] += y0
        if right_line is not None:
            right_line[[1, 3]] += y0

        if left_line is not None and right_line is not None:
            vp = self._line_intersection(left_line, right_line)
        else:
            vp = None

        if vp is None:
            vp_x, vp_y = w * 0.5, h * 0.5
        else:
            vp_x, vp_y = vp

        bottom_y = h - 1
        top_y = int(max(0, min(h - 1, vp_y + roi_h * 0.25)))
        half_bottom = int(w * 0.3)
        half_top = int(max(40, w * 0.12))

        src = np.float32(
            [
                [max(0, vp_x - half_top), top_y],
                [min(w - 1, vp_x + half_top), top_y],
                [min(w - 1, vp_x + half_bottom), bottom_y],
                [max(0, vp_x - half_bottom), bottom_y],
            ]
        )
        dst = np.float32(
            [
                [0, 0],
                [self.config.bev_width - 1, 0],
                [self.config.bev_width - 1, self.config.bev_height - 1],
                [0, self.config.bev_height - 1],
            ]
        )
        h_mat = cv2.getPerspectiveTransform(src, dst)

        if self._h_ema is None:
            self._h_ema = h_mat
        else:
            alpha = self.config.ema_alpha
            self._h_ema = (1.0 - alpha) * self._h_ema + alpha * h_mat
        return self._h_ema

    def warp_mask(self, lane_mask: np.ndarray, h_mat: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(lane_mask, h_mat, (self.config.bev_width, self.config.bev_height))

    @staticmethod
    def transform_points(points: np.ndarray, h_mat: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        points = points.astype(np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, h_mat)
        return transformed.reshape(-1, 2)
