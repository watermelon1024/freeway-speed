from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from .config import SystemConfig
from .curve import fit_lane_polynomial
from .geometry import distance_to_camera_m, estimate_scale_from_dashed_line, estimate_scale_from_lane_width
from .ipm import DynamicIPM
from .perception import (
    LaneSegmenter,
    ONNXLaneSegmenter,
    ThresholdLaneSegmenter,
    ThresholdVehicleDetector,
    VehicleDetector,
    YoloV8Detector,
)
from .speed import SpeedEstimator
from .tracking import ByteTrackTracker, SimpleByteLikeTracker
from .types import Detection, FrameState, TrackedVehicle


@dataclass
class FreewaySpeedPipeline:
    config: SystemConfig
    frame_rate: int = 30

    def __post_init__(self) -> None:
        self.vehicle_detector = self._build_vehicle_detector(self.config)
        self.lane_segmenter = self._build_lane_segmenter(self.config)
        self.ipm = DynamicIPM(self.config.ipm)
        self._lane_background_avg: np.ndarray | None = None
        if self.config.tracking.tracker_backend == "bytetrack":
            self.tracker = ByteTrackTracker(self.config.tracking, frame_rate=self.frame_rate)
        else:
            self.tracker = SimpleByteLikeTracker(self.config.tracking)
        self.speed = SpeedEstimator(self.config.tracking)

        self.frame_idx = 0
        self.state = FrameState()
        self.state.scale_m_per_px = self.config.calibration.default_scale_m_per_px
        self.state.scale_source = "default"

    @staticmethod
    def _build_vehicle_detector(cfg: SystemConfig) -> VehicleDetector:
        if cfg.perception.vehicle_backend == "yolo":
            return YoloV8Detector(
                model_path=cfg.perception.yolo_model,
                classes=cfg.perception.yolo_classes,
                conf_threshold=cfg.perception.yolo_conf,
                imgsz=cfg.perception.yolo_imgsz,
                iou_threshold=cfg.perception.yolo_iou,
                upscale_factor=cfg.perception.vehicle_upscale_factor,
            )
        return ThresholdVehicleDetector()

    @staticmethod
    def _build_lane_segmenter(cfg: SystemConfig) -> LaneSegmenter:
        if cfg.perception.lane_backend == "onnx" and cfg.perception.lane_onnx:
            return ONNXLaneSegmenter(cfg.perception.lane_onnx)
        return ThresholdLaneSegmenter()

    def _prepare_lane_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.perception.lane_temporal_smoothing:
            return frame

        if self._lane_background_avg is None:
            self._lane_background_avg = frame.astype(np.float32)
        cv2.accumulateWeighted(frame, self._lane_background_avg, self.config.perception.lane_temporal_alpha)
        return cv2.convertScaleAbs(self._lane_background_avg)

    def _update_lane_model_if_needed(self, lane_frame: np.ndarray) -> None:
        if self.frame_idx % self.config.runtime.lane_update_every_n_frames != 0:
            return

        lane_mask = self.lane_segmenter.segment(lane_frame)
        h_mat = self.ipm.estimate_homography(lane_mask)
        bev_mask = self.ipm.warp_mask(lane_mask, h_mat)
        poly = fit_lane_polynomial(bev_mask, self.config.curve)

        self.state.lane_mask = lane_mask
        self.state.homography = h_mat
        self.state.lane_mask_bev = bev_mask
        self.state.lane_poly = poly

        if poly is not None:
            scale = estimate_scale_from_dashed_line(bev_mask, poly, self.config.calibration)
            if scale is not None:
                self.state.scale_m_per_px = scale
                self.state.scale_source = "dashed"
            else:
                lane_width_scale = estimate_scale_from_lane_width(bev_mask, self.config.calibration)
                if lane_width_scale is not None:
                    self.state.scale_m_per_px = lane_width_scale
                    self.state.scale_source = "lane_width"
                else:
                    self.state.scale_m_per_px = self.config.calibration.default_scale_m_per_px
                    self.state.scale_source = "default"

    def process_frame(self, frame: np.ndarray, timestamp: float) -> FrameState:
        self.frame_idx += 1
        lane_frame = self._prepare_lane_frame(frame)
        self._update_lane_model_if_needed(lane_frame)

        detections: list[Detection] = self.vehicle_detector.detect(frame)
        if self.config.tracking.tracker_backend == "bytetrack":
            bt_tracker = cast(ByteTrackTracker, self.tracker)
            tracks = bt_tracker.update(detections, frame=frame)
        else:
            simple_tracker = cast(SimpleByteLikeTracker, self.tracker)
            tracks = simple_tracker.update(detections)

        tracked: list[TrackedVehicle] = []
        homography = self.state.homography
        poly = self.state.lane_poly
        scale = self.state.scale_m_per_px

        for tid, tr in tracks.items():
            vehicle = TrackedVehicle(
                track_id=tid,
                bbox=tr.bbox,
                score=tr.score,
                class_name=tr.class_name,
            )

            if homography is not None and poly is not None and scale is not None:
                x1, y1, x2, y2 = tr.bbox
                pt = np.asarray([[(x1 + x2) * 0.5, y2]], dtype=np.float32)
                bev_pt = self.ipm.transform_points(pt, homography)
                if len(bev_pt) == 1:
                    bx, by = float(bev_pt[0, 0]), float(bev_pt[0, 1])
                    vehicle.bev_point = (bx, by)
                    vehicle.distance_m = distance_to_camera_m(
                        poly=poly,
                        y_vehicle=by,
                        bev_bottom_y=float(self.config.ipm.bev_height - 1),
                        scale_m_per_px=scale,
                    )
                    vehicle.speed_kmh = self.speed.update(tid, by, timestamp, poly, scale)

            tracked.append(vehicle)

        self.state.tracked = tracked
        return self.state


def draw_overlay(frame: np.ndarray, state: FrameState) -> np.ndarray:
    vis = frame.copy()
    for obj in state.tracked:
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 1)
        text = f"ID {obj.track_id}"
        if obj.distance_m is not None:
            text += f" | {obj.distance_m:.1f} m"
        if obj.speed_kmh is not None:
            text += f" | {obj.speed_kmh:.1f} km/h"
        cv2.putText(
            vis,
            text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

    if state.scale_m_per_px is not None:
        cv2.putText(
            vis,
            f"scale: {state.scale_m_per_px:.5f} m/px",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
    return vis
