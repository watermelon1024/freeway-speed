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
from .traffic import TrafficAnalyzer
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
        self.traffic = TrafficAnalyzer(self.config.traffic, bev_width=self.config.ipm.bev_width)

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
                    vehicle.direction, vehicle.lane = self.traffic.update_vehicle(
                        track_id=tid,
                        timestamp=timestamp,
                        bev_x=bx,
                        bev_y=by,
                        speed_kmh=vehicle.speed_kmh,
                    )

            tracked.append(vehicle)

        self.state.tracked = tracked
        traffic_snapshot = self.traffic.snapshot(timestamp)
        self.state.lane_centers_departing = traffic_snapshot.lane_centers.get("Departing", tuple())
        self.state.lane_centers_approaching = traffic_snapshot.lane_centers.get("Approaching", tuple())
        self.state.lane_avg_speed_kmh = traffic_snapshot.lane_avg_speed_kmh
        self.state.direction_avg_speed_kmh = traffic_snapshot.direction_avg_speed_kmh
        return self.state


def draw_overlay(frame: np.ndarray, state: FrameState) -> np.ndarray:
    base_h, base_w = frame.shape[:2]
    render_scale = 1.5
    top_margin = max(80, int(round(base_h * 0.12)))
    right_margin = max(460, int(round(base_w * 0.52)))
    out_w = int(round(base_w * render_scale)) + right_margin
    out_h = int(round(base_h * render_scale)) + top_margin

    vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    scaled_frame = cv2.resize(
        frame,
        (int(round(base_w * render_scale)), int(round(base_h * render_scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    vis[top_margin : top_margin + scaled_frame.shape[0], : scaled_frame.shape[1]] = scaled_frame

    lane_x0 = 0
    lane_y0 = top_margin

    def sx(x: float) -> int:
        return int(round(lane_x0 + x * render_scale))

    def sy(y: float) -> int:
        return int(round(lane_y0 + y * render_scale))

    box_thickness = 1
    text_scale_small = 0.35 * render_scale
    text_scale_mid = 0.6 * render_scale
    text_thickness = max(1, int(round(1.2 * render_scale)))
    vehicle_text_thickness = 2

    for obj in state.tracked:
        x1, y1, x2, y2 = obj.bbox
        ix1, iy1, ix2, iy2 = sx(x1), sy(y1), sx(x2), sy(y2)
        cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 200, 255), box_thickness)
        speed_text = f"{obj.speed_kmh:.0f}km/h" if obj.speed_kmh is not None else "-"
        dir_text = (obj.direction or "N/A").replace("Unknown", "N/A")
        lane_text = (obj.lane or "N/A").replace("Unknown", "N/A")
        text = f"#{obj.track_id} | {speed_text} | {dir_text} | {lane_text}"
        cv2.putText(
            vis,
            text,
            (ix1, max(top_margin + 20, iy1 - int(round(10 * render_scale)))),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale_small,
            (0, 255, 0),
            vehicle_text_thickness,
        )

    if state.scale_m_per_px is not None:
        cv2.putText(
            vis,
            f"scale: {state.scale_m_per_px:.5f} m/px",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale_mid,
            (255, 255, 0),
            text_thickness,
        )

    y0 = 36 + int(round(34 * render_scale))
    for direction in ("Departing", "Approaching"):
        avg = state.direction_avg_speed_kmh.get(direction)
        if avg is None:
            continue
        cv2.putText(
            vis,
            f"{direction} avg: {avg:.1f} km/h",
            (24, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale_mid,
            (80, 255, 80),
            text_thickness,
        )
        y0 += int(round(26 * render_scale))

    # Right-side panel lives in dedicated black margin, so text never clips into camera view.
    panel_w = max(420, right_margin - 24)
    x0 = vis.shape[1] - panel_w - 10
    y_panel = top_margin + 12
    row_h = int(round(22 * render_scale))
    header_h = int(round(40 * render_scale))
    panel_h = min(vis.shape[0] - y_panel - 12, header_h + min(len(state.tracked), 24) * row_h + 8)
    if panel_h > 40:
        overlay = vis.copy()
        cv2.rectangle(overlay, (x0, y_panel), (x0 + panel_w, y_panel + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.42, vis, 0.58, 0, vis)
        cv2.rectangle(vis, (x0, y_panel), (x0 + panel_w, y_panel + panel_h), (100, 230, 255), 1)
        cv2.putText(
            vis,
            "#ID | speed | dir | lane",
            (x0 + 10, y_panel + int(round(18 * render_scale))),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale_small,
            (230, 255, 255),
            2,
        )

        y = y_panel + header_h
        max_rows = max(1, (panel_h - header_h - 8) // row_h)
        sorted_objs = sorted(state.tracked, key=lambda o: o.track_id)
        for obj in sorted_objs[:max_rows]:
            direction = (obj.direction or "N/A").replace("Unknown", "N/A")
            lane = (obj.lane or "N/A").replace("Unknown", "N/A")
            speed_text = f"{obj.speed_kmh:.0f}" if obj.speed_kmh is not None else "-"
            row = f"#{obj.track_id:<3} | {speed_text:>3} | {direction[:4]:<4} | {lane[:8]:<8}"
            cv2.putText(
                vis,
                row,
                (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale_small,
                (240, 245, 240),
                2,
            )
            y += row_h
    return vis
