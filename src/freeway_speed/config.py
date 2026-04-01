from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PerceptionConfig:
    vehicle_backend: str = "threshold"
    lane_backend: str = "threshold"
    yolo_model: str = "yolov8n.pt"
    yolo_classes: tuple[str, ...] = ("car", "bus", "truck")
    yolo_conf: float = 0.10
    yolo_iou: float = 0.45
    yolo_imgsz: int = 640
    vehicle_upscale_factor: float = 2.0
    lane_temporal_smoothing: bool = True
    lane_temporal_alpha: float = 0.05
    lane_onnx: str = ""


@dataclass
class IPMConfig:
    roi_ratio_from_bottom: float = 0.3
    ema_alpha: float = 0.2
    bev_width: int = 640
    bev_height: int = 720


@dataclass
class CurveFitConfig:
    min_points: int = 20
    bins: int = 40


@dataclass
class CalibrationConfig:
    dash_length_m: float = 4.0
    min_dash_pixels: int = 20
    max_dash_pixels: int = 180
    min_gap_pixels: int = 8
    min_dash_runs: int = 2
    search_band_px: int = 10
    lane_width_m: float = 3.5
    lane_hist_roi_start_ratio: float = 0.55
    lane_width_min_px: int = 120
    lane_width_max_px: int = 620
    lane_peak_min_votes: int = 12
    min_scale_m_per_px: float = 0.01
    max_scale_m_per_px: float = 0.08
    default_scale_m_per_px: float = 0.025


@dataclass
class TrackingConfig:
    tracker_backend: str = "bytetrack"
    bt_track_high_thresh: float = 0.15
    bt_track_low_thresh: float = 0.05
    bt_new_track_thresh: float = 0.2
    bt_track_buffer: int = 60
    bt_match_thresh: float = 0.9
    bt_fuse_score: bool = True
    min_dt_sec: float = 0.4
    speed_smooth_window: int = 5
    max_speed_kmh: float = 180.0


@dataclass
class RuntimeConfig:
    lane_update_every_n_frames: int = 5


@dataclass
class SystemConfig:
    perception: PerceptionConfig
    ipm: IPMConfig
    curve: CurveFitConfig
    calibration: CalibrationConfig
    tracking: TrackingConfig
    runtime: RuntimeConfig


def _get(data: dict[str, Any], key: str, default: Any) -> Any:
    value = data.get(key)
    if value is None:
        return default
    return value


def load_config(path: str | Path) -> SystemConfig:
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    perception_raw = raw.get("perception", {})
    ipm_raw = raw.get("ipm", {})
    curve_raw = raw.get("curve", {})
    calibration_raw = raw.get("calibration", {})
    tracking_raw = raw.get("tracking", {})
    runtime_raw = raw.get("runtime", {})

    lane_onnx_path = _get(perception_raw, "lane_onnx", "")
    if lane_onnx_path:
        lane_onnx_path = str((cfg_path.parent / lane_onnx_path).resolve())

    return SystemConfig(
        perception=PerceptionConfig(
            vehicle_backend=_get(perception_raw, "vehicle_backend", "threshold"),
            lane_backend=_get(perception_raw, "lane_backend", "threshold"),
            yolo_model=_get(perception_raw, "yolo_model", "yolov8n.pt"),
            yolo_classes=tuple(_get(perception_raw, "yolo_classes", ["car", "bus", "truck"])),
            yolo_conf=float(_get(perception_raw, "yolo_conf", 0.10)),
            yolo_iou=float(_get(perception_raw, "yolo_iou", 0.45)),
            yolo_imgsz=int(_get(perception_raw, "yolo_imgsz", 640)),
            vehicle_upscale_factor=float(_get(perception_raw, "vehicle_upscale_factor", 2.0)),
            lane_temporal_smoothing=bool(_get(perception_raw, "lane_temporal_smoothing", True)),
            lane_temporal_alpha=float(_get(perception_raw, "lane_temporal_alpha", 0.05)),
            lane_onnx=lane_onnx_path,
        ),
        ipm=IPMConfig(
            roi_ratio_from_bottom=float(_get(ipm_raw, "roi_ratio_from_bottom", 0.3)),
            ema_alpha=float(_get(ipm_raw, "ema_alpha", 0.2)),
            bev_width=int(_get(ipm_raw, "bev_width", 640)),
            bev_height=int(_get(ipm_raw, "bev_height", 720)),
        ),
        curve=CurveFitConfig(
            min_points=int(_get(curve_raw, "min_points", 20)),
            bins=int(_get(curve_raw, "bins", 40)),
        ),
        calibration=CalibrationConfig(
            dash_length_m=float(_get(calibration_raw, "dash_length_m", 4.0)),
            min_dash_pixels=int(_get(calibration_raw, "min_dash_pixels", 20)),
            max_dash_pixels=int(_get(calibration_raw, "max_dash_pixels", 180)),
            min_gap_pixels=int(_get(calibration_raw, "min_gap_pixels", 8)),
            min_dash_runs=int(_get(calibration_raw, "min_dash_runs", 2)),
            search_band_px=int(_get(calibration_raw, "search_band_px", 10)),
            lane_width_m=float(_get(calibration_raw, "lane_width_m", 3.5)),
            lane_hist_roi_start_ratio=float(_get(calibration_raw, "lane_hist_roi_start_ratio", 0.55)),
            lane_width_min_px=int(_get(calibration_raw, "lane_width_min_px", 120)),
            lane_width_max_px=int(_get(calibration_raw, "lane_width_max_px", 620)),
            lane_peak_min_votes=int(_get(calibration_raw, "lane_peak_min_votes", 12)),
            min_scale_m_per_px=float(_get(calibration_raw, "min_scale_m_per_px", 0.01)),
            max_scale_m_per_px=float(_get(calibration_raw, "max_scale_m_per_px", 0.08)),
            default_scale_m_per_px=float(_get(calibration_raw, "default_scale_m_per_px", 0.025)),
        ),
        tracking=TrackingConfig(
            tracker_backend=_get(tracking_raw, "tracker_backend", "bytetrack"),
            bt_track_high_thresh=float(_get(tracking_raw, "bt_track_high_thresh", 0.15)),
            bt_track_low_thresh=float(_get(tracking_raw, "bt_track_low_thresh", 0.05)),
            bt_new_track_thresh=float(_get(tracking_raw, "bt_new_track_thresh", 0.2)),
            bt_track_buffer=int(_get(tracking_raw, "bt_track_buffer", 60)),
            bt_match_thresh=float(_get(tracking_raw, "bt_match_thresh", 0.9)),
            bt_fuse_score=bool(_get(tracking_raw, "bt_fuse_score", True)),
            min_dt_sec=float(_get(tracking_raw, "min_dt_sec", 0.4)),
            speed_smooth_window=int(_get(tracking_raw, "speed_smooth_window", 5)),
            max_speed_kmh=float(_get(tracking_raw, "max_speed_kmh", 180.0)),
        ),
        runtime=RuntimeConfig(
            lane_update_every_n_frames=int(_get(runtime_raw, "lane_update_every_n_frames", 5)),
        ),
    )
