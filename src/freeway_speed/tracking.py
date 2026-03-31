from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackingConfig
from .types import Detection, TrackObservation


@dataclass
class _Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    class_name: str
    score: float
    missed: int = 0
    history: deque[TrackObservation] = field(default_factory=lambda: deque(maxlen=64))
    speed_hist: deque[float] = field(default_factory=lambda: deque(maxlen=10))

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


class SimpleByteLikeTracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self.max_distance_px = float(getattr(cfg, "max_distance_px", 80.0))
        self.max_missed_frames = int(getattr(cfg, "max_missed_frames", 15))

    def _new_track(self, det: Detection) -> _Track:
        tr = _Track(
            track_id=self._next_id,
            bbox=det.bbox,
            class_name=det.class_name,
            score=det.score,
            speed_hist=deque(maxlen=self.cfg.speed_smooth_window),
        )
        self._tracks[self._next_id] = tr
        self._next_id += 1
        return tr

    def update(self, detections: list[Detection]) -> dict[int, _Track]:
        if not self._tracks:
            for det in detections:
                self._new_track(det)
            return self._tracks

        track_ids = list(self._tracks.keys())
        track_centers = np.stack([self._tracks[tid].center for tid in track_ids], axis=0)
        det_centers = np.asarray(
            [[(d.bbox[0] + d.bbox[2]) * 0.5, (d.bbox[1] + d.bbox[3]) * 0.5] for d in detections],
            dtype=np.float32,
        )

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if len(detections) > 0 and len(track_ids) > 0:
            cost = np.linalg.norm(track_centers[:, None, :] - det_centers[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] > self.max_distance_px:
                    continue
                tid = track_ids[r]
                det = detections[c]
                tr = self._tracks[tid]
                tr.bbox = det.bbox
                tr.class_name = det.class_name
                tr.score = det.score
                tr.missed = 0
                matched_tracks.add(tid)
                matched_dets.add(c)

        for tid, tr in list(self._tracks.items()):
            if tid not in matched_tracks:
                tr.missed += 1
            if tr.missed > self.max_missed_frames:
                del self._tracks[tid]

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._new_track(det)

        return self._tracks


class _ByteTrackResults:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        xyxy_arr = np.asarray(xyxy, dtype=np.float32)
        conf_arr = np.asarray(conf, dtype=np.float32).reshape(-1)
        cls_arr = np.asarray(cls, dtype=np.float32).reshape(-1)

        if xyxy_arr.size == 0:
            xyxy_arr = np.zeros((0, 4), dtype=np.float32)
        else:
            xyxy_arr = xyxy_arr.reshape(-1, 4)

        n = min(int(xyxy_arr.shape[0]), int(conf_arr.shape[0]), int(cls_arr.shape[0]))
        self.xyxy = xyxy_arr[:n]
        self.conf = conf_arr[:n]
        self.cls = cls_arr[:n]

        if len(self.xyxy) > 0:
            w = self.xyxy[:, 2] - self.xyxy[:, 0]
            h = self.xyxy[:, 3] - self.xyxy[:, 1]
            cx = self.xyxy[:, 0] + w * 0.5
            cy = self.xyxy[:, 1] + h * 0.5
            self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __getitem__(self, idx: Any) -> _ByteTrackResults:
        return _ByteTrackResults(self.xyxy[idx], self.conf[idx], self.cls[idx])


class ByteTrackTracker:
    def __init__(self, cfg: TrackingConfig, frame_rate: int = 30):
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as e:
            raise RuntimeError("ultralytics is required for ByteTrack backend") from e

        args = SimpleNamespace(
            track_high_thresh=cfg.bt_track_high_thresh,
            track_low_thresh=cfg.bt_track_low_thresh,
            new_track_thresh=cfg.bt_new_track_thresh,
            track_buffer=cfg.bt_track_buffer,
            match_thresh=cfg.bt_match_thresh,
            fuse_score=cfg.bt_fuse_score,
        )
        self._tracker = BYTETracker(args=args, frame_rate=frame_rate)
        self._id_to_track: dict[int, _Track] = {}
        self._class_to_id: dict[str, int] = {}
        self._id_to_class: dict[int, str] = {}

    def _encode_classes(self, detections: list[Detection]) -> np.ndarray:
        class_ids: list[int] = []
        for det in detections:
            if det.class_name not in self._class_to_id:
                cls_id = len(self._class_to_id)
                self._class_to_id[det.class_name] = cls_id
                self._id_to_class[cls_id] = det.class_name
            class_ids.append(self._class_to_id[det.class_name])
        return np.asarray(class_ids, dtype=np.float32)

    def update(self, detections: list[Detection], frame: np.ndarray | None = None) -> dict[int, _Track]:
        if detections:
            xyxy = np.asarray([d.bbox for d in detections], dtype=np.float32)
            conf = np.asarray([d.score for d in detections], dtype=np.float32)
            cls = self._encode_classes(detections)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)

        results = _ByteTrackResults(xyxy=xyxy, conf=conf, cls=cls)
        tracked = self._tracker.update(results=results, img=frame)

        alive: dict[int, _Track] = {}
        for row in tracked:
            x1, y1, x2, y2, tid, score, cls_id, _idx = row.tolist()
            track_id = int(tid)
            cls_name = self._id_to_class.get(int(cls_id), "vehicle")

            prev = self._id_to_track.get(track_id)
            if prev is None:
                prev = _Track(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    class_name=cls_name,
                    score=score,
                )
            prev.bbox = (x1, y1, x2, y2)
            prev.class_name = cls_name
            prev.score = float(score)
            prev.missed = 0
            self._id_to_track[track_id] = prev
            alive[track_id] = prev

        self._id_to_track = alive
        return alive
