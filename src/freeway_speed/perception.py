from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from .types import Detection


class VehicleDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


@dataclass
class ThresholdVehicleDetector:
    min_area: int = 300

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if aspect < 0.6 or aspect > 4.5:
                continue
            detections.append(
                Detection(
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                    score=min(1.0, area / 5000.0),
                    class_name="vehicle",
                )
            )
        return detections


class YoloV8Detector:
    def __init__(
        self,
        model_path: str,
        classes: tuple[str, ...],
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        iou_threshold: float = 0.45,
        upscale_factor: float = 1.0,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("ultralytics is required for YOLO backend") from e

        self.model = YOLO(model_path)
        self.classes = set(classes)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.iou_threshold = iou_threshold
        self.upscale_factor = max(1.0, float(upscale_factor))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        if self.upscale_factor > 1.0:
            resized_w = max(1, int(round(w * self.upscale_factor)))
            resized_h = max(1, int(round(h * self.upscale_factor)))
            infer_frame = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            infer_frame = frame

        result = self.model(
            infer_frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
        )[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = str(names.get(cls_id, cls_id))
            if cls_name not in self.classes:
                continue
            score = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            if self.upscale_factor > 1.0:
                x1 /= self.upscale_factor
                y1 /= self.upscale_factor
                x2 /= self.upscale_factor
                y2 /= self.upscale_factor
                x1 = float(np.clip(x1, 0, max(0, w - 1)))
                y1 = float(np.clip(y1, 0, max(0, h - 1)))
                x2 = float(np.clip(x2, 0, max(0, w - 1)))
                y2 = float(np.clip(y2, 0, max(0, h - 1)))
            detections.append(Detection(bbox=(x1, y1, x2, y2), score=score, class_name=cls_name))
        return detections


class LaneSegmenter(Protocol):
    def segment(self, frame: np.ndarray) -> np.ndarray: ...


@dataclass
class ThresholdLaneSegmenter:
    def segment(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        grad_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)

        _, bright = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
        _, edge = cv2.threshold(grad_x, 40, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(bright, edge)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask


class ONNXLaneSegmenter:
    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("onnxruntime is required for ONNX lane backend") from e
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name

        # Try to respect the model static input shape; fallback to 640x640.
        shape = input_meta.shape
        in_h = shape[2] if len(shape) > 2 else 640
        in_w = shape[3] if len(shape) > 3 else 640
        self.input_h = int(in_h) if isinstance(in_h, int) else 640
        self.input_w = int(in_w) if isinstance(in_w, int) else 640

    @staticmethod
    def _to_prob_map(arr: np.ndarray) -> np.ndarray | None:
        tensor = np.asarray(arr)

        if tensor.ndim == 4:
            tensor = tensor[0]

        if tensor.ndim == 3:
            c, h, w = tensor.shape
            if c <= 0:
                return None
            # Prefer last channel for lane-like logits when multiple classes exist.
            plane = tensor[c - 1]
            if plane.max() > 1.0 or plane.min() < 0.0:
                plane = 1.0 / (1.0 + np.exp(-plane))
            return plane.astype(np.float32)

        if tensor.ndim == 2:
            plane = tensor.astype(np.float32)
            if plane.max() > 1.0 or plane.min() < 0.0:
                plane = 1.0 / (1.0 + np.exp(-plane))
            return plane

        return None

    def _extract_lane_mask(self, outputs: list[np.ndarray], out_h: int, out_w: int) -> np.ndarray:
        best_map: np.ndarray | None = None
        best_score = -1.0

        for out in outputs:
            prob = self._to_prob_map(out)
            if prob is None:
                continue
            # Prefer maps that have enough activated area but are not all white.
            score = float(np.mean(prob))
            if score > best_score:
                best_score = score
                best_map = prob

        if best_map is None:
            return np.zeros((out_h, out_w), dtype=np.uint8)

        lane = cv2.resize(best_map, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        lane = (lane > 0.5).astype(np.uint8) * 255
        return lane

    def segment(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]

        outputs = self.session.run(None, {self.input_name: tensor})
        return self._extract_lane_mask(outputs, h, w)
