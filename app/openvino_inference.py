"""
app/openvino_inference.py
-------------------------
OpenVINO-based object detection with lightweight IoU tracking.

This module provides a unified inference interface using:
- OpenVINO Core for exported detection-model inference
- Simple IoU-based tracking with no external tracking dependency
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
try:
    from openvino import Core
except ImportError:
    import openvino as ov

    Core = ov.Core
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result from OpenVINO inference."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int


@dataclass
class TrackedObject:
    """Tracked object with temporal continuity."""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    frame_id: int


class STrack:
    """A simple track object for tracking."""
    
    next_id = 1
    
    def __init__(self, tlbr: np.ndarray, conf: float, frame_id: int):
        """tlbr: [x1, y1, x2, y2], conf: confidence, frame_id: frame number"""
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.conf = conf
        self.frame_id = frame_id
        self.track_id = STrack.next_id
        STrack.next_id += 1
        self.time_since_update = 0
        self.hits = 1
        self.is_activated = False

    def update(self, tlbr: np.ndarray, conf: float, frame_id: int) -> None:
        """Update track with new detection."""
        self.tlbr = np.array(tlbr, dtype=np.float32)
        self.conf = conf
        self.frame_id = frame_id
        self.time_since_update = 0
        self.hits += 1
    
    def mark_missed(self) -> None:
        """Mark this track as having a missed detection."""
        self.time_since_update += 1


class SimpleTracker:
    """Simple Euclidean + IoU-based tracker without Kalman filtering."""
    
    def __init__(self, max_age: int = 30, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracked_stracks: list[STrack] = []
        self.frame_id = 0
    
    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, detections: np.ndarray, img_info: tuple) -> list[STrack]:
        """Update tracker with detections. detections: Nx5 [x1,y1,x2,y2,conf]"""
        self.frame_id += 1
        if len(self.tracked_stracks) > 0 and len(detections) > 0:
            ious = np.zeros((len(self.tracked_stracks), len(detections)))
            for i, track in enumerate(self.tracked_stracks):
                for j, det in enumerate(detections):
                    ious[i, j] = self._iou(track.tlbr, det[:4])
            row_ind, col_ind = linear_sum_assignment(-ious)
            matched = set()
            for i, j in zip(row_ind, col_ind):
                if ious[i, j] > self.iou_threshold:
                    self.tracked_stracks[i].update(detections[j][:4], detections[j][4], self.frame_id)
                    matched.add(j)
            for i in range(len(self.tracked_stracks)):
                if i not in row_ind:
                    self.tracked_stracks[i].mark_missed()
            for j in range(len(detections)):
                if j not in matched:
                    new_track = STrack(detections[j][:4], detections[j][4], self.frame_id)
                    new_track.is_activated = True
                    self.tracked_stracks.append(new_track)
        else:
            for det in detections:
                new_track = STrack(det[:4], det[4], self.frame_id)
                new_track.is_activated = True
                self.tracked_stracks.append(new_track)
        active = [t for t in self.tracked_stracks if t.time_since_update < self.max_age]
        self.tracked_stracks = active

        # Only expose tracks that were refreshed by a detection on this frame.
        # Otherwise stale boxes can remain visible after confidence filtering
        # removes the underlying detections, which makes runtime conf changes
        # look ineffective.
        return [t for t in active if t.time_since_update == 0 and t.hits >= self.min_hits]
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracked_stracks = []
        self.frame_id = 0
        STrack.next_id = 1


class OpenVINODetector:
    """
    Detection-model runtime using OpenVINO.
    
    Supports:
    - OpenVINO IR format (.xml/.bin)
    - Exported detection models with YOLO-style output tensors
    - Configurable confidence and IoU thresholds
    - Class filtering
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        device: str = "CPU",
    ):
        """
        Initialize the OpenVINO detector runtime.

        Args:
            model_path: Path to OpenVINO IR (.xml) or ONNX model
            conf_threshold: Detection confidence threshold [0, 1]
            iou_threshold: NMS IoU threshold [0, 1]
            classes: List of class IDs to detect (None = all classes)
            device: OpenVINO device ("CPU", "GPU", etc.)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes or []
        self.device = device

        self.core = Core()
        self.model = None
        self.net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = []

        # Load model
        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load and compile OpenVINO model."""
        path = Path(model_path)

        # If path is a directory, look for best.xml or model.xml inside
        if path.is_dir():
            for xml_file in ['best.xml', 'model.xml']:
                candidate = path / xml_file
                if candidate.exists():
                    xml_path = candidate
                    break
            else:
                raise FileNotFoundError(f"No .xml model found in directory: {path}")
        else:
            # Support both .xml (IR format) and .onnx
            xml_path = path.with_suffix(".xml") if path.suffix != ".xml" else path
        
        if not xml_path.exists():
            raise FileNotFoundError(f"Model not found: {xml_path}")

        logger.info(f"Loading OpenVINO model from {xml_path}")
        self.model = self.core.read_model(str(xml_path))
        self.net = self.core.compile_model(self.model, self.device)

        # Get input/output info
        input_layer = self.net.input(0)
        try:
            self.input_name = input_layer.any_name
        except:
            self.input_name = "input"
        self.input_shape = input_layer.shape
        
        # Typically shape is [1, 3, 640, 640]
        _, _, self.img_h, self.img_w = self.input_shape

        output_layers = self.net.outputs

        # Get output names (fallback to indices if names not available)
        self.output_names = []
        for i, out in enumerate(output_layers):
            try:
                name = out.any_name
            except:
                name = f"output_{i}"
            self.output_names.append(name)
        logger.info(
            f"Model loaded. Input shape: {self.input_shape}, "
            f"Output layers: {len(output_layers)}"
        )

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, int, int]:
        """
        Preprocess frame for detection inference.

        Returns:
            (input_blob, scale_x, scale_y, pad_w, pad_h) where scale factors
            map model-space coordinates back into the original frame.
        """
        h, w = frame.shape[:2]

        # Letterbox: resize with aspect ratio preservation
        scale = min(self.img_h / h, self.img_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to target size
        img = np.full((self.img_h, self.img_w, 3), 114, dtype=np.uint8)
        pad_h = (self.img_h - new_h) // 2
        pad_w = (self.img_w - new_w) // 2
        img[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # Normalize: [0, 255] -> [0, 1] and convert to CHW
        blob = img.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)  # Add batch dimension

        # Inverse scale factors for unpacking detections back into the source
        # frame. Using input_size/new_size is incorrect for letterboxed inputs
        # and shifts boxes on non-square frames.
        scale_x = w / new_w if new_w > 0 else 1.0
        scale_y = h / new_h if new_h > 0 else 1.0

        return blob, scale_x, scale_y, pad_w, pad_h

    def _postprocess(
        self,
        predictions: np.ndarray,
        frame_h: int,
        frame_w: int,
        scale_x: float,
        scale_y: float,
        pad_w: int,
        pad_h: int,
    ) -> list[Detection]:
        """
        Postprocess model output to detections.

        Expected output format: [1, 84, 8400]
        - First 4 values: x, y, w, h (center-based)
        - Next 80 values: class confidences
        - Max confidence determines class_id
        """
        detections = []

        # predictions shape: [1, 84, 8400] -> transpose to [1, 8400, 84]
        if predictions.ndim == 3 and predictions.shape[0] == 1:
            predictions = predictions[0].T  # [8400, 84]
        else:
            predictions = predictions.T

        for pred in predictions:
            if len(pred) <= 4:
                continue

            # Support both common YOLO export layouts:
            # 1) [x, y, w, h, cls0, cls1, ...] (no explicit objectness)
            # 2) [x, y, w, h, obj, cls0, cls1, ...] (explicit objectness)
            scores_no_obj = pred[4:]
            class_id_no_obj = int(np.argmax(scores_no_obj))
            conf_no_obj = float(scores_no_obj[class_id_no_obj])

            conf_obj = -1.0
            class_id_obj = 0
            if len(pred) > 5:
                scores_obj = pred[5:]
                if len(scores_obj) > 0:
                    class_id_obj = int(np.argmax(scores_obj))
                    conf_obj = float(pred[4] * scores_obj[class_id_obj])

            if conf_obj > conf_no_obj:
                conf = conf_obj
                class_id = class_id_obj
            else:
                conf = conf_no_obj
                class_id = class_id_no_obj

            # Filter by class if specified
            if self.classes and class_id not in self.classes:
                continue

            if conf < self.conf_threshold:
                continue

            # Decode bbox (x_center, y_center, w, h) -> (x1, y1, x2, y2)
            x_c, y_c, w, h = pred[:4]

            # Remove padding
            x_c = (x_c - pad_w) * scale_x
            y_c = (y_c - pad_h) * scale_y
            w = w * scale_x
            h = h * scale_y

            x1 = x_c - w / 2.0
            y1 = y_c - h / 2.0
            x2 = x_c + w / 2.0
            y2 = y_c + h / 2.0

            # Clip to frame bounds
            x1 = max(0.0, min(float(frame_w), x1))
            y1 = max(0.0, min(float(frame_h), y1))
            x2 = max(0.0, min(float(frame_w), x2))
            y2 = max(0.0, min(float(frame_h), y2))

            if x2 > x1 and y2 > y1:
                detections.append(
                    Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        conf=conf, class_id=class_id
                    )
                )

        # NMS (non-maximum suppression)
        if detections:
            detections = self._nms(detections)

        return detections

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        """
        Non-maximum suppression to remove overlapping detections.
        """
        if not detections:
            return detections

        # cv2.dnn.NMSBoxes expects [x, y, w, h], not [x1, y1, x2, y2].
        boxes = np.array([
            [d.x1, d.y1, max(0.0, d.x2 - d.x1), max(0.0, d.y2 - d.y1)]
            for d in detections
        ])
        scores = np.array([d.conf for d in detections])

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )

        if len(indices) == 0:
            return []

        return [detections[i] for i in indices.flatten()]

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect objects in a frame.

        Args:
            frame: Input image (BGR, uint8, any size)

        Returns:
            List of Detection objects
        """
        frame_h, frame_w = frame.shape[:2]

        # Preprocess
        blob, scale_x, scale_y, pad_w, pad_h = self._preprocess(frame)

        # Infer
        outputs = self.net([blob])

        # Resolve first output robustly because OpenVINO output keys can vary
        # across versions (e.g. tensor objects vs string names like output_0).
        try:
            predictions = outputs[self.output_names[0]]
        except Exception:
            predictions = None
            try:
                predictions = outputs[self.net.output(0)]
            except Exception:
                pass

            if predictions is None:
                if isinstance(outputs, dict):
                    predictions = next(iter(outputs.values()), None)
                elif isinstance(outputs, (list, tuple)) and outputs:
                    predictions = outputs[0]

            if predictions is None:
                raise RuntimeError("OpenVINO inference returned no outputs")

        # Postprocess
        detections = self._postprocess(
            predictions,
            frame_h,
            frame_w,
            scale_x,
            scale_y,
            pad_w,
            pad_h,
        )

        return detections


class OpenVINOTracker:
    """
    Combined OpenVINO detection + simple IoU-based tracking pipeline.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: list[int] | None = None,
        device: str = "CPU",
        track_max_age: int = 30,
        track_min_hits: int = 1,
        track_iou_threshold: float = 0.3,
    ):
        """
        Initialize detector + simple tracker.

        Args:
            model_path: Path to OpenVINO model
            conf_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            classes: Class IDs to detect (None = all)
            device: OpenVINO device
            track_max_age: Max frames to keep track without detection
            track_min_hits: Min detections before track is active
            track_iou_threshold: IoU threshold for track matching
        """
        self.detector = OpenVINODetector(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
            device=device,
        )
        
        self.tracker = SimpleTracker(
            max_age=track_max_age,
            min_hits=track_min_hits,
            iou_threshold=track_iou_threshold,
        )
        
        self.frame_id = 0

    def __call__(self, frame: np.ndarray) -> list[TrackedObject]:
        """
        Detect and track objects in a frame.

        Args:
            frame: Input image (BGR, uint8)

        Returns:
            List of TrackedObject with track_id assignments
        """
        self.frame_id += 1

        # Detect
        detections = self.detector(frame)

        # Convert to tracker format: [x1, y1, x2, y2, conf]
        if detections:
            dets = np.array([
                [d.x1, d.y1, d.x2, d.y2, d.conf]
                for d in detections
            ])
        else:
            dets = np.empty((0, 5))

        # Track
        tracked_stracks = self.tracker.update(dets, img_info=frame.shape)

        # Convert back to TrackedObject format
        tracked_objects = []
        for strack in tracked_stracks:
            bbox = strack.tlbr
            tracked_objects.append(
                TrackedObject(
                    track_id=int(strack.track_id),
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[2]),
                    y2=float(bbox[3]),
                    conf=float(strack.conf),
                    class_id=0,
                    frame_id=self.frame_id,
                )
            )

        return tracked_objects

    def reset(self) -> None:
        """Reset tracker state (e.g., on video restart)."""
        self.frame_id = 0
        self.tracker.reset()


# Backward-compatible aliases for older imports.
OpenVINOYOLODetector = OpenVINODetector
YOLOTracker = OpenVINOTracker
