"""
Shared helper functions for the CV engine worker.

This module keeps app/engine.py focused on loop orchestration and state flow.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.state import SharedState
from app.openvino_inference import OpenVINOTracker


_CV2_DRAW_AVAILABLE = all(
    hasattr(cv2, attr)
    for attr in ("putText", "line", "rectangle", "addWeighted", "getTextSize")
)
_CV2_ENCODE_AVAILABLE = hasattr(cv2, "imencode")


def load_model(*, state: SharedState, model_path: str, model_task: str | None):
    """Load the OpenVINO detection model and update shared status/event state."""
    try:
        # model_task is kept for config compatibility but not used by the runtime.
        model = OpenVINOTracker(model_path=model_path, device="CPU")
        with state.lock:
            state.status_text = f"Model loaded: {model_path}"
            state.model_ready = True
        state.add_event("model_loaded", {"model_path": model_path, "backend": "openvino+iou-tracker"})
        return model
    except Exception as exc:
        with state.lock:
            state.status_text = f"Model load failed ({exc})"
            state.model_ready = False
        state.add_event("model_load_failed", {"model_path": model_path, "error": str(exc)})
        return None


def open_video(state: SharedState, video_path: str) -> cv2.VideoCapture | None:
    """Open a video source and report structured failure events."""
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as exc:
        with state.lock:
            state.status_text = f"Video backend unavailable: {exc}"
        state.add_event("video_open_exception", {"video_path": video_path, "error": str(exc)})
        return None

    if not cap.isOpened():
        with state.lock:
            state.status_text = f"Cannot open video: {video_path}"
        state.add_event("video_open_failed", {"video_path": video_path})
        return None
    return cap


def put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw high-contrast text in-place."""
    if not _CV2_DRAW_AVAILABLE:
        return

    put_text_fn = cv2.putText

    put_text_fn(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    put_text_fn(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def placeholder_frame(*, message: str, width: int, height: int) -> np.ndarray:
    """Build a dark placeholder frame with one status line."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    put_text(frame, message, (20, int(height * 0.5)), 1.2, (200, 200, 200), 3)
    return frame


def draw_gate(
    frame: np.ndarray,
    *,
    active: bool,
    polylines: list[list[tuple[float, float]]],
    crossing_order: list[int],
    flash: bool = False,
) -> None:
    """Render tripwire lines on the frame in-place."""
    if not _CV2_DRAW_AVAILABLE:
        return

    if not polylines:
        return

    h, w = frame.shape[:2]
    colour = (0, 0, 255) if flash else ((0, 255, 0) if active else (0, 200, 200))
    ordered_lines = set(crossing_order)
    line_fn = cv2.line

    for index, polyline in enumerate(polylines, start=1):
        p1 = (int(polyline[0][0] * w), int(polyline[0][1] * h))
        p2 = (int(polyline[1][0] * w), int(polyline[1][1] * h))
        line_colour = colour if index in ordered_lines else (160, 160, 160)
        line_fn(frame, p1, p2, line_colour, 5)


def extract_tracks(tracked_objects) -> list[dict]:
    """Convert OpenVINOTracker TrackedObject list to counter track schema."""
    if not tracked_objects:
        return []

    return [
        {
            "id": obj.track_id,
            "bbox": (obj.x1, obj.y1, obj.x2, obj.y2)
        }
        for obj in tracked_objects
    ]


def annotate_detections(frame: np.ndarray, tracked_objects) -> None:
    """Draw bboxes and track IDs on frame in-place."""
    if not _CV2_DRAW_AVAILABLE or not tracked_objects:
        return

    rectangle_fn = cv2.rectangle
    
    for obj in tracked_objects:
        x1, y1, x2, y2 = int(obj.x1), int(obj.y1), int(obj.x2), int(obj.y2)
        
        # Draw bbox
        rectangle_fn(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw track ID
        label = f"ID:{obj.track_id} {obj.conf:.2f}"
        label_top = max(36, y1)
        label_width = max(160, len(label) * 14)
        rectangle_fn(frame, (x1, label_top - 36), (x1 + label_width, label_top), (0, 255, 0), -1)
        put_text(frame, label, (x1 + 8, label_top - 10), 0.95, (0, 0, 0), 2)


def open_output_writer(
    *,
    enabled: bool,
    output_path: str,
    codec: str,
    frame_w: int,
    frame_h: int,
    fps: float,
) -> cv2.VideoWriter | None:
    """Create output writer when enabled and available."""
    if not enabled:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        return None
    return writer
