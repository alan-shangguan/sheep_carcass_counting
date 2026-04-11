"""
app/engine.py
-------------
Background worker thread that drives the CV pipeline.

Architecture
~~~~~~~~~~~~
run_engine() is started once as a daemon thread by app/main.py.
It owns the video capture handle and the YOLO model instance.
All communication with the FastAPI layer goes through SharedState
(app/state.py).  No FastAPI internals are imported here.

Thread-safety contract
~~~~~~~~~~~~~~~~~~~~~~
* `state.lock` is held for the *minimum* time necessary – never across
  YOLO inference or JPEG encoding (both can take tens of milliseconds).
* The engine is the only thread that writes `state.latest_jpeg`,
  `state.count`, `state.track_memory`, and `state.status_text`.
* The API layer writes `state.running` and `state.reset_requested`.
  The engine reads these flags under the lock, then operates on its own
  local copies for the rest of the loop iteration.

Reset protocol
~~~~~~~~~~~~~~
The engine checks `state.reset_requested` at the top of each iteration.
When True it:
  1. Zeroes `state.count`.
  2. Clears `state.track_memory` (sufficient to reset counter logic).
  3. Sets `state.reset_requested = False`  (momentary – not latched).
The YOLO tracker retains its internal ID sequence; after the memory
clear, any re-appearing track_id is simply treated as a fresh object,
which is the correct behaviour for a new session.

Configuration
~~~~~~~~~~~~~
All tuneable values are exposed via config.yaml, with optional environment
variable overrides for quick local experiments.  The engine can also save
an annotated output video so counting results can be reviewed offline.

Set ad-hoc overrides before starting uvicorn, e.g.:

    VIDEO_PATH=footage/run1.mp4 MODEL_PATH=weights/best.pt uvicorn app.main:app
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.config import load_config
from app.openvino_inference import OpenVINOTracker
from app.runtime_logging import log_event
from app.state import SharedState
from app import counter as counter_module

# ---------------------------------------------------------------------------
# Configuration – loaded from config.yaml, with optional env overrides
# ---------------------------------------------------------------------------

CONFIG = load_config()

# Environment variables still win when explicitly provided, which is useful
# for quick ad-hoc debugging without editing config.yaml.
VIDEO_PATH: str = os.environ.get("VIDEO_PATH", CONFIG.video.path)
MODEL_PATH: str = os.environ.get("MODEL_PATH", CONFIG.model.path)
TRIPWIRE_POLYLINES: list[list[tuple[float, float]]] = CONFIG.counter.polylines
TRIPWIRE_CROSSING_DIRECTIONS: list[str] = CONFIG.counter.crossing_directions
TRIPWIRE_CROSSING_ORDER: list[int] = CONFIG.counter.crossing_order
TRIPWIRE_UNIT: str = CONFIG.counter.unit
TRIPWIRE_ANCHOR_POINT: str = CONFIG.counter.anchor_point
TRIPWIRE_STATE_THRESHOLD: int = CONFIG.counter.state_threshold
MIN_HITS: int = int(os.environ.get("MIN_HITS", str(CONFIG.counter.min_hits)))
MODEL_CLASSES: list[int] | None = CONFIG.model.classes
SKIP_FRAME: int = max(1, int(CONFIG.model.skip_frame))
MODEL_CONF_THRESHOLD: float = float(CONFIG.model.conf_threshold)
MODEL_IOU_THRESHOLD: float = float(CONFIG.model.iou_threshold)
JPEG_QUALITY: int = int(os.environ.get("JPEG_QUALITY", str(CONFIG.stream.jpeg_quality)))
OUTPUT_VIDEO_ENABLED: bool = CONFIG.output_video.enabled
OUTPUT_VIDEO_PATH_TEMPLATE: str = os.environ.get("OUTPUT_VIDEO_PATH", CONFIG.output_video.path)
OUTPUT_VIDEO_CODEC: str = CONFIG.output_video.codec
OUTPUT_VIDEO_FPS: float | None = CONFIG.output_video.fps
OUTPUT_VIDEO_WRITE_WHEN_PAUSED: bool = CONFIG.output_video.write_when_paused

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(state: SharedState):
    """
    Attempt to load an OpenVINO tracker model. Returns the model object on
    success, or None when the runtime or weights are unavailable.
    Updates state.status_text with the outcome.
    """
    try:
        model = OpenVINOTracker(
            model_path=MODEL_PATH,
            conf_threshold=MODEL_CONF_THRESHOLD,
            iou_threshold=MODEL_IOU_THRESHOLD,
            classes=MODEL_CLASSES,
            device="CPU",
            track_min_hits=1,
            track_iou_threshold=0.1,
        )
        with state.lock:
            state.status_text = f"Model loaded: {MODEL_PATH}"
            state.model_ready = True
        return model
    except Exception as exc:
        with state.lock:
            state.status_text = f"Model load failed ({exc}) – mock/preview mode"
            state.model_ready = False
        return None


def _open_video(state: SharedState, video_path: str) -> cv2.VideoCapture | None:
    """
    Try to open the configured video file.  Returns a VideoCapture object
    that isOpened(), or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with state.lock:
            state.status_text = f"Cannot open video: {video_path}"
        log_event("video_open_failed", video_path=video_path)
        return None
    return cap


def _reset_model_tracking(model: OpenVINOTracker | None, reason: str) -> None:
    if model is None:
        return
    try:
        model.reset()
        log_event("tracker_reset", reason=reason)
    except Exception as exc:
        log_event("tracker_reset_failed", reason=reason, error=str(exc))


def _placeholder_frame(message: str) -> np.ndarray:
    """Return a plain dark frame with a centred status message."""
    frame = np.zeros(
        (CONFIG.stream.placeholder_height, CONFIG.stream.placeholder_width, 3),
        dtype=np.uint8,
    )
    _put_text(frame, message, (20, 240), 1.2, (200, 200, 200), 3)
    return frame


def _put_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float,
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw high-contrast text for recorded output and MJPEG preview."""
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _draw_gate(
    frame: np.ndarray,
    active: bool,
    polylines: list[list[tuple[float, float]]],
    crossing_order: list[int],
    thickness: int,
    flash: bool = False,
) -> None:
    """Overlay configured ordered tripwire polylines on the frame in-place."""
    h, w = frame.shape[:2]
    colour = (0, 0, 255) if flash else ((0, 255, 0) if active else (0, 200, 200))

    for index, polyline in enumerate(polylines, start=1):
        p1 = (int(polyline[0][0] * w), int(polyline[0][1] * h))
        p2 = (int(polyline[1][0] * w), int(polyline[1][1] * h))
        line_colour = colour if index in crossing_order else (160, 160, 160)
        cv2.line(frame, p1, p2, line_colour, thickness)
        label_pos = (p1[0] + 4, max(20, p1[1] + 16))
        _put_text(frame, f"G{index}", label_pos, 0.9, line_colour, 2)


def _draw_config_overlay(frame: np.ndarray) -> None:
    """Render the active runtime configuration as a compact overlay."""
    lines = [
        f"Anchor: {TRIPWIRE_ANCHOR_POINT}",
        f"Dir: {'/'.join(TRIPWIRE_CROSSING_DIRECTIONS)}",
        f"Order: {TRIPWIRE_CROSSING_ORDER}",
        f"Conf/IoU: {MODEL_CONF_THRESHOLD:.2f}/{MODEL_IOU_THRESHOLD:.2f}",
        f"Skip: {SKIP_FRAME}",
        (
            "SizeSanity: "
            f"{'on' if CONFIG.counter.size_sanity_enabled else 'off'} "
            f"A[{CONFIG.counter.min_area_ratio:.3f},{CONFIG.counter.max_area_ratio:.3f}] "
            f"R[{CONFIG.counter.min_aspect_ratio:.2f},{CONFIG.counter.max_aspect_ratio:.2f}]"
        ),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    line_height = 28
    padding = 12
    max_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    box_width = max_width + padding * 2
    box_height = len(lines) * line_height + padding * 2

    x1 = max(10, frame.shape[1] - box_width - 10)
    y1 = 10
    x2 = x1 + box_width
    y2 = y1 + box_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

    for index, line in enumerate(lines):
        text_x = x1 + padding
        text_y = y1 + padding + 13 + index * line_height
        _put_text(frame, line, (text_x, text_y), font_scale, (255, 255, 255), thickness)


def _open_output_writer(
    output_path: Path,
    frame_w: int,
    frame_h: int,
    fps: float,
) -> cv2.VideoWriter | None:
    """Create the annotated output video writer, if enabled and possible."""
    if not OUTPUT_VIDEO_ENABLED:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        return None
    return writer


def _resolve_output_video_path(video_path: str) -> Path:
    """Derive output render path from configured template and active input video."""
    template = Path(OUTPUT_VIDEO_PATH_TEMPLATE)
    input_stem = Path(video_path).stem or "video"
    suffix = template.suffix or ".mp4"

    if "{video}" in str(template):
        return Path(str(template).replace("{video}", input_stem))

    base_dir = template.parent if str(template.parent) not in ("", ".") else Path("outputs")
    return base_dir / f"{input_stem}{suffix}"


def _extract_tracks(tracked_objects) -> list[dict]:
    """
    Convert OpenVINO tracker objects into the track schema expected by
    counter.process_frame().
    """
    if not tracked_objects:
        return []

    return [
        {
            "id": int(obj.track_id),
            "bbox": (float(obj.x1), float(obj.y1), float(obj.x2), float(obj.y2)),
        }
        for obj in tracked_objects
    ]


def _draw_tracked_objects(frame: np.ndarray, tracked_objects) -> None:
    """Draw tracked bounding boxes and IDs in-place for preview/output."""
    if not tracked_objects:
        return

    for obj in tracked_objects:
        x1, y1, x2, y2 = int(obj.x1), int(obj.y1), int(obj.x2), int(obj.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"ID:{int(obj.track_id)} {float(obj.conf):.2f}"
        label_top = max(24, y1)
        _put_text(frame, label, (x1 + 2, label_top - 4), 0.7, (0, 255, 0), 2)


def _passes_size_sanity(bbox: tuple[float, float, float, float], frame_w: int, frame_h: int) -> bool:
    """Return True when a tracked box falls inside the configured size/aspect envelope."""
    if not CONFIG.counter.size_sanity_enabled:
        return True

    x1, y1, x2, y2 = bbox
    box_w = max(0.0, float(x2) - float(x1))
    box_h = max(0.0, float(y2) - float(y1))
    if box_w <= 0.0 or box_h <= 0.0 or frame_w <= 0 or frame_h <= 0:
        return False

    width_ratio = box_w / float(frame_w)
    height_ratio = box_h / float(frame_h)
    area_ratio = (box_w * box_h) / float(frame_w * frame_h)
    aspect_ratio = box_w / box_h

    return (
        CONFIG.counter.min_width_ratio <= width_ratio <= CONFIG.counter.max_width_ratio
        and CONFIG.counter.min_height_ratio <= height_ratio <= CONFIG.counter.max_height_ratio
        and CONFIG.counter.min_area_ratio <= area_ratio <= CONFIG.counter.max_area_ratio
        and CONFIG.counter.min_aspect_ratio <= aspect_ratio <= CONFIG.counter.max_aspect_ratio
    )


def _filter_tracks_for_counting(tracks: list[dict], frame_w: int, frame_h: int) -> list[dict]:
    """Filter tracked objects using the configured size sanity envelope."""
    return [track for track in tracks if _passes_size_sanity(track["bbox"], frame_w, frame_h)]


# ---------------------------------------------------------------------------
# Main engine loop
# ---------------------------------------------------------------------------

def run_engine(state: SharedState) -> None:
    """
    Worker function – call once from a daemon thread.  Runs indefinitely.

    Layout of each loop iteration
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1.  Handle reset (brief lock).
    2.  Snapshot control flags (brief lock).
    3.  Ensure video capture is open; produce placeholder frame on error.
    4.  Read next frame; loop video on EOF.
    5.  When running:  run YOLO inference, extract tracks, call counter.
    6.  Annotate frame (gate line, status overlay, count display).
    7.  Encode to JPEG and store in shared state (brief lock).
    8.  Sleep to honour the source video's frame rate.
    """
    with state.lock:
        state.engine_thread_alive = True

    model = _load_model(state)
    cap: cv2.VideoCapture | None = None
    output_writer: cv2.VideoWriter | None = None
    output_writer_failed = False
    output_writer_released = False
    current_output_video_path: Path | None = None
    frame_index = 0
    current_video_path = VIDEO_PATH
    gate_flash_frames_remaining = 0
    last_crossing_text = ""
    last_inference_error = ""
    inference_runs = 0
    last_pause_state = False

    # Target frame interval; updated once the video is opened.
    target_frame_time: float = 1.0 / 25.0

    while True:
        t_iter_start = time.perf_counter()

        switch_to_video: str | None = None
        restart_video = False

        # ------------------------------------------------------------------ #
        # 1. Handle reset (momentary flag)                                    #
        # ------------------------------------------------------------------ #
        with state.lock:
            if state.reset_requested:
                state.count = 0
                state.track_memory.clear()
                state.reset_requested = False
                # Status reverts to the current running mode.
                state.status_text = "Running" if state.running else "Idle"
                _reset_model_tracking(model, "reset_requested")

        # ------------------------------------------------------------------ #
        # 2. Snapshot control flags                                            #
        # ------------------------------------------------------------------ #
        with state.lock:
            running = state.running
            video_paused = state.video_paused
            if state.requested_video_path and state.requested_video_path != current_video_path:
                switch_to_video = state.requested_video_path
                state.requested_video_path = None
            restart_video = bool(state.restart_video_requested)
            if state.restart_video_requested:
                state.restart_video_requested = False
            runtime_conf_threshold = float(state.runtime_conf_threshold)
            runtime_iou_threshold = float(state.runtime_iou_threshold)
            runtime_skip_frame = max(1, int(state.runtime_skip_frame))
            runtime_polylines = [list(polyline) for polyline in state.runtime_polylines] or TRIPWIRE_POLYLINES
            runtime_crossing_directions = list(state.runtime_crossing_directions) or TRIPWIRE_CROSSING_DIRECTIONS
            runtime_crossing_order = list(state.runtime_crossing_order) or TRIPWIRE_CROSSING_ORDER
            runtime_anchor_point = str(state.runtime_anchor_point).lower() or TRIPWIRE_ANCHOR_POINT
            runtime_min_hits = int(state.runtime_min_hits)
            runtime_state_threshold = int(state.runtime_state_threshold)
            runtime_reverse_decrease_counting = bool(state.runtime_reverse_decrease_counting)
            runtime_gate_thickness = max(1, int(state.runtime_gate_thickness))
            runtime_jpeg_quality = max(20, min(100, int(state.runtime_jpeg_quality)))

        if video_paused != last_pause_state:
            if video_paused:
                _reset_model_tracking(model, "video_paused")
                log_event("video_pause_applied", video_path=current_video_path, frame=frame_index)
            else:
                log_event("video_resume_applied", video_path=current_video_path, frame=frame_index)
            last_pause_state = video_paused

        if model is not None:
            model.detector.conf_threshold = runtime_conf_threshold
            model.detector.iou_threshold = runtime_iou_threshold

        if switch_to_video is not None:
            current_video_path = switch_to_video
            if cap is not None and cap.isOpened():
                cap.release()
            cap = None
            if output_writer is not None and not output_writer_released:
                output_writer.release()
            output_writer = None
            output_writer_failed = False
            output_writer_released = False
            current_output_video_path = None
            frame_index = 0
            inference_runs = 0
            _reset_model_tracking(model, "video_switched")
            with state.lock:
                state.current_video_path = current_video_path
                state.count = 0
                state.track_memory.clear()
                state.status_text = f"Switched video: {current_video_path}"
            log_event("video_switched", video_path=current_video_path)

        if restart_video and cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if output_writer is not None and not output_writer_released:
                output_writer.release()
            output_writer = None
            output_writer_failed = False
            output_writer_released = False
            current_output_video_path = None
            frame_index = 0
            inference_runs = 0
            _reset_model_tracking(model, "video_restarted")
            with state.lock:
                state.current_video_path = current_video_path
                state.count = 0
                state.track_memory.clear()
                state.status_text = f"Restarted video: {current_video_path}"
            log_event("video_restarted", video_path=current_video_path)

        if video_paused:
            overlay_lines = [
                "State: Paused",
                f"Video: {current_video_path}",
                f"Conf/IoU: {runtime_conf_threshold:.2f}/{runtime_iou_threshold:.2f}",
                f"SkipFrame: {runtime_skip_frame}",
                f"Anchor: {runtime_anchor_point}",
                f"Directions: {'/'.join(runtime_crossing_directions)}",
                f"Order: {runtime_crossing_order}",
                f"MinHits/StateThreshold: {runtime_min_hits}/{runtime_state_threshold}",
            ]
            if last_crossing_text:
                overlay_lines.append(last_crossing_text)
            if last_inference_error:
                overlay_lines.append(last_inference_error)
            with state.lock:
                # Keep the latest computed visual result visible while paused,
                # without reading new frames or running inference.
                if state.latest_jpeg is not None:
                    state.latest_frame_ts = time.time()
                state.frame_overlay_lines = overlay_lines
            time.sleep(min(target_frame_time, 0.05))
            continue

        # ------------------------------------------------------------------ #
        # 3. Ensure video capture is open                                     #
        # ------------------------------------------------------------------ #
        if cap is None or not cap.isOpened():
            cap = _open_video(state, current_video_path)
            if cap is None:
                # Emit a placeholder frame so /stream stays alive.
                ph = _placeholder_frame(f"Waiting for video: {current_video_path}")
                _, jpeg_buf = cv2.imencode(
                    ".jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, runtime_jpeg_quality]
                )
                with state.lock:
                    state.latest_jpeg = jpeg_buf.tobytes()
                    state.latest_frame_ts = time.time()
                    state.current_video_path = current_video_path
                log_event("video_waiting", video_path=current_video_path)
                time.sleep(1.0)
                continue
            with state.lock:
                state.current_video_path = current_video_path

            # Read FPS from the file; fall back to 25 if unavailable.
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0 and fps <= 120:
                target_frame_time = 1.0 / fps

        # ------------------------------------------------------------------ #
        # 4. Read next frame; loop at end-of-file                             #
        # ------------------------------------------------------------------ #
        ret, frame = cap.read()
        if not ret:
            if CONFIG.video.loop:
                _reset_model_tracking(model, "video_loop")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                log_event("video_loop", video_path=current_video_path)
            else:
                if output_writer is not None and not output_writer_released:
                    output_writer.release()
                    output_writer_released = True
                with state.lock:
                    if OUTPUT_VIDEO_ENABLED:
                        rendered_path = str(current_output_video_path) if current_output_video_path else "(not generated)"
                        state.status_text = f"Video ended. Output saved to {rendered_path}"
                    else:
                        state.status_text = f"Video ended: {current_video_path}"
                    state.current_video_path = current_video_path
                log_event(
                    "video_end",
                    video_path=current_video_path,
                    output_path=str(current_output_video_path) if current_output_video_path else None,
                )
                time.sleep(0.25)
            continue

        frame_index += 1

        frame_h, frame_w = frame.shape[:2]
        if output_writer is None and not output_writer_failed and not output_writer_released:
            current_output_video_path = _resolve_output_video_path(current_video_path)
            writer_fps = OUTPUT_VIDEO_FPS if OUTPUT_VIDEO_FPS and OUTPUT_VIDEO_FPS > 0 else (1.0 / target_frame_time)
            output_writer = _open_output_writer(current_output_video_path, frame_w, frame_h, writer_fps)
            if OUTPUT_VIDEO_ENABLED and output_writer is None:
                output_writer_failed = True
                with state.lock:
                    state.status_text = f"Output video open failed: {current_output_video_path}"
                log_event("output_writer_failed", output_path=str(current_output_video_path))
            elif output_writer is not None:
                log_event(
                    "output_writer_opened",
                    output_path=str(current_output_video_path),
                    fps=writer_fps,
                    frame_size=[frame_w, frame_h],
                )

        # ------------------------------------------------------------------ #
        # 5. Inference + counting (no lock held across inference)              #
        # ------------------------------------------------------------------ #
        count_increment = 0
        inference_latency_ms: float | None = None

        if running:
            if model is not None:
                try:
                    # Run detection/tracking on one frame out of every SKIP_FRAME.
                    if (frame_index - 1) % runtime_skip_frame == 0:
                        infer_start = time.perf_counter()
                        inference_runs += 1
                        tracked_objects = model(frame)
                        inference_latency_ms = (time.perf_counter() - infer_start) * 1000.0
                        if tracked_objects:
                            _draw_tracked_objects(frame, tracked_objects)

                        # Extract track list and run gate counter.
                        raw_tracks = _extract_tracks(tracked_objects)
                        tracks = _filter_tracks_for_counting(raw_tracks, frame_w, frame_h)
                        count_increment = counter_module.process_frame(
                            tracks,
                            state.track_memory,   # mutated in-place; engine owns it
                            frame_w,
                            frame_h,
                            polylines=runtime_polylines,
                            crossing_directions=runtime_crossing_directions,
                            crossing_order=runtime_crossing_order,
                            unit=TRIPWIRE_UNIT,
                            anchor_point=runtime_anchor_point,
                            min_hits=runtime_min_hits,
                            state_threshold=runtime_state_threshold,
                        )
                        if count_increment < 0 and not runtime_reverse_decrease_counting:
                            count_increment = 0
                        log_event(
                            "inference",
                            frame=frame_index,
                            inference_run=inference_runs,
                            raw_tracks=len(raw_tracks),
                            filtered_tracks=len(tracks),
                            count_increment=count_increment,
                            conf_threshold=runtime_conf_threshold,
                            iou_threshold=runtime_iou_threshold,
                            video_path=current_video_path,
                        )
                        if count_increment != 0:
                            last_event = state.track_memory.get("__last_event__", {})
                            event_uuid = str(last_event.get("object_uuid", "unknown"))[:8]
                            event_delta = int(last_event.get("delta", count_increment))
                            event_first = str(last_event.get("first_zone", "?"))
                            event_last = str(last_event.get("last_zone", "?"))
                            sign = "+" if event_delta > 0 else ""
                            last_crossing_text = (
                                f"Cross detected {sign}{event_delta} UUID {event_uuid} "
                                f"{event_first}->{event_last}"
                            )
                            gate_flash_frames_remaining = max(2, SKIP_FRAME)
                            log_event("crossing", video_path=current_video_path, **last_event)
                except Exception as exc:
                    # Inference errors must not crash the stream.
                    last_inference_error = f"Inference error: {exc}"
                    log_event("inference_error", frame=frame_index, error=str(exc), video_path=current_video_path)
            else:
                # Mock mode: no inference, no counting.
                last_inference_error = "Mock mode - no model"

        # ------------------------------------------------------------------ #
        # 6. Annotate frame                                                    #
        # ------------------------------------------------------------------ #

        # Gate line (green when running, cyan when paused).
        gate_flash_active = gate_flash_frames_remaining > 0
        _draw_gate(
            frame,
            running,
            runtime_polylines,
            runtime_crossing_order,
            runtime_gate_thickness,
            flash=gate_flash_active,
        )

        # Count overlay (always visible).
        with state.lock:
            state.count += count_increment
            count_display = state.count

        if frame_index % max(30, runtime_skip_frame * 10) == 0:
            log_event(
                "heartbeat",
                frame=frame_index,
                running=running,
                count=count_display,
                inference_runs=inference_runs,
                video_path=current_video_path,
                video_paused=video_paused,
            )

        overlay_lines = [
            f"State: {'Running' if running else 'Paused'}",
            f"Video: {current_video_path}",
            f"Conf/IoU: {runtime_conf_threshold:.2f}/{runtime_iou_threshold:.2f}",
            f"SkipFrame: {runtime_skip_frame}",
            f"Anchor: {runtime_anchor_point}",
            f"Directions: {'/'.join(runtime_crossing_directions)}",
            f"Order: {runtime_crossing_order}",
            f"MinHits/StateThreshold: {runtime_min_hits}/{runtime_state_threshold}",
        ]
        if last_crossing_text:
            overlay_lines.append(last_crossing_text)
        if last_inference_error:
            overlay_lines.append(last_inference_error)
        if gate_flash_frames_remaining > 0:
            gate_flash_frames_remaining -= 1

        if output_writer is not None and (running or OUTPUT_VIDEO_WRITE_WHEN_PAUSED):
            output_writer.write(frame)

        # ------------------------------------------------------------------ #
        # 7. Encode to JPEG and store                                          #
        # ------------------------------------------------------------------ #
        ok, jpeg_buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, runtime_jpeg_quality]
        )
        if ok:
            with state.lock:
                state.latest_jpeg = jpeg_buf.tobytes()
                state.latest_frame_ts = time.time()
                state.frame_overlay_lines = overlay_lines

        state.update_loop_metrics(
            frame_processed=True,
            inference_latency_ms=inference_latency_ms,
        )

        # ------------------------------------------------------------------ #
        # 8. Frame-rate throttle                                               #
        # ------------------------------------------------------------------ #
        elapsed = time.perf_counter() - t_iter_start
        sleep_time = target_frame_time - elapsed
        if sleep_time > 0.0:
            time.sleep(sleep_time)


def _debug_main() -> None:
    """Run the engine module without FastAPI for a short standalone smoke test."""
    parser = argparse.ArgumentParser(description="Standalone engine debugger")
    parser.add_argument("--seconds", type=float, default=5.0, help="How long to observe the engine")
    args = parser.parse_args()

    cap = cv2.VideoCapture(VIDEO_PATH)
    print(
        {
            "video_path": VIDEO_PATH,
            "video_opened": cap.isOpened(),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": cap.get(cv2.CAP_PROP_FRAME_COUNT),
        }
    )
    ok, frame = cap.read()
    print({"first_read_ok": ok, "frame_shape": None if frame is None else frame.shape})
    cap.release()

    state = SharedState(running=True, status_text="Engine standalone debug")
    worker = threading.Thread(target=run_engine, args=(state,), daemon=True, name="engine-debug")
    worker.start()

    start_time = time.time()
    last_ts = 0.0
    frames_seen = 0

    while time.time() - start_time < args.seconds:
        time.sleep(0.25)
        snapshot = state.snapshot()
        if snapshot["latest_frame_ts"] != last_ts:
            last_ts = snapshot["latest_frame_ts"]
            frames_seen += 1
        print(f"frames_seen={frames_seen} snapshot={snapshot}")


if __name__ == "__main__":
    _debug_main()
