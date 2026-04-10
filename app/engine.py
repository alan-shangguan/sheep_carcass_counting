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
TRIPWIRE_REVERSE_DECREASE_COUNTING: bool = CONFIG.counter.reverse_decrease_counting
MIN_HITS: int = int(os.environ.get("MIN_HITS", str(CONFIG.counter.min_hits)))
MODEL_CLASSES: list[int] | None = CONFIG.model.classes
SKIP_FRAME: int = max(1, int(CONFIG.model.skip_frame))
MODEL_CONF_THRESHOLD: float = float(CONFIG.model.conf_threshold)
MODEL_IOU_THRESHOLD: float = float(CONFIG.model.iou_threshold)
JPEG_QUALITY: int = int(os.environ.get("JPEG_QUALITY", str(CONFIG.stream.jpeg_quality)))
OUTPUT_VIDEO_ENABLED: bool = CONFIG.output_video.enabled
OUTPUT_VIDEO_PATH: str = os.environ.get("OUTPUT_VIDEO_PATH", CONFIG.output_video.path)
OUTPUT_VIDEO_CODEC: str = CONFIG.output_video.codec
OUTPUT_VIDEO_FPS: float | None = CONFIG.output_video.fps
OUTPUT_VIDEO_WRITE_WHEN_PAUSED: bool = CONFIG.output_video.write_when_paused

# ---------------------------------------------------------------------------
# YOLO availability check
# ---------------------------------------------------------------------------

try:
    from ultralytics import YOLO as _YOLO  # noqa: N812
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(state: SharedState):
    """
    Attempt to load a YOLO model.  Returns the model object on success, or
    None when ultralytics is not installed or the weights file is missing.
    Updates state.status_text with the outcome.
    """
    if not _YOLO_AVAILABLE:
        with state.lock:
            state.status_text = "YOLO not installed – running in mock/preview mode"
        return None

    try:
        if CONFIG.model.task:
            model = _YOLO(MODEL_PATH, task=CONFIG.model.task)
        else:
            model = _YOLO(MODEL_PATH)
        with state.lock:
            state.status_text = f"Model loaded: {MODEL_PATH}"
        return model
    except Exception as exc:
        with state.lock:
            state.status_text = f"Model load failed ({exc}) – mock/preview mode"
        return None


def _open_video(state: SharedState) -> cv2.VideoCapture | None:
    """
    Try to open the configured video file.  Returns a VideoCapture object
    that isOpened(), or None on failure.
    """
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        with state.lock:
            state.status_text = f"Cannot open video: {VIDEO_PATH}"
        return None
    return cap


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


def _draw_gate(frame: np.ndarray, active: bool, flash: bool = False) -> None:
    """Overlay configured ordered tripwire polylines on the frame in-place."""
    h, w = frame.shape[:2]
    colour = (0, 0, 255) if flash else ((0, 255, 0) if active else (0, 200, 200))

    for index, polyline in enumerate(TRIPWIRE_POLYLINES, start=1):
        p1 = (int(polyline[0][0] * w), int(polyline[0][1] * h))
        p2 = (int(polyline[1][0] * w), int(polyline[1][1] * h))
        line_colour = colour if index in TRIPWIRE_CROSSING_ORDER else (160, 160, 160)
        cv2.line(frame, p1, p2, line_colour, 2)
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
    frame_w: int,
    frame_h: int,
    fps: float,
) -> cv2.VideoWriter | None:
    """Create the annotated output video writer, if enabled and possible."""
    if not OUTPUT_VIDEO_ENABLED:
        return None

    output_path = Path(OUTPUT_VIDEO_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        return None
    return writer


def _extract_tracks(results) -> list[dict]:
    """
    Parse ultralytics Results into a list of track dicts compatible with
    counter.process_frame().

    Returns an empty list when no boxes or no tracking IDs are present.
    """
    tracks: list[dict] = []
    if not results or results[0].boxes is None:
        return tracks

    boxes = results[0].boxes
    if boxes.id is None:
        return tracks

    ids = boxes.id.int().tolist()
    xyxy = boxes.xyxy.tolist()

    for tid, bbox in zip(ids, xyxy):
        tracks.append({"id": tid, "bbox": tuple(bbox)})

    return tracks


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
    model = _load_model(state)
    cap: cv2.VideoCapture | None = None
    output_writer: cv2.VideoWriter | None = None
    output_writer_failed = False
    output_writer_released = False
    frame_index = 0
    gate_flash_frames_remaining = 0
    last_crossing_text = ""
    inference_runs = 0

    # Target frame interval; updated once the video is opened.
    target_frame_time: float = 1.0 / 25.0

    while True:
        t_iter_start = time.perf_counter()

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

        # ------------------------------------------------------------------ #
        # 2. Snapshot control flags                                            #
        # ------------------------------------------------------------------ #
        with state.lock:
            running = state.running

        # ------------------------------------------------------------------ #
        # 3. Ensure video capture is open                                     #
        # ------------------------------------------------------------------ #
        if cap is None or not cap.isOpened():
            cap = _open_video(state)
            if cap is None:
                # Emit a placeholder frame so /stream stays alive.
                ph = _placeholder_frame(f"Waiting for video: {VIDEO_PATH}")
                _, jpeg_buf = cv2.imencode(
                    ".jpg", ph, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                with state.lock:
                    state.latest_jpeg = jpeg_buf.tobytes()
                    state.latest_frame_ts = time.time()
                time.sleep(1.0)
                continue

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
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print({"event": "video_loop", "video_path": VIDEO_PATH}, flush=True)
            else:
                if output_writer is not None and not output_writer_released:
                    output_writer.release()
                    output_writer_released = True
                with state.lock:
                    if OUTPUT_VIDEO_ENABLED:
                        state.status_text = f"Video ended. Output saved to {OUTPUT_VIDEO_PATH}"
                    else:
                        state.status_text = f"Video ended: {VIDEO_PATH}"
                print({"event": "video_end", "video_path": VIDEO_PATH, "output_path": OUTPUT_VIDEO_PATH}, flush=True)
                time.sleep(0.25)
            continue

        frame_index += 1

        frame_h, frame_w = frame.shape[:2]
        if output_writer is None and not output_writer_failed and not output_writer_released:
            writer_fps = OUTPUT_VIDEO_FPS if OUTPUT_VIDEO_FPS and OUTPUT_VIDEO_FPS > 0 else (1.0 / target_frame_time)
            output_writer = _open_output_writer(frame_w, frame_h, writer_fps)
            if OUTPUT_VIDEO_ENABLED and output_writer is None:
                output_writer_failed = True
                with state.lock:
                    state.status_text = f"Output video open failed: {OUTPUT_VIDEO_PATH}"
                print({"event": "output_writer_failed", "output_path": OUTPUT_VIDEO_PATH}, flush=True)
            elif output_writer is not None:
                print(
                    {
                        "event": "output_writer_opened",
                        "output_path": OUTPUT_VIDEO_PATH,
                        "fps": writer_fps,
                        "frame_size": [frame_w, frame_h],
                    },
                    flush=True,
                )

        # ------------------------------------------------------------------ #
        # 5. Inference + counting (no lock held across inference)              #
        # ------------------------------------------------------------------ #
        count_increment = 0

        if running:
            if model is not None:
                try:
                    # Run detection/tracking on one frame out of every SKIP_FRAME.
                    if (frame_index - 1) % SKIP_FRAME == 0:
                        inference_runs += 1
                        results = model.track(
                            frame,
                            persist=CONFIG.model.persist_tracking,
                            verbose=CONFIG.model.verbose,
                            classes=MODEL_CLASSES,
                            conf=MODEL_CONF_THRESHOLD,
                            iou=MODEL_IOU_THRESHOLD,
                        )
                        # Use the annotated frame produced by ultralytics.
                        frame = results[0].plot()

                        # Extract track list and run gate counter.
                        raw_tracks = _extract_tracks(results)
                        tracks = _filter_tracks_for_counting(raw_tracks, frame_w, frame_h)
                        count_increment = counter_module.process_frame(
                            tracks,
                            state.track_memory,   # mutated in-place; engine owns it
                            frame_w,
                            frame_h,
                            polylines=TRIPWIRE_POLYLINES,
                            crossing_directions=TRIPWIRE_CROSSING_DIRECTIONS,
                            crossing_order=TRIPWIRE_CROSSING_ORDER,
                            unit=TRIPWIRE_UNIT,
                            anchor_point=TRIPWIRE_ANCHOR_POINT,
                            min_hits=MIN_HITS,
                            state_threshold=TRIPWIRE_STATE_THRESHOLD,
                            reverse_decrease_counting=TRIPWIRE_REVERSE_DECREASE_COUNTING,
                        )
                        print(
                            {
                                "event": "inference",
                                "frame": frame_index,
                                "inference_run": inference_runs,
                                "raw_tracks": len(raw_tracks),
                                "filtered_tracks": len(tracks),
                                "count_increment": count_increment,
                            },
                            flush=True,
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
                            print({"event": "crossing", **last_event}, flush=True)
                except Exception as exc:
                    # Inference errors must not crash the stream.
                    _put_text(frame, f"Inference error: {exc}", (10, 45), 1.0, (0, 0, 255), 2)
                    print({"event": "inference_error", "frame": frame_index, "error": str(exc)}, flush=True)
            else:
                # Mock mode: no inference, no counting.
                _put_text(frame, "MOCK MODE - no YOLO model", (10, 45), 1.1, (0, 120, 255), 2)

        # ------------------------------------------------------------------ #
        # 6. Annotate frame                                                    #
        # ------------------------------------------------------------------ #

        # Gate line (green when running, cyan when paused).
        gate_flash_active = gate_flash_frames_remaining > 0
        _draw_gate(frame, running, flash=gate_flash_active)

        # Count overlay (always visible).
        with state.lock:
            state.count += count_increment
            count_display = state.count

        if frame_index % max(30, SKIP_FRAME * 10) == 0:
            print(
                {
                    "event": "heartbeat",
                    "frame": frame_index,
                    "running": running,
                    "count": count_display,
                    "inference_runs": inference_runs,
                },
                flush=True,
            )

        # Consolidated top-left HUD block for all status text.
        _put_text(frame, f"Count: {count_display}", (16, 52), 1.4, (255, 255, 0), 3)
        mode_text = "Mode: RUNNING" if running else "Mode: PAUSED"
        mode_colour = (80, 255, 80) if running else (0, 220, 220)
        _put_text(frame, mode_text, (16, 86), 0.95, mode_colour, 2)

        info_y = 116
        if SKIP_FRAME > 1:
            _put_text(frame, f"SkipFrame: {SKIP_FRAME}", (16, info_y), 0.9, (255, 255, 255), 2)
            info_y += 28
        _put_text(
            frame,
            f"Conf: {MODEL_CONF_THRESHOLD:.2f} IoU: {MODEL_IOU_THRESHOLD:.2f}",
            (16, info_y),
            0.9,
            (255, 255, 255),
            2,
        )

        # Keep latest crossing event in a bottom banner to avoid overlap with boxes/tripwires.
        if last_crossing_text:
            event_colour = (0, 0, 255) if gate_flash_active else (255, 255, 255)
            _put_text(frame, last_crossing_text, (16, frame_h - 24), 0.9, event_colour, 2)
        _draw_config_overlay(frame)
        if gate_flash_frames_remaining > 0:
            gate_flash_frames_remaining -= 1

        if output_writer is not None and (running or OUTPUT_VIDEO_WRITE_WHEN_PAUSED):
            output_writer.write(frame)

        # ------------------------------------------------------------------ #
        # 7. Encode to JPEG and store                                          #
        # ------------------------------------------------------------------ #
        ok, jpeg_buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if ok:
            with state.lock:
                state.latest_jpeg = jpeg_buf.tobytes()
                state.latest_frame_ts = time.time()

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
