"""
app/main.py
-----------
FastAPI application entry point.

Endpoints
~~~~~~~~~
POST /start   – set running = True
POST /stop    – set running = False
POST /reset   – request a momentary counter/tracker reset
GET  /state   – return JSON snapshot of current state
GET  /stream  – MJPEG live video stream (multipart/x-mixed-replace)
GET  /        – serve the HTML control page

Design principles
~~~~~~~~~~~~~~~~~
* API handlers only touch SharedState flags.  They contain no CV or
  counting logic.
* /stream is a true async generator; it yields the latest frame and
  sleeps via asyncio.sleep() so it never blocks the event loop.
* The engine worker is launched once as a daemon thread during startup.
  It outlives all requests and is terminated automatically when the
  process exits.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, Request, UploadFile
from fastapi import Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient
import uvicorn

from app.config import load_config
from app.engine import run_engine
from app.runtime_logging import log_event
from app.state import SharedState, shared_state

CONFIG = load_config()
# Force Reverse Decrease to start as true by default
REVERSE_DECREASE_DEFAULT = True
GATE_THICKNESS_DEFAULT = 5

_ALLOWED_DIRECTIONS = {
    "left-to-right",
    "right-to-left",
    "top-to-bottom",
    "bottom-to-top",
    "any",
}
_ALLOWED_ANCHOR_POINTS = {"topcenter", "bottomcenter", "bottomright"}

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v"}


def _get_videos_root() -> Path:
    configured = Path(CONFIG.video.path)
    root = configured.parent if str(configured.parent) not in ("", ".") else Path("videos")
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root


def _list_video_files() -> list[str]:
    root = _get_videos_root()
    if not root.exists() or not root.is_dir():
        return []

    videos: list[str] = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS:
            videos.append(f"videos/{path.name}")
    return videos

# ---------------------------------------------------------------------------
# Template renderer
# ---------------------------------------------------------------------------

templates = Jinja2Templates(directory="app/templates")


def _validate_startup() -> list[str]:
    """Return actionable startup validation errors for headless deployment."""
    errors: list[str] = []

    video_path = Path(os.environ.get("VIDEO_PATH", CONFIG.video.path))
    model_path = Path(os.environ.get("MODEL_PATH", CONFIG.model.path))

    if not video_path.exists():
        errors.append(f"VIDEO_PATH does not exist: {video_path}")

    if not model_path.exists():
        errors.append(f"MODEL_PATH does not exist: {model_path}")

    try:
        import openvino  # noqa: F401
        import scipy  # noqa: F401
    except Exception as exc:
        errors.append(f"openvino and/or scipy are required but unavailable: {exc}")

    if CONFIG.counter.state_threshold < 2:
        errors.append("counter.state_threshold should be >= 2 for stable crossing decisions")

    if len(CONFIG.counter.crossing_order) == 0:
        errors.append("counter.CrossingOrder must not be empty")

    return errors


# ---------------------------------------------------------------------------
# Application lifespan – start the engine thread before accepting requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the CV engine worker in a daemon thread on application startup."""
    startup_errors = _validate_startup()
    with shared_state.lock:
        shared_state.running = True
        shared_state.startup_validated = True
        shared_state.startup_errors = startup_errors
        shared_state.runtime_conf_threshold = float(CONFIG.model.conf_threshold)
        shared_state.runtime_iou_threshold = float(CONFIG.model.iou_threshold)
        shared_state.runtime_skip_frame = max(1, int(CONFIG.model.skip_frame))
        shared_state.runtime_polylines = [list(polyline) for polyline in CONFIG.counter.polylines]
        shared_state.runtime_crossing_directions = list(CONFIG.counter.crossing_directions)
        shared_state.runtime_crossing_order = list(CONFIG.counter.crossing_order)
        shared_state.runtime_anchor_point = str(CONFIG.counter.anchor_point).lower()
        shared_state.runtime_min_hits = int(CONFIG.counter.min_hits)
        shared_state.runtime_state_threshold = int(CONFIG.counter.state_threshold)
        shared_state.runtime_reverse_decrease_counting = REVERSE_DECREASE_DEFAULT
        shared_state.runtime_gate_thickness = GATE_THICKNESS_DEFAULT
        shared_state.runtime_jpeg_quality = int(CONFIG.stream.jpeg_quality)
        shared_state.status_text = "Running"

    if startup_errors:
        with shared_state.lock:
            shared_state.status_text = "Startup validation failed"
        shared_state.add_event("startup_validation_failed", {"errors": startup_errors})
        log_event("startup_validation_failed", errors=startup_errors)
        app.state.engine_thread = None
        yield
        return

    engine_thread = threading.Thread(
        target=run_engine,
        args=(shared_state,),
        daemon=True,         # thread exits automatically when the process does
        name="cv-engine",
    )
    app.state.engine_thread = engine_thread
    engine_thread.start()
    log_event("engine_thread_started", thread_name=engine_thread.name)
    yield

    with shared_state.lock:
        shared_state.shutdown_requested = True
        shared_state.status_text = "Shutting down"
    shared_state.add_event("shutdown_requested", {})

    engine_thread.join(timeout=5.0)
    if engine_thread.is_alive():
        shared_state.add_event("shutdown_timeout", {"timeout_seconds": 5.0})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sheep Carcass Counter",
    description="Headless backend for counting sheep carcasses on a conveyor.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Control endpoints
# ---------------------------------------------------------------------------

@app.post("/start", summary="Start counting")
def start() -> JSONResponse:
    """Enable inference and counting in the engine thread."""
    with shared_state.lock:
        shared_state.running = True
        shared_state.status_text = "Running"
    shared_state.add_event("start", {})
    return JSONResponse({"ok": True, "running": True})


@app.post("/stop", summary="Stop counting")
def stop() -> JSONResponse:
    """Pause inference and counting; the video stream continues."""
    with shared_state.lock:
        shared_state.running = False
        shared_state.status_text = "Stopped"
    shared_state.add_event("stop", {})
    return JSONResponse({"ok": True, "running": False})


@app.post("/reset", summary="Reset session count")
def reset() -> JSONResponse:
    """
    Request a counter/tracker reset.

    The flag is consumed by the engine on the next loop iteration and
    immediately cleared (momentary signal).  The count returns to zero
    and all per-track memory is erased.
    """
    with shared_state.lock:
        shared_state.reset_requested = True
        shared_state.status_text = "Resetting..."
    shared_state.add_event("reset", {})
    return JSONResponse({"ok": True, "reset_requested": True})


# ---------------------------------------------------------------------------
# State endpoint
# ---------------------------------------------------------------------------

@app.get("/state", summary="Get current state snapshot")
def get_state() -> JSONResponse:
    """Return the current running flag, status text, and session count."""
    with shared_state.lock:
        return JSONResponse({
            "running": shared_state.running,
            "status":  shared_state.status_text,
            "count":   shared_state.count,
            "current_video_path": shared_state.current_video_path,
            "video_paused": shared_state.video_paused,
            "requested_video_path": shared_state.requested_video_path,
            "frame_overlay_lines": list(shared_state.frame_overlay_lines),
            "runtime": {
                "conf_threshold": shared_state.runtime_conf_threshold,
                "iou_threshold": shared_state.runtime_iou_threshold,
                "skip_frame": shared_state.runtime_skip_frame,
                "tripwire_polylines": [list(polyline) for polyline in shared_state.runtime_polylines],
                "crossing_directions": list(shared_state.runtime_crossing_directions),
                "crossing_order": list(shared_state.runtime_crossing_order),
                "anchor_point": shared_state.runtime_anchor_point,
                "min_hits": shared_state.runtime_min_hits,
                "state_threshold": shared_state.runtime_state_threshold,
                "reverse_decrease_counting": shared_state.runtime_reverse_decrease_counting,
                "gate_thickness": shared_state.runtime_gate_thickness,
                "jpeg_quality": shared_state.runtime_jpeg_quality,
            },
        })


@app.get("/health", summary="Liveness endpoint for headless ops")
def health() -> JSONResponse:
    """Return process and engine liveness plus current metrics."""
    snap = shared_state.snapshot()
    return JSONResponse(
        {
            "ok": True,
            "status": "alive",
            "uptime_seconds": snap["uptime_seconds"],
            "engine_thread_alive": snap["engine_thread_alive"],
            "model_ready": snap["model_ready"],
            "frames_processed": snap["frames_processed"],
            "inference_runs": snap["inference_runs"],
            "loop_fps": snap["loop_fps"],
            "last_inference_latency_ms": snap["last_inference_latency_ms"],
            "avg_inference_latency_ms": snap["avg_inference_latency_ms"],
        }
    )


@app.get("/ready", summary="Readiness endpoint for orchestration")
def ready() -> JSONResponse:
    """Return readiness with actionable startup validation errors."""
    snap = shared_state.snapshot()
    is_ready = (
        snap["startup_validated"]
        and len(snap["startup_errors"]) == 0
        and snap["engine_thread_alive"]
        and snap["model_ready"]
    )
    payload = {
        "ready": is_ready,
        "startup_errors": snap["startup_errors"],
        "engine_thread_alive": snap["engine_thread_alive"],
        "model_ready": snap["model_ready"],
    }
    return JSONResponse(payload, status_code=200 if is_ready else 503)


@app.get("/events/recent", summary="Recent structured audit/count events")
def events_recent(limit: int = Query(50, ge=1, le=500)) -> JSONResponse:
    """Return the most recent structured backend events for diagnostics/audit."""
    with shared_state.lock:
        events = list(shared_state.event_history)
        total = len(events)
        sliced = events[-limit:]
    return JSONResponse({"events": sliced, "count": len(sliced), "total_buffered": total})


@app.get("/config", summary="Active runtime configuration")
def get_config() -> JSONResponse:
    """Expose selected active config values used by the counting engine."""
    return JSONResponse(
        {
            "video": {
                "path": CONFIG.video.path,
                "loop": CONFIG.video.loop,
                "available_files": _list_video_files(),
            },
            "model": {
                "path": CONFIG.model.path,
                "task": CONFIG.model.task,
                "classes": CONFIG.model.classes,
                "conf_threshold": CONFIG.model.conf_threshold,
                "iou_threshold": CONFIG.model.iou_threshold,
                "skip_frame": CONFIG.model.skip_frame,
            },
            "counter": {
                "name": CONFIG.counter.name,
                "event_type": CONFIG.counter.event_type,
                "input_name": CONFIG.counter.input_name,
                "polylines": CONFIG.counter.polylines,
                "crossing_directions": CONFIG.counter.crossing_directions,
                "crossing_order": CONFIG.counter.crossing_order,
                "anchor_point": CONFIG.counter.anchor_point,
                "state_threshold": CONFIG.counter.state_threshold,
                "reverse_decrease_counting": REVERSE_DECREASE_DEFAULT,
                "min_hits": CONFIG.counter.min_hits,
            },
        }
    )


@app.get("/runtime-settings", summary="Live runtime detection settings")
def get_runtime_settings() -> JSONResponse:
    """Return runtime-adjustable parameters currently used by the engine loop."""
    with shared_state.lock:
        return JSONResponse(
            {
                "conf_threshold": shared_state.runtime_conf_threshold,
                "iou_threshold": shared_state.runtime_iou_threshold,
                "skip_frame": shared_state.runtime_skip_frame,
                "tripwire_polylines": [list(polyline) for polyline in shared_state.runtime_polylines],
                "crossing_directions": list(shared_state.runtime_crossing_directions),
                "crossing_order": list(shared_state.runtime_crossing_order),
                "anchor_point": shared_state.runtime_anchor_point,
                "min_hits": shared_state.runtime_min_hits,
                "state_threshold": shared_state.runtime_state_threshold,
                "reverse_decrease_counting": shared_state.runtime_reverse_decrease_counting,
                "gate_thickness": shared_state.runtime_gate_thickness,
                "jpeg_quality": shared_state.runtime_jpeg_quality,
            }
        )


@app.post("/runtime-settings", summary="Update live runtime detection settings")
async def update_runtime_settings(request: Request) -> JSONResponse:
    """Apply runtime settings without restarting the backend process."""
    payload = await request.json()

    with shared_state.lock:
        current_conf = float(shared_state.runtime_conf_threshold)
        current_iou = float(shared_state.runtime_iou_threshold)
        current_skip = int(shared_state.runtime_skip_frame)
        current_polylines = [list(polyline) for polyline in shared_state.runtime_polylines]
        current_directions = list(shared_state.runtime_crossing_directions)
        current_order = list(shared_state.runtime_crossing_order)
        current_anchor_point = str(shared_state.runtime_anchor_point)
        current_min_hits = int(shared_state.runtime_min_hits)
        current_state_threshold = int(shared_state.runtime_state_threshold)
        current_reverse_decrease = bool(shared_state.runtime_reverse_decrease_counting)
        current_gate_thickness = int(shared_state.runtime_gate_thickness)
        current_jpeg_quality = int(shared_state.runtime_jpeg_quality)

    try:
        conf = float(payload.get("conf_threshold", current_conf))
        iou = float(payload.get("iou_threshold", current_iou))
        skip = int(payload.get("skip_frame", current_skip))
        min_hits = int(payload.get("min_hits", current_min_hits))
        state_threshold = int(payload.get("state_threshold", current_state_threshold))
        reverse_decrease_counting = bool(payload.get("reverse_decrease_counting", current_reverse_decrease))
        gate_thickness = int(payload.get("gate_thickness", current_gate_thickness))
        jpeg_quality = int(payload.get("jpeg_quality", current_jpeg_quality))
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid runtime settings values"}, status_code=400)

    directions_payload = payload.get("crossing_directions")
    crossing_order_payload = payload.get("crossing_order")
    anchor_point_payload = payload.get("anchor_point")

    crossing_directions = current_directions
    if directions_payload is not None:
        if not isinstance(directions_payload, list) or not directions_payload:
            return JSONResponse({"ok": False, "error": "crossing_directions must be a non-empty list"}, status_code=400)
        parsed_directions: list[str] = []
        for item in directions_payload:
            direction = str(item).strip().lower()
            if direction not in _ALLOWED_DIRECTIONS:
                return JSONResponse({"ok": False, "error": f"unsupported crossing direction: {direction}"}, status_code=400)
            parsed_directions.append(direction)
        crossing_directions = parsed_directions

    crossing_order = current_order
    if crossing_order_payload is not None:
        if not isinstance(crossing_order_payload, list) or not crossing_order_payload:
            return JSONResponse({"ok": False, "error": "crossing_order must be a non-empty list"}, status_code=400)
        try:
            crossing_order = [int(item) for item in crossing_order_payload]
        except Exception:
            return JSONResponse({"ok": False, "error": "crossing_order must be integers"}, status_code=400)

    anchor_point = current_anchor_point
    if anchor_point_payload is not None:
        anchor_point = str(anchor_point_payload).strip().lower()
        if anchor_point not in _ALLOWED_ANCHOR_POINTS:
            return JSONResponse({"ok": False, "error": "anchor_point must be topcenter, bottomcenter, or bottomright"}, status_code=400)

    polylines_payload = payload.get("tripwire_polylines")
    line1_y_payload = payload.get("line1_y")
    line2_y_payload = payload.get("line2_y")

    polylines = current_polylines
    if line1_y_payload is not None or line2_y_payload is not None:
        if len(current_polylines) < 2:
            return JSONResponse({"ok": False, "error": "current tripwire setup requires at least two lines"}, status_code=400)
        try:
            y1 = float(line1_y_payload if line1_y_payload is not None else current_polylines[0][0][1])
            y2 = float(line2_y_payload if line2_y_payload is not None else current_polylines[1][0][1])
        except Exception:
            return JSONResponse({"ok": False, "error": "line1_y/line2_y must be numeric"}, status_code=400)
        if not 0.0 <= y1 <= 1.0 or not 0.0 <= y2 <= 1.0:
            return JSONResponse({"ok": False, "error": "line1_y/line2_y must be between 0.0 and 1.0"}, status_code=400)
        polylines = [list(polyline) for polyline in current_polylines]
        polylines[0] = [(0.0, y1), (1.0, y1)]
        polylines[1] = [(0.0, y2), (1.0, y2)]
    elif polylines_payload is not None:
        try:
            parsed: list[list[tuple[float, float]]] = []
            for polyline in polylines_payload:
                if not isinstance(polyline, list) or len(polyline) != 2:
                    raise ValueError()
                p1 = (float(polyline[0][0]), float(polyline[0][1]))
                p2 = (float(polyline[1][0]), float(polyline[1][1]))
                parsed.append([p1, p2])
            polylines = parsed
        except Exception:
            return JSONResponse({"ok": False, "error": "tripwire_polylines must be [[[x,y],[x,y]], ...]"}, status_code=400)

    if not 0.0 <= conf <= 1.0:
        return JSONResponse({"ok": False, "error": "conf_threshold must be between 0.0 and 1.0"}, status_code=400)
    if not 0.0 <= iou <= 1.0:
        return JSONResponse({"ok": False, "error": "iou_threshold must be between 0.0 and 1.0"}, status_code=400)
    if skip < 1:
        return JSONResponse({"ok": False, "error": "skip_frame must be >= 1"}, status_code=400)
    if min_hits < 1:
        return JSONResponse({"ok": False, "error": "min_hits must be >= 1"}, status_code=400)
    if state_threshold < 2:
        return JSONResponse({"ok": False, "error": "state_threshold must be >= 2"}, status_code=400)
    if gate_thickness < 1 or gate_thickness > 20:
        return JSONResponse({"ok": False, "error": "gate_thickness must be between 1 and 20"}, status_code=400)
    if jpeg_quality < 20 or jpeg_quality > 100:
        return JSONResponse({"ok": False, "error": "jpeg_quality must be between 20 and 100"}, status_code=400)
    if len(polylines) != len(CONFIG.counter.crossing_directions):
        return JSONResponse(
            {"ok": False, "error": "tripwire_polylines count must match configured crossing directions"},
            status_code=400,
        )
    if len(crossing_directions) != len(polylines):
        return JSONResponse({"ok": False, "error": "crossing_directions count must match tripwire_polylines"}, status_code=400)
    if len(set(crossing_order)) != len(crossing_order):
        return JSONResponse({"ok": False, "error": "crossing_order must contain unique line indices"}, status_code=400)
    if any(line_index < 1 or line_index > len(polylines) for line_index in crossing_order):
        return JSONResponse({"ok": False, "error": "crossing_order contains an out-of-range line index"}, status_code=400)

    with shared_state.lock:
        shared_state.runtime_conf_threshold = conf
        shared_state.runtime_iou_threshold = iou
        shared_state.runtime_skip_frame = skip
        shared_state.runtime_polylines = [list(polyline) for polyline in polylines]
        shared_state.runtime_crossing_directions = list(crossing_directions)
        shared_state.runtime_crossing_order = list(crossing_order)
        shared_state.runtime_anchor_point = anchor_point
        shared_state.runtime_min_hits = min_hits
        shared_state.runtime_state_threshold = state_threshold
        shared_state.runtime_reverse_decrease_counting = reverse_decrease_counting
        shared_state.runtime_gate_thickness = gate_thickness
        shared_state.runtime_jpeg_quality = jpeg_quality
        shared_state.status_text = (
            f"Runtime settings updated: conf={conf:.2f} iou={iou:.2f} skip={skip} "
            f"min_hits={min_hits} state_threshold={state_threshold} "
            f"thickness={gate_thickness} jpg={jpeg_quality}"
        )

    shared_state.add_event(
        "runtime_settings_updated",
        {
            "conf_threshold": conf,
            "iou_threshold": iou,
            "skip_frame": skip,
            "tripwire_polylines": [list(polyline) for polyline in polylines],
            "crossing_directions": list(crossing_directions),
            "crossing_order": list(crossing_order),
            "anchor_point": anchor_point,
            "min_hits": min_hits,
            "state_threshold": state_threshold,
            "reverse_decrease_counting": reverse_decrease_counting,
            "gate_thickness": gate_thickness,
            "jpeg_quality": jpeg_quality,
        },
    )

    return JSONResponse(
        {
            "ok": True,
            "conf_threshold": conf,
            "iou_threshold": iou,
            "skip_frame": skip,
            "tripwire_polylines": [list(polyline) for polyline in polylines],
            "crossing_directions": list(crossing_directions),
            "crossing_order": list(crossing_order),
            "anchor_point": anchor_point,
            "min_hits": min_hits,
            "state_threshold": state_threshold,
            "reverse_decrease_counting": reverse_decrease_counting,
            "gate_thickness": gate_thickness,
            "jpeg_quality": jpeg_quality,
        }
    )


@app.get("/videos", summary="List selectable video files")
def list_videos() -> JSONResponse:
    """Return available input video files under the configured videos directory."""
    with shared_state.lock:
        current = shared_state.current_video_path or CONFIG.video.path
    return JSONResponse({"videos": _list_video_files(), "current_video_path": current})


@app.post("/videos/select", summary="Select active input video")
async def select_video(request: Request) -> JSONResponse:
    """Request the engine to switch to a different video file."""
    payload = await request.json()
    selected_path = str(payload.get("path", "")).strip()
    if not selected_path:
        return JSONResponse({"ok": False, "error": "path is required"}, status_code=400)

    root = _get_videos_root()
    candidate = (root / Path(selected_path).name).resolve()
    if candidate.parent != root.resolve() or not candidate.exists() or not candidate.is_file():
        return JSONResponse({"ok": False, "error": "video file not found"}, status_code=404)
    if candidate.suffix.lower() not in _VIDEO_EXTENSIONS:
        return JSONResponse({"ok": False, "error": "unsupported video file extension"}, status_code=400)

    requested = f"videos/{candidate.name}"
    with shared_state.lock:
        shared_state.requested_video_path = requested
        shared_state.status_text = f"Switching video to {requested}"
    shared_state.add_event("video_switch_requested", {"video_path": requested})
    return JSONResponse({"ok": True, "requested_video_path": requested})


@app.post("/videos/restart", summary="Restart current video from beginning")
def restart_video() -> JSONResponse:
    """Request engine to seek current video back to frame 0 and reset counting state."""
    with shared_state.lock:
        current = shared_state.current_video_path or CONFIG.video.path
        shared_state.restart_video_requested = True
        shared_state.status_text = f"Restarting video: {current}"
    shared_state.add_event("video_restart_requested", {"video_path": current})
    return JSONResponse({"ok": True, "video_path": current})


@app.post("/videos/pause", summary="Pause current video playback")
def pause_video() -> JSONResponse:
    """Pause frame progression while keeping the stream endpoint active."""
    with shared_state.lock:
        shared_state.video_paused = True
        current = shared_state.current_video_path or CONFIG.video.path
        shared_state.status_text = f"Video paused: {current}"
    shared_state.add_event("video_paused", {"video_path": current})
    log_event("video_pause_requested", video_path=current)
    return JSONResponse({"ok": True, "video_paused": True, "video_path": current})


@app.post("/videos/resume", summary="Resume current video playback")
def resume_video() -> JSONResponse:
    """Resume frame progression after a pause."""
    with shared_state.lock:
        shared_state.video_paused = False
        current = shared_state.current_video_path or CONFIG.video.path
        shared_state.status_text = f"Video playing: {current}"
    shared_state.add_event("video_resumed", {"video_path": current})
    log_event("video_resume_requested", video_path=current)
    return JSONResponse({"ok": True, "video_paused": False, "video_path": current})


@app.post("/videos/upload", summary="Upload and select input video")
async def upload_video(file: UploadFile = File(...)) -> JSONResponse:
    """Upload a local video file from the browser and switch engine source to it."""
    filename = (file.filename or "").strip()
    suffix = Path(filename).suffix.lower()
    if suffix not in _VIDEO_EXTENSIONS:
        return JSONResponse(
            {"ok": False, "error": "unsupported video file extension"},
            status_code=400,
        )

    root = _get_videos_root()
    root.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem or "uploaded_video"
    safe_stem = "".join(ch for ch in stem if ch.isalnum() or ch in ("-", "_")) or "uploaded_video"
    target_name = f"{safe_stem}_{int(time.time())}{suffix}"
    target_path = (root / target_name).resolve()

    if target_path.parent != root.resolve():
        return JSONResponse({"ok": False, "error": "invalid target path"}, status_code=400)

    try:
        with target_path.open("wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    finally:
        await file.close()

    requested = f"videos/{target_name}"
    with shared_state.lock:
        shared_state.requested_video_path = requested
        shared_state.status_text = f"Switching video to {requested}"
    shared_state.add_event("video_uploaded", {"video_path": requested, "original_name": filename})
    shared_state.add_event("video_switch_requested", {"video_path": requested})

    return JSONResponse({"ok": True, "requested_video_path": requested})


# ---------------------------------------------------------------------------
# MJPEG stream endpoint
# ---------------------------------------------------------------------------

async def _mjpeg_generator(state: SharedState):
    """
    Async generator that yields MJPEG frames for StreamingResponse.

    * Checks for a new frame by comparing latest_frame_ts with the last
      seen timestamp; only yields when a genuinely new frame is available.
    * Uses asyncio.sleep() between polls so the event loop stays responsive
      and other requests are not starved.
    """
    last_ts: float = 0.0

    while True:
        # Brief lock to read the latest frame reference and its timestamp.
        with state.lock:
            jpeg = state.latest_jpeg
            ts = state.latest_frame_ts

        if jpeg is not None and ts != last_ts:
            last_ts = ts
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
        else:
            # No new frame yet; yield control to the event loop briefly.
            await asyncio.sleep(0.020)   # poll at ~50 Hz maximum


@app.get("/stream", summary="MJPEG video stream")
async def stream():
    """
    Live annotated video stream using multipart/x-mixed-replace (MJPEG).
    Compatible with a plain HTML <img> element – no JavaScript needed for
    the video display itself.
    """
    return StreamingResponse(
        _mjpeg_generator(shared_state),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# UI endpoint
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, summary="Web control UI")
async def index(request: Request) -> HTMLResponse:
    """Serve the single-page HTML control interface."""
    return templates.TemplateResponse(request, "index.html", {"request": request})


def _debug_main() -> None:
    """Run a standalone self-test or start a real server when requested."""
    parser = argparse.ArgumentParser(description="Standalone FastAPI debugger")
    parser.add_argument("--host", default=CONFIG.server.host, help="Bind host")
    parser.add_argument("--port", type=int, default=CONFIG.server.port, help="Bind port")
    parser.add_argument("--reload", action="store_true", default=CONFIG.server.reload, help="Enable auto-reload")
    parser.add_argument("--serve", action="store_true", help="Start a real Uvicorn server instead of running a self-test")
    args = parser.parse_args()

    if not args.serve:
        with TestClient(app) as client:
            state_response = client.get("/state")
            index_response = client.get("/")
            print(
                {
                    "state_status": state_response.status_code,
                    "state_json": state_response.json(),
                    "index_status": index_response.status_code,
                    "index_has_title": "Sheep Carcass Counter" in index_response.text,
                }
            )
        return

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    _debug_main()
