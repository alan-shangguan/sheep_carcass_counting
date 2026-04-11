"""
app/state.py
------------
Thread-safe shared state for the sheep carcass counting backend.

All inter-thread communication goes through this single object.
The engine thread writes frames, counts, and status.
The API layer reads/writes control flags (running, reset_requested).
The lock must be held for the minimum time necessary – never hold it
during long operations like model inference or JPEG encoding.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from app.runtime_logging import log_event


@dataclass
class SharedState:
    # ------------------------------------------------------------------ #
    # Control flags – written by API layer, read by engine thread          #
    # ------------------------------------------------------------------ #

    # True  → engine runs inference and increments count.
    # False → engine keeps streaming frames but does no counting.
    running: bool = False

    # Human-readable description of the current engine state.
    status_text: str = "Idle"

    # Optional runtime video source switch requested by the UI/API.
    requested_video_path: str | None = None

    # Request to seek current video back to frame 0.
    restart_video_requested: bool = False

    # Video file currently opened by the engine loop.
    current_video_path: str = ""

    # Pause frame advancement while keeping stream alive.
    video_paused: bool = False

    # Set to True by POST /reset.  The engine clears count and
    # track_memory on the next iteration, then immediately sets this back
    # to False (momentary signal – not a latched flag).
    reset_requested: bool = False

    # ------------------------------------------------------------------ #
    # Output – written by engine thread, read by API / stream endpoint    #
    # ------------------------------------------------------------------ #

    # Running total of carcasses counted in the current session.
    count: int = 0

    # Latest annotated video frame encoded as JPEG bytes.
    # None until the engine produces its first frame.
    latest_jpeg: bytes | None = None

    # Monotonic timestamp (time.time()) of the latest JPEG.
    # Used by the stream generator to detect a new frame.
    latest_frame_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle, readiness, and metrics                                   #
    # ------------------------------------------------------------------ #

    # Process start timestamp used for uptime calculations.
    started_at: float = field(default_factory=time.time)

    # Set during startup validation. When False, /ready should fail.
    startup_validated: bool = False
    startup_errors: list[str] = field(default_factory=list)

    # Engine thread lifecycle flags.
    engine_thread_alive: bool = False
    model_ready: bool = False
    shutdown_requested: bool = False

    # Runtime metrics updated by the engine.
    frames_processed: int = 0
    inference_runs: int = 0
    loop_fps: float = 0.0
    last_inference_latency_ms: float = 0.0
    avg_inference_latency_ms: float = 0.0
    _last_frame_tick: float = 0.0

    # Text lines currently rendered as frame overlays by the engine.
    frame_overlay_lines: list[str] = field(default_factory=list)

    # Runtime-adjustable model parameters controlled by Web UI.
    runtime_conf_threshold: float = 0.25
    runtime_iou_threshold: float = 0.45
    runtime_skip_frame: int = 1
    runtime_polylines: list[list[tuple[float, float]]] = field(default_factory=list)
    runtime_crossing_directions: list[str] = field(default_factory=list)
    runtime_crossing_order: list[int] = field(default_factory=list)
    runtime_anchor_point: str = "topcenter"
    runtime_min_hits: int = 3
    runtime_state_threshold: int = 3
    runtime_reverse_decrease_counting: bool = False
    runtime_gate_thickness: int = 5
    runtime_jpeg_quality: int = 80

    # ------------------------------------------------------------------ #
    # Per-track counting memory – owned by engine/counter, cleared on     #
    # reset.  Structure is defined and mutated by app/counter.py.         #
    # ------------------------------------------------------------------ #

    # Dict[track_id: int, track_entry: dict]
    # See counter.py for the schema of each entry.
    track_memory: dict = field(default_factory=dict)

    # Structured audit/history events for backend APIs.
    event_history: deque = field(default_factory=lambda: deque(maxlen=500))
    next_event_id: int = 1

    # ------------------------------------------------------------------ #
    # Synchronisation primitive                                            #
    # ------------------------------------------------------------------ #

    # Protects every field above from concurrent read/write races.
    # Use fine-grained locking: acquire, read/write the relevant fields,
    # release immediately – never hold across blocking I/O or inference.
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_event(self, event_type: str, payload: dict | None = None) -> None:
        """Append a structured event to the in-memory ring buffer."""
        with self.lock:
            event = {
                "id": self.next_event_id,
                "ts": time.time(),
                "type": event_type,
                "payload": payload or {},
            }
            self.next_event_id += 1
            self.event_history.append(event)
        log_event(
            "state_event",
            event_id=event["id"],
            state_event_type=event_type,
            payload=event["payload"],
            ts=event["ts"],
        )

    def update_loop_metrics(self, *, frame_processed: bool, inference_latency_ms: float | None = None) -> None:
        """Update FPS and inference latency metrics from the engine loop."""
        now = time.time()
        with self.lock:
            if frame_processed:
                self.frames_processed += 1
                if self._last_frame_tick > 0.0:
                    dt = now - self._last_frame_tick
                    if dt > 0:
                        instant_fps = 1.0 / dt
                        self.loop_fps = instant_fps if self.loop_fps <= 0.0 else (self.loop_fps * 0.9 + instant_fps * 0.1)
                self._last_frame_tick = now

            if inference_latency_ms is not None:
                self.last_inference_latency_ms = float(inference_latency_ms)
                self.avg_inference_latency_ms = (
                    self.last_inference_latency_ms
                    if self.avg_inference_latency_ms <= 0.0
                    else (self.avg_inference_latency_ms * 0.9 + self.last_inference_latency_ms * 0.1)
                )
                self.inference_runs += 1

    def snapshot(self) -> dict:
        """Return a small serialisable state snapshot for logging/debugging."""
        with self.lock:
            return {
                "running": self.running,
                "status_text": self.status_text,
                "requested_video_path": self.requested_video_path,
                "restart_video_requested": self.restart_video_requested,
                "current_video_path": self.current_video_path,
                "video_paused": self.video_paused,
                "count": self.count,
                "reset_requested": self.reset_requested,
                "latest_jpeg_ready": self.latest_jpeg is not None,
                "latest_frame_ts": self.latest_frame_ts,
                "track_memory_size": len(self.track_memory),
                "startup_validated": self.startup_validated,
                "startup_errors": list(self.startup_errors),
                "engine_thread_alive": self.engine_thread_alive,
                "model_ready": self.model_ready,
                "frames_processed": self.frames_processed,
                "inference_runs": self.inference_runs,
                "loop_fps": self.loop_fps,
                "last_inference_latency_ms": self.last_inference_latency_ms,
                "avg_inference_latency_ms": self.avg_inference_latency_ms,
                "frame_overlay_lines": list(self.frame_overlay_lines),
                "runtime_conf_threshold": self.runtime_conf_threshold,
                "runtime_iou_threshold": self.runtime_iou_threshold,
                "runtime_skip_frame": self.runtime_skip_frame,
                "runtime_polylines": [list(polyline) for polyline in self.runtime_polylines],
                "runtime_crossing_directions": list(self.runtime_crossing_directions),
                "runtime_crossing_order": list(self.runtime_crossing_order),
                "runtime_anchor_point": self.runtime_anchor_point,
                "runtime_min_hits": self.runtime_min_hits,
                "runtime_state_threshold": self.runtime_state_threshold,
                "runtime_reverse_decrease_counting": self.runtime_reverse_decrease_counting,
                "runtime_gate_thickness": self.runtime_gate_thickness,
                "runtime_jpeg_quality": self.runtime_jpeg_quality,
                "uptime_seconds": max(0.0, time.time() - self.started_at),
            }


# ---------------------------------------------------------------------------
# Module-level singleton.
# Import and use `shared_state` directly; do not create additional instances.
# ---------------------------------------------------------------------------
shared_state: SharedState = SharedState()


def _debug_main() -> None:
    """Simple standalone smoke test for this module."""
    state = SharedState()
    print("Initial:", state.snapshot())

    with state.lock:
        state.running = True
        state.status_text = "Standalone debug"
        state.count = 3
        state.track_memory[101] = {"counted": False}

    print("Updated:", state.snapshot())


if __name__ == "__main__":
    _debug_main()
