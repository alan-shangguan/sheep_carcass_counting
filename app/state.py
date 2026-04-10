"""
app/state.py
------------
Thread-safe shared state for the sheep carcass counting backend.

All inter-thread communication goes through this single object.
The engine thread writes frames, counts, and status.
The API layer reads/writes control flags (running, reset_requested).
The lock must be held for the minimum time necessary – never hold it
during long operations like YOLO inference or JPEG encoding.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


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
    # Per-track counting memory – owned by engine/counter, cleared on     #
    # reset.  Structure is defined and mutated by app/counter.py.         #
    # ------------------------------------------------------------------ #

    # Dict[track_id: int, track_entry: dict]
    # See counter.py for the schema of each entry.
    track_memory: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Synchronisation primitive                                            #
    # ------------------------------------------------------------------ #

    # Protects every field above from concurrent read/write races.
    # Use fine-grained locking: acquire, read/write the relevant fields,
    # release immediately – never hold across blocking I/O or inference.
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        """Return a small serialisable state snapshot for logging/debugging."""
        with self.lock:
            return {
                "running": self.running,
                "status_text": self.status_text,
                "count": self.count,
                "reset_requested": self.reset_requested,
                "latest_jpeg_ready": self.latest_jpeg is not None,
                "latest_frame_ts": self.latest_frame_ts,
                "track_memory_size": len(self.track_memory),
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
