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
import threading
from contextlib import asynccontextmanager
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient
import uvicorn

from app.config import load_config
from app.engine import run_engine
from app.state import SharedState, shared_state

CONFIG = load_config()

# ---------------------------------------------------------------------------
# Template renderer
# ---------------------------------------------------------------------------

templates = Jinja2Templates(directory="app/templates")


# ---------------------------------------------------------------------------
# Application lifespan – start the engine thread before accepting requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the CV engine worker in a daemon thread on application startup."""
    engine_thread = threading.Thread(
        target=run_engine,
        args=(shared_state,),
        daemon=True,         # thread exits automatically when the process does
        name="cv-engine",
    )
    engine_thread.start()
    yield
    # No explicit cleanup needed; daemon thread terminates with the process.


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
    return JSONResponse({"ok": True, "running": True})


@app.post("/stop", summary="Stop counting")
def stop() -> JSONResponse:
    """Pause inference and counting; the video stream continues."""
    with shared_state.lock:
        shared_state.running = False
        shared_state.status_text = "Stopped"
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
        })


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
