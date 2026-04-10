from __future__ import annotations

import time
import threading
from pathlib import Path

from app.config import load_config
from app.engine import run_engine
from app.state import SharedState


def main() -> None:
    config = load_config()
    output_path = Path(config.output_video.path)

    state = SharedState(running=True, status_text="Render run")
    worker = threading.Thread(target=run_engine, args=(state,), daemon=True)
    worker.start()

    deadline = time.time() + 7200
    last_status = None
    last_heartbeat = 0.0
    final_status = "Timed out"
    final_count = 0

    while time.time() < deadline:
        time.sleep(0.5)
        with state.lock:
            status = state.status_text
            count = state.count
        if status != last_status:
            print({"status": status, "count": count}, flush=True)
            last_status = status

        now = time.time()
        if now - last_heartbeat >= 10.0:
            print({"event": "render_heartbeat", "status": status, "count": count}, flush=True)
            last_heartbeat = now

        final_status = status
        final_count = count
        if status.startswith("Video ended") or status.startswith("Error:"):
            break

    print(
        {
            "final_status": final_status,
            "final_count": final_count,
            "output_exists": output_path.exists(),
            "output_path": str(output_path),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()