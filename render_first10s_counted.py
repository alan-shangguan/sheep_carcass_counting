import os
import time
import threading
from pathlib import Path

os.environ["VIDEO_PATH"] = "videos/Event20260204013619006_first10s.mp4"
os.environ["OUTPUT_VIDEO_PATH"] = "outputs/Event20260204013619006_first10s_counted_top_h05_h06_b2t_v2.mkv"

from app.state import SharedState
from app.engine import run_engine


def main() -> None:
    state = SharedState(running=True, status_text="Render run")
    worker = threading.Thread(target=run_engine, args=(state,), daemon=True)
    worker.start()

    output_path = Path(os.environ["OUTPUT_VIDEO_PATH"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + 900
    last_status = None
    while time.time() < deadline:
        time.sleep(0.5)
        with state.lock:
            status = state.status_text
            count = state.count
        if status != last_status:
            print({"status": status, "count": count})
            last_status = status
        if status.startswith("Video ended"):
            break

    print(
        {
            "final_status": status,
            "final_count": count,
            "output_exists": output_path.exists(),
            "output_path": str(output_path),
        }
    )


if __name__ == "__main__":
    main()
