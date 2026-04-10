# Sheep Carcass Counter

A headless video analytics backend for sheep carcass counting using FastAPI,
OpenCV, and Ultralytics YOLO/OpenVINO models. The application supports live
browser monitoring, REST controls, and offline annotated render output.

## What it does

This project reads a conveyor video, tracks carcasses, and applies an
ordered multi-polyline counter. It is designed for production-style
review workflows where you need both a live stream and a saved rendered
output video with overlays and event evidence.

## Current feature set

- YAML-driven runtime configuration from config.yaml (video, model, counter,
  stream, output, server).
- OpenVINO-ready model loading via model path in config.
- Ordered polyline crossing logic with per-line motion direction settings.
- Configurable anchor point for crossing checks: TopCenter, BottomCenter,
  BottomRight.
- Per-object UUID assignment and rolling 5-state side memory per polyline.
- Count only when 5 states exist, using first-vs-last side comparison.
- Signed counting:
  - crossing all polylines in configured order adds +1
  - crossing all polylines in reverse order subtracts -1
- Skip-frame inference (for example, detect every 5th frame).
- Configurable detector/tracker thresholds:
  - model.conf_threshold
  - model.iou_threshold
- Optional size sanity filtering before counting using width/height/area/
  aspect-ratio envelopes.
- On-frame visual feedback:
  - gate lines and order labels
  - larger count, status, event, and config text overlays
  - crossing event text (UUID, delta, side transition)
  - gate flash (red) on crossing
  - full active-config overlay (anchor, direction, order, thresholds, skip,
    size-sanity summary)
- Render output support to an annotated video file.
- Verbose process logging (inference events, crossing events, periodic
  heartbeat, output-writer lifecycle).

## Project layout

```text
sheep_carcass_counting/
├── app/
│   ├── config.py                 # YAML schema, defaults, validation
│   ├── counter.py                # ordered directed-polyline counting logic
│   ├── engine.py                 # OpenCV + YOLO/OpenVINO worker loop
│   ├── main.py                   # FastAPI routes/UI/stream
│   ├── state.py                  # shared state + lock
│   └── templates/
│       └── index.html            # browser control UI
├── config.yaml                   # primary runtime configuration
├── render_current_config.py      # offline render runner using config.yaml
├── render_first10s_counted.py    # helper script for short sample render
├── requirements.txt
├── videos/
├── weights/
└── outputs/
```

## Counting model summary

1. Each tracked box is mapped to an anchor point.
2. Each polyline splits the frame into two sides relative to the line.
3. CrossingDirections describes object motion across those sides, not the point order of the polyline itself.
4. A rolling 5-state side history is maintained per object per polyline.
5. A polyline crossing is evaluated only when 5 side states are available.
6. A count is emitted only after the object crosses all polylines in CrossingOrder.
7. Last event metadata is emitted for overlay and logging.

## Configuration highlights

Most changes should be made in config.yaml.

Important keys:

- video.path, video.loop
- model.path, model.task, model.classes
- model.skip_frame
- model.conf_threshold
- model.iou_threshold
- counter.Polylines
- counter.CrossingDirections
- counter.CrossingOrder
- counter.AnchorPoint
- counter.size_sanity.*
- output_video.enabled, output_video.path, output_video.codec, output_video.fps

Notes:

- counter.Unit currently supports only Normalized.
- counter.CrossingDirections defines object motion across a polyline side boundary.
- The start/end point order of a polyline does not need to numerically match the text in CrossingDirections.
- model.classes can be null for all classes, or an integer/list to filter classes.
- Environment variables can still override selected values for quick debugging
  (for example VIDEO_PATH, MODEL_PATH, OUTPUT_VIDEO_PATH).

## Getting Started

### Prerequisites

**System Requirements:**
- Python 3.8+
- 4GB RAM minimum (8GB recommended for real-time inference)
- GPU optional but recommended (NVIDIA CUDA 11.8+ or Intel Arc)
- Linux, macOS, or Windows

**Dependencies:**
- OpenCV (cv2)
- PyTorch/Ultralytics YOLO
- OpenVINO Runtime (for edge inference)
- FastAPI + Uvicorn
- NumPy, Pillow

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/alan-shangguan/sheep_carcass_counting.git
cd sheep_carcass_counting
```

#### 2. Create Python virtual environment

**Using venv (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Or using conda:**
```bash
conda create -n sheep-counting python=3.10
conda activate sheep-counting
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Prepare files

- **Video files**: Place video files in `videos/` directory
- **Model weights**: Place OpenVINO model in `weights/best_openvino_model/` 
  - Should contain: `openvino.xml`, `openvino.bin`
- **Config**: Edit `config.yaml` with your video path and model path

### Configuration (config.yaml)

Key settings to adjust:

```yaml
video:
  path: videos/YOUR_VIDEO.mp4      # Path to input video
  loop: true                        # Loop video at end

model:
  path: weights/best_openvino_model # Path to model weights
  task: detect                     # yolo task (detect/segment)
  conf_threshold: 0.5              # Detection confidence threshold
  iou_threshold: 0.45              # NMS IoU threshold
  skip_frame: 1                    # Inference every Nth frame (1=every frame)

counter:
  AnchorPoint: TopCenter           # TopCenter/BottomCenter/BottomRight
  CrossingOrder: [1, 2]            # Polyline IDs in counting sequence
  Polylines:
    1: [[0.2, 0.5], [0.8, 0.5]]   # Line 1 (normalized coordinates)
    2: [[0.2, 0.6], [0.8, 0.6]]   # Line 2

output_video:
  enabled: true
  path: outputs/counting_result.mp4
  codec: mp4v                       # Video codec
  fps: 25.0

server:
  host: 0.0.0.0                    # 0.0.0.0 for network access
  port: 8000
```

## Run modes

### 1. Web Browser UI (Recommended)

Start the FastAPI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Or using environment variables:

```bash
export VIDEO_PATH=videos/Event20260204013619006_first10s.mp4
export MODEL_PATH=weights/best_openvino_model
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open your web browser:
```
http://localhost:8000
```

**Dashboard includes:**
- Live annotated video stream
- Real-time count display
- Start/Stop/Reset controls
- Performance metrics (FPS, latency, uptime)
- System health indicators
- Recent events log

### 2. Offline Render (Batch Processing)

Render an annotated output video without browser:

```bash
python render_current_config.py
```

Output will be saved to the path specified in `config.yaml` under `output_video.path`.

### 3. Docker

#### Build and run with Docker Compose

```bash
docker compose up --build
```

This will:
- Build the FastAPI app container
- Expose port 8000
- Mount local videos/ and weights/ directories
- Start the system in web browser mode

Access at: http://localhost:8000

#### Run standalone Docker container

```bash
docker build -t sheep-counting:latest .
docker run -p 8000:8000 \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/outputs:/app/outputs \
  -e VIDEO_PATH=videos/YOUR_VIDEO.mp4 \
  -e MODEL_PATH=weights/best_openvino_model \
  sheep-counting:latest
```

### 4. Linux Headless Deployment

For production Linux environments (e.g., industrial edge devices):

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip python3-venv

# 2. Clone and setup
git clone https://github.com/alan-shangguan/sheep_carcass_counting.git
cd sheep_carcass_counting
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure config.yaml for your video files

# 4. Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. Access UI from any machine on the network
# http://<EDGE_DEVICE_IP>:8000
```

**Systemd Service (optional):**

Create `/etc/systemd/system/sheep-counting.service`:

```ini
[Unit]
Description=Sheep Carcass Counting System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/sheep_carcass_counting
Environment="PATH=/home/ubuntu/sheep_carcass_counting/venv/bin"
ExecStart=/home/ubuntu/sheep_carcass_counting/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable sheep-counting
sudo systemctl start sheep-counting
sudo systemctl status sheep-counting
```



## REST API

| Method | Path | Description |
|---|---|---|
| POST | /start | Enable inference and counting |
| POST | /stop | Pause counting (stream continues) |
| POST | /reset | Reset count and per-object tracking memory |
| GET | /state | Returns {running, status, count} snapshot |
| GET | /health | Liveness check: {ok, status, engine_thread_alive, model_ready, frames_processed, loop_fps, uptime_seconds, ...} |
| GET | /ready | Readiness check: {ready, startup_errors, engine_thread_alive, model_ready} (503 if not ready) |
| GET | /events/recent | Recent event history: {events: [{timestamp, event_type, ...}]} |
| GET | /stream | MJPEG video stream (use in <img src="/stream">) |
| GET | / | Browser dashboard UI (HTML) |

### Example API calls

```bash
# Start counting
curl -X POST http://localhost:8000/start

# Check system health
curl http://localhost:8000/health
# Response: {
#   "ok": true,
#   "status": "alive",
#   "engine_thread_alive": true,
#   "model_ready": true,
#   "frames_processed": 1250,
#   "loop_fps": 25.0,
#   "uptime_seconds": 45.3
# }

# Check readiness
curl http://localhost:8000/ready
# Response: {
#   "ready": true,
#   "startup_errors": [],
#   "engine_thread_alive": true,
#   "model_ready": true
# }

# Get recent events
curl http://localhost:8000/events/recent
# Response: {
#   "events": [
#     {
#       "timestamp": "2026-04-11T12:34:56.789Z",
#       "event_type": "crossing",
#       "details": {...}
#     }
#   ]
# }

# Get current state
curl http://localhost:8000/state
# Response: {
#   "running": true,
#   "status": "Running",
#   "count": 42
# }
```

## Troubleshooting

### System won't start or shows errors

**Problem:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

**Problem:** `FileNotFoundError: Video file not found`

**Solution:** Ensure the video path in `config.yaml` is correct and files exist:
```bash
ls -la videos/
# Check that your video file is there
```

Or set via environment variable:
```bash
export VIDEO_PATH=/path/to/your/video.mp4
```

**Problem:** `Model path not found` or model won't load

**Solution:** Verify model files exist:
```bash
ls -la weights/best_openvino_model/
# Should contain: openvino.xml and openvino.bin
```

If using a Ultralytics YOLO model, convert to OpenVINO:
```bash
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='openvino')
```

### Web UI doesn't load

**Problem:** Browser shows connection refused on localhost:8000

**Solution:** 
- Verify server is running: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- Check firewall isn't blocking port 8000
- Try accessing from same machine first: `http://127.0.0.1:8000`

**Problem:** Video stream shows "Initializing stream..."

**Solution:**
- Check that the video file exists and is readable
- Try running `/health` endpoint to see engine status
- Check logs for OpenCV errors
- Ensure video codec is supported (H.264, MPEG-4 recommended)

### Low FPS or performance issues

**Problem:** Engine reports FPS < 5

**Solutions:**
1. Increase `model.skip_frame` in config.yaml to run inference less frequently
2. Reduce video resolution or quality
3. Enable GPU acceleration (set CUDA/OpenVINO device)
4. Reduce detection confidence threshold to speed up NMS
5. Check system resources: `top` (Linux) or Task Manager (Windows)

**Problem:** Memory usage keeps growing

**Solution:**
- Check that background frame cleanup is happening
- Verify output video writer isn't holding too many frames
- Monitor with `/health` endpoint for `frames_processed` growth

### Counting seems wrong

**Problem:** Count is too high or too low

**Solutions:**
1. Verify counting configuration in `config.yaml`:
   - `CrossingOrder` matches your polyline IDs
   - `Polylines` coordinates look correct on the video overlay
   - `AnchorPoint` is appropriate for your objects
2. Try adjusting `model.conf_threshold` (lower = more detections = higher count)
3. Check `counter.size_sanity` filters aren't filtering out real objects
4. Review overlay text to see which crossing events are being logged

**Problem:** Same object counted multiple times

**Solution:** Increase the size of the 5-state history or reduce skip_frame to get finer tracking

### Docker issues

**Problem:** `docker compose up` fails to build

**Solution:**
```bash
# Clean up previous builds
docker compose down -v
# Rebuild from scratch
docker compose up --build --no-cache
```

**Problem:** Container exits immediately

**Solution:** Check logs:
```bash
docker compose logs -f
# Look for startup errors or missing files
```

## Performance Tips

1. **For real-time on consumer hardware:**
   - Use OpenVINO with CPU backend
   - Set `skip_frame: 3` or higher (trade responsiveness for speed)
   - Reduce video resolution in source files
   - Run on Linux (often faster than Windows)

2. **For maximum accuracy:**
   - Set `skip_frame: 1` (every frame)
   - Use GPU acceleration if available
   - Lower `model.conf_threshold` to ~0.3 for more detections
   - Ensure good video quality from camera

3. **For industrial deployment:**
   - Use systemd service for auto-restart
   - Monitor `/health` endpoint for liveness
   - Set up log rotation for long-running sessions
   - Use Docker for reproducible environments
   - Consider multiple instances with load balancing for high throughput

## Advanced Configuration

### Custom polylines

Edit `config.yaml` to define crossing lines:

```yaml
counter:
  Polylines:
    1: [[0.1, 0.3], [0.9, 0.3]]   # Horizontal line 30% down
    2: [[0.5, 0.2], [0.5, 0.8]]   # Vertical line at 50% width
```

Coordinates are normalized 0-1 (0,0 = top-left, 1,1 = bottom-right).

### Custom size filters

Exclude objects that don't match expected dimensions:

```yaml
counter:
  size_sanity:
    enable: true
    width: [0.02, 0.3]     # Object width 2%-30% of frame width
    height: [0.02, 0.5]    # Object height 2%-50% of frame height
    aspect_ratio: [0.3, 3] # Width/height ratio between 0.3 and 3
```

## Support & Contributing

For bugs or feature requests, open an issue on GitHub.
Pull requests welcome!
