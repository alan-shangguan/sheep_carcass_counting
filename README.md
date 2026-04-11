# Sheep Carcass Counting System

Technical assessment submission for a Linux-based edge AI system that counts
sheep carcasses moving on a conveyor belt. The system is designed for a
headless unit: the computer vision engine runs in the backend, and operators
monitor and control it through a browser dashboard.

## Quick Start

Prerequisites:

- Linux shell or WSL on Windows.
- Docker Engine with the Docker Compose plugin (`docker compose version` should
  work).
- The supplied videos copied into this repo's `videos/` directory.
- The exported model files under `weights/best_openvino_model/` (included in
  this submission).

Then run:

```bash
chmod +x run_system.sh
./run_system.sh
```

Open the dashboard:

```text
http://127.0.0.1:8000/
```

and click Start

## Data Handling And Training

I split the 8 videos by video, not by adjacent random frames: 5 train videos
for 500+ images, 1 validation video for 100+ images, and 2 test videos for
100+ images. I deliberately included one litter/debris-only video in train and
one in test so the detector sees clutter that should not be counted.

I used SAM3 to label carcasses, converted the labels to YOLO bounding boxes,
trained a one-class YOLOv8 nano `object` detector in Google Colab, then exported
the model to OpenVINO for CPU-only Linux edge inference. Test performance was
good overall; the main remaining false positives were on white litter, handled
after the confidence threshold with size sanity checks and tripwire logic.

I chose OpenVINO because my local PC does not have a GPU. Exporting the YOLOv8
nano model to OpenVINO lets the same detector run efficiently on CPU for local
testing and on a Linux edge unit without depending on CUDA.

Training/export commands in Google Colab:

```bash
python -m pip install ultralytics
yolo detect train data=dataset/carcass.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16 project=runs/carcass name=sheep_counter
yolo export model=runs/carcass/sheep_counter/weights/best.pt format=openvino imgsz=640
```

The edge runtime does not require the Ultralytics training package. It uses the
exported OpenVINO model under `weights/best_openvino_model`.

## Post-Model Counting Logic

After training, I do not rely on the detector alone for the final count. A raw
detection can flicker, a bounding box can shake as the carcass sways, and
white-colored litter can still appear as a false positive. I handle these
issues in software by putting a tracking and tripwire layer after the model.

The active setup uses two normalized horizontal tripwire lines to create a
small counting corridor:

```yaml
counter:
  Polylines:
    - [[1.00, 0.40], [0.00, 0.40]]
    - [[1.00, 0.20], [0.00, 0.20]]
  CrossingDirections: [left-to-right, left-to-right]
  CrossingOrder: [1, 2]
  AnchorPoint: TopCenter
  state_threshold: 3
```

The runtime logic is:

1. Filter weak detections first using the configured confidence threshold.
2. Apply size sanity checks after the confidence threshold. This
   rejects boxes whose width, height, area, or aspect ratio does not look like a
   real carcass. This is useful for the remaining white-litter false positives.
3. Pass the remaining detections into ByteTrack so one carcass can keep an
   identity across multiple frames, even if the detector confidence fluctuates.
4. Convert each tracked bounding box into an anchor point. The current config
   uses `TopCenter`, which was selected for this camera/conveyor view.
5. For each surviving track, record which side of each tripwire the anchor is
   on.
6. Require several side observations using `state_threshold` before accepting a
   crossing. This reduces false counts from bbox shaking, carcass sway, and
   one-frame detector flicker.
7. Count only when the same track crosses line `1` and then line `2` in
   `CrossingOrder`. A single detection near the line is not enough.
8. Treat reverse-order movement separately so reverse movement can subtract
   from the count when reverse-decrease counting is enabled.

An ROI filter could also be applied before counting to ignore detections
outside the conveyor/counting area. I chose not to enable it for this test
because the scene is simple and the combination of confidence threshold, size
sanity check, ByteTrack, and ordered tripwires was enough. The design still
leaves room to add ROI filtering if the camera view becomes more cluttered.

This handles the main failure modes:

- False positives: litter must pass confidence, size sanity, tracking, and the
  ordered tripwire sequence before it can be counted.
- False negatives: a missed frame does not immediately break the count because
  the tracker and side-history logic look across multiple frames.
- Bbox shaking: the anchor must show a side transition over several
  observations, not just jump across a line once.
- Carcass swaying: the two-line corridor requires ordered movement through the
  gate, which is more reliable than a single tripwire hit.

> Note: the core assessment answer is above this point. The remaining sections
> are supporting reference material for reviewers who want implementation
> details, API routes, configuration keys, and repository layout.

## Requirement Coverage

| Requirement | Implementation |
| --- | --- |
| Train/use a carcass detector | The repo includes trained/exported model artifacts in `weights/`, with the active OpenVINO model at `weights/best_openvino_model`. Metadata shows a one-class YOLO-style detector for `object`. |
| Robust counting despite imperfect detections | Detection is combined with IoU tracking, configurable confidence/IoU thresholds, skip-frame inference, anchor-point selection, ordered virtual tripwires, side-history smoothing, and size sanity filters. |
| Start/Stop latching | `POST /start` and `POST /stop` latch `SharedState.running`. Frames continue streaming, but inference/counting only happens in the running state. |
| Reset momentary | `POST /reset` sets `SharedState.reset_requested`. The engine consumes it on the next loop, zeroes the count, clears tracking memory, resets tracker state, and clears the flag. |
| Real-time backend responsiveness | FastAPI handlers only update shared state under a short lock. The OpenCV/OpenVINO worker thread owns video capture and inference, and it does not hold the API lock during inference or JPEG encoding. |
| Web dashboard | `GET /` serves a browser UI with live annotated MJPEG stream, current count, running/stopped state, video controls, runtime settings, health/readiness metrics, and recent events. |
| Linux/headless deployment | Dockerfile and Docker Compose files are provided. The runtime uses CPU-friendly dependencies: FastAPI, OpenCV headless, OpenVINO, SciPy, NumPy, and PyYAML. |

## Architecture

The application is intentionally split into three layers:

1. `app/main.py`: FastAPI API, browser UI routes, startup validation, and control endpoints.
2. `app/engine.py`: long-running OpenCV/OpenVINO worker loop that reads video, runs inference/tracking, performs counting, draws overlays, writes optional render output, and updates shared state.
3. `app/state.py`: thread-safe state object used as the boundary between the web layer and CV engine.

This avoids coupling request handling to expensive CV work. The UI can remain
responsive while the engine is reading frames, running inference, or encoding
MJPEG output.

```text
Browser UI
   |
   | HTTP controls + MJPEG stream
   v
FastAPI app (app/main.py)
   |
   | short locked reads/writes
   v
SharedState (app/state.py)
   |
   | owned and updated by worker thread
   v
CV engine (app/engine.py)
   |
   | OpenCV video + OpenVINO detector + ByteTrack
   v
Tripwire counter (app/counter.py)
```

## Model Artifacts

The active inference path uses OpenVINO for CPU deployment. The repo includes:

- `weights/best.pt`: Ultralytics checkpoint.
- `weights/last.pt`: Ultralytics checkpoint.
- `weights/best.onnx`: exported ONNX model.
- `weights/best_openvino_model/best.xml`: OpenVINO IR graph.
- `weights/best_openvino_model/best.bin`: OpenVINO weights.
- `weights/best_openvino_model/metadata.yaml`: export metadata.
- `weights/best_openvino_model/best.json`: preprocessing/postprocessing metadata.

The model metadata identifies a YOLO-style detection model trained with
Ultralytics, image size `640x640`, task `detect`, and one class:

```yaml
names:
  0: object
```

## State And Control Logic

The mocked industrial control signals are exposed as HTTP endpoints and handled
through `SharedState`.

### Start

```http
POST /start
```

Sets `shared_state.running = True`. This is latching. The engine processes
frames, runs inference, updates tracking memory, and increments/decrements the
count only while running.

### Stop

```http
POST /stop
```

Sets `shared_state.running = False`. This is latching. The video stream
continues, but inference/counting is skipped.

### Reset

```http
POST /reset
```

Sets `shared_state.reset_requested = True`. This is momentary. The engine
consumes the flag on its next loop iteration, then:

- sets `count = 0`
- clears `track_memory`
- clears `reset_requested`
- resets the OpenVINO tracker when available
- returns status to `Running` or `Idle` based on the current latch state

The reset signal therefore does not stick.

### Synchronization

The API layer uses the state lock only for brief reads/writes. The engine takes
a snapshot of runtime settings at the top of the loop, releases the lock, and
then performs video I/O, inference, counting, drawing, and JPEG encoding. This
keeps control requests responsive while the CV loop is busy.

## Web Interface

The dashboard served from `app/templates/index.html` displays:

- Live annotated video stream from `GET /stream`.
- Current session count from `GET /state`.
- Active state, running/stopped, video paused/resumed, and backend status.
- Health metrics from `GET /health`.
- Readiness from `GET /ready`.
- Recent structured events from `GET /events/recent`.
- Video selection/upload/restart/pause/resume controls.
- Runtime settings controls backed by `GET/POST /runtime-settings`.

## API Summary

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Browser dashboard |
| `GET /stream` | MJPEG annotated stream |
| `POST /start` | Latch counting on |
| `POST /stop` | Latch counting off |
| `POST /reset` | Momentary count/tracker reset |
| `GET /state` | Count, state, video state, runtime settings |
| `GET /health` | Liveness and performance metrics |
| `GET /ready` | Readiness and startup validation |
| `GET /events/recent` | Recent in-memory audit events |
| `GET /config` | Selected config values and available videos |
| `GET /runtime-settings` | Current live tuning parameters |
| `POST /runtime-settings` | Apply live tuning parameters |
| `GET /videos` | List available local videos |
| `POST /videos/select` | Switch active video |
| `POST /videos/restart` | Restart current video from frame 0 |
| `POST /videos/pause` | Pause video frame advancement |
| `POST /videos/resume` | Resume video frame advancement |
| `POST /videos/upload` | Upload and queue a video from the browser |

## Repository Layout

```text
sheep_carcass_counting/
|-- app/
|   |-- __init__.py
|   |-- config.py                 # config schema, defaults, validation
|   |-- counter.py                # ordered directed-polyline counting
|   |-- engine.py                 # OpenCV/OpenVINO worker loop
|   |-- engine_helpers.py         # older/shared helper functions
|   |-- main.py                   # FastAPI routes, UI, stream, lifecycle
|   |-- openvino_inference.py     # OpenVINO detector + tracker
|   |-- runtime_logging.py        # JSON-lines runtime logger
|   |-- state.py                  # shared state, metrics, event buffer
|   `-- templates/
|       `-- index.html            # browser dashboard
|-- weights/
|   |-- best.onnx
|   |-- best.pt
|   |-- last.pt
|   `-- best_openvino_model/
|       |-- best.bin
|       |-- best.json
|       |-- best.xml
|       `-- metadata.yaml
|-- config.yml                   # primary runtime configuration
|-- docker-compose.yml           # Linux/headless service definition
|-- Dockerfile                   # API/browser runtime image
|-- Dockerfile.render            # render base image
|-- run_system.sh                # Linux Docker Compose launcher
`-- requirements.txt             # CPU-only Python dependencies
```

The runtime expects local `videos/` and `outputs/` directories. They are ignored
by Git because they contain input media and generated artifacts.

## Configuration

The main configuration file is `config.yml`. The loader prefers `config.yml`
and falls back to `config.yaml`; `APP_CONFIG_PATH` can point to another file.

Important keys:

- `video.path`, `video.loop`
- `model.path`, `model.task`, `model.classes`
- `model.skip_frame`
- `model.conf_threshold`
- `model.iou_threshold`
- `counter.Polylines`
- `counter.CrossingDirections`
- `counter.CrossingOrder`
- `counter.AnchorPoint`
- `counter.state_threshold`
- `counter.min_hits`
- `counter.size_sanity.*`
- `stream.jpeg_quality`
- `output_video.enabled`
- `output_video.path`
- `output_video.codec`
- `output_video.fps`
- `output_video.write_when_paused`
- `server.host`, `server.port`, `server.reload`

Current active defaults:

- Input video: `videos/Event20260123020157006.mp4`
- Model path: `weights/best_openvino_model`
- Class filter: `0`
- Confidence threshold: `0.70`
- IoU threshold: `0.45`
- Skip frame: `5`
- Anchor point: `TopCenter`
- State threshold: `3`
- Size sanity filter: enabled
- Output video: enabled

Selected environment variables can override runtime paths/settings for quick
testing, including `VIDEO_PATH`, `MODEL_PATH`, `OUTPUT_VIDEO_PATH`, `MIN_HITS`,
and `JPEG_QUALITY`.

## Docker Details

The run script creates `videos/` and `outputs/` if needed, checks for Docker
Compose, and starts the service with `docker compose up --build sheep-counter`.

The Docker Compose service mounts:

- `./videos` to `/app/videos:ro`
- `./weights` to `/app/weights:ro`
- `./config.yml` to `/app/config.yml:ro`
- `./outputs` to `/app/outputs`

Rebuild after code changes:

```bash
docker compose build sheep-counter
docker compose restart sheep-counter
```

Check logs:

```bash
docker compose logs --tail 60 sheep-counter
```

Open the dashboard from another machine:

```text
http://<edge-unit-ip>:8000/
```

## Logging And Diagnostics

- Runtime JSON-lines log: `outputs/sheep_counter.log` by default.
- In-memory recent events: `GET /events/recent`.
- Health metrics: `GET /health`.
- Readiness checks: `GET /ready`.
- Rendered output videos: derived by `app/engine.py` from `output_video.path`
  and the active input video name.

## Known Notes

- The project currently contains deployment/inference artifacts, but not the
  full labeling/training script used to produce the included weights.
- `Dockerfile.render` remains as a render-oriented base image, but this
  workspace no longer contains a standalone render runner script.
- `app/engine_helpers.py` contains helper functions, while the current
  `app/engine.py` loop keeps its own local helper implementations.
- The dashboard tripwire Y controls were removed from the UI summary; runtime
  tripwire polylines can still be controlled through the backend payload shape
  supported by `POST /runtime-settings`.
