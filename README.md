# Sheep Carcass Counting System

Technical assessment submission for a Linux-based edge AI system that counts
sheep carcasses moving on a conveyor belt. The system is designed for a
headless unit: the computer vision engine runs in the backend, and operators
monitor and control it through a browser dashboard.

## Quick Start

On Linux, put input videos under `videos/`, then run:

```bash
chmod +x run_system.sh
./run_system.sh
```

Open the dashboard:

```text
http://127.0.0.1:8000/
```

## Data Handling And Training

I treat the supplied carcass videos as the source of both model data and
counting validation data. The important point is to split by video/time segment,
not by random adjacent frames, because neighboring video frames are nearly
duplicates and can make validation look better than it really is.

I received 8 videos and split them by video, not by random frames:

- 5 videos for training
- 1 video for validation
- 2 videos for test

Two of the videos are deliberate negative examples:

- 1 training video has litter/debris but no sheep carcass
- 1 test video has litter/debris but no sheep carcass

I included these negative videos on purpose so the detector and counting logic
see conveyor clutter that should not be counted as carcasses.

Final dataset size was approximately:

- 500+ training images
- 100+ validation images
- 100+ test images

Data handling workflow:

1. Extract representative frames from each video at a fixed interval and around
   difficult sections such as overlap, sway, partial carcasses, and lighting
   changes.
2. Use SAM3 to generate labels for visible carcasses.
3. Convert the SAM3 label results into YOLO-format bounding box labels.
4. Label each visible carcass as one class, `object`.
5. Keep litter/debris-only frames as hard negatives with no carcass boxes.
6. Train a YOLOv8 nano detector in Google Colab.
7. Export the selected model to OpenVINO so it can run on my CPU-only machine
   and on a Linux edge unit without requiring a GPU.

Example dataset layout:

```text
dataset/
|-- carcass.yaml
|-- images/
|   |-- train/
|   |-- val/
|   `-- test/
`-- labels/
    |-- train/
    |-- val/
    `-- test/
```

Example `dataset/carcass.yaml`:

```yaml
path: dataset
train: images/train
val: images/val
test: images/test
names:
  0: object
```

Example training/export commands in Google Colab:

```bash
python -m pip install ultralytics
yolo detect train data=dataset/carcass.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16 project=runs/carcass name=sheep_counter
yolo export model=runs/carcass/sheep_counter/weights/best.pt format=openvino imgsz=640
```

The edge runtime does not require the Ultralytics training package. It uses the
exported OpenVINO model under `weights/best_openvino_model`.

## Requirement Coverage

| Requirement | Implementation |
| --- | --- |
| Train/use a carcass detector | The repo includes trained/exported model artifacts in `weights/`, with the active OpenVINO model at `weights/best_openvino_model`. Metadata shows a one-class YOLO-style detector for `object`. |
| Robust counting despite imperfect detections | Detection is combined with IoU tracking, configurable confidence/IoU thresholds, skip-frame inference, anchor-point selection, ordered virtual tripwires, side-history smoothing, and optional size sanity filters. |
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
   | OpenCV video + OpenVINO detector + IoU tracker
   v
Tripwire counter (app/counter.py)
```

## Model Artifacts

The active inference path uses OpenVINO for deployment. The repo includes:

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

## Counting Logic

The counter is implemented in `app/counter.py`.

Each tracked detection becomes:

```text
track_id -> bbox -> configured anchor point -> side of each tripwire
```

The active config uses two normalized horizontal tripwires:

```yaml
counter:
  Polylines:
    - [[1.00, 0.40], [0.00, 0.40]]
    - [[1.00, 0.20], [0.00, 0.20]]
  CrossingDirections: [left-to-right, left-to-right]
  CrossingOrder: [1, 2]
  state_threshold: 3
  AnchorPoint: TopCenter
```

Important details:

- Coordinates are normalized to the frame.
- Only `counter.Unit: Normalized` is currently supported.
- `CrossingDirections` describes movement across a side boundary, not the
  literal point order of the polyline.
- Each object receives a UUID in per-track memory.
- Each track stores rolling side history per line.
- A crossing is considered only after `counter.state_threshold` side
  observations.
- A positive count is emitted when a track crosses all lines in
  `CrossingOrder`.
- A reverse sequence can subtract `1` when reverse-decrease counting is enabled.
- Optional size sanity filtering can reject detections outside configured
  width, height, area, and aspect-ratio ranges.

This combination is meant to avoid counting every raw detection. The model can
flicker for a few frames, boxes can wobble as carcasses sway, and tracker IDs
can be imperfect; the ordered gate sequence and side-history check make the
count depend on motion through a corridor rather than a single frame.

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

The dashboard is served from `app/templates/index.html` at:

```text
http://127.0.0.1:8000/
```

It displays:

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
|   |-- openvino_inference.py     # OpenVINO detector + IoU tracker
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

- Input video: `videos/Event20260123020659006.mp4`
- Model path: `weights/best_openvino_model`
- Class filter: `0`
- Confidence threshold: `0.85`
- IoU threshold: `0.45`
- Skip frame: `5`
- Anchor point: `TopCenter`
- State threshold: `3`
- Size sanity filter: disabled
- Output video: enabled

Selected environment variables can override runtime paths/settings for quick
testing, including `VIDEO_PATH`, `MODEL_PATH`, `OUTPUT_VIDEO_PATH`, `MIN_HITS`,
and `JPEG_QUALITY`.

## Linux Setup

Put input videos under `videos/` and keep model artifacts under `weights/`.
Then build and run the Linux/headless service:

```bash
chmod +x run_system.sh
./run_system.sh
```

The script creates `videos/` and `outputs/` if needed, checks for Docker
Compose, and runs:

```bash
docker compose up --build sheep-counter
```

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

Open the dashboard from another machine or browser:

```text
http://<edge-unit-ip>:8000/
```

For local testing on the same machine:

```text
http://127.0.0.1:8000/
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
