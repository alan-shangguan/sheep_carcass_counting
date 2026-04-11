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

## Run modes

### API + browser stream

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000

### Offline render using current config

```bash
python render_current_config.py
```

This runs the engine in render mode and writes the configured annotated output
file, for example outputs/counting_result.mp4.

## Recent Updates

- Runtime Settings panel now displays all live settings (except tripwire fields) in a read-only summary for transparency.
- Tripwire fields (L1 Y, L2 Y) are no longer shown in the Runtime Settings panel or summary.
- The Reverse Decrease setting now always starts as true by default, regardless of config.
- To rebuild and restart the Docker service after changes:
  ```bash
  docker compose build sheep-counter
  docker compose restart sheep-counter
  ```
- For troubleshooting, check logs with:
  ```bash
  docker compose logs --tail 60 sheep-counter
  ```
