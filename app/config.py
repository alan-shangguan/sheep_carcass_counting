"""
app/config.py
-------------
YAML-backed application configuration.

The project keeps runtime settings in a single root-level `config.yaml` file.
This module loads that file, applies defaults, normalises types, and exposes
an immutable dataclass structure to the rest of the app.

Only the config file path itself is environment-configurable, via:

    APP_CONFIG_PATH=another-config.yaml
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VideoConfig:
    path: str
    loop: bool


@dataclass(frozen=True)
class ModelConfig:
    path: str
    classes: list[int] | None
    task: str | None
    persist_tracking: bool
    verbose: bool
    skip_frame: int
    conf_threshold: float
    iou_threshold: float


@dataclass(frozen=True)
class CounterConfig:
    polylines: list[list[tuple[float, float]]]
    crossing_directions: list[str]
    crossing_order: list[int]
    unit: str
    anchor_point: str
    min_hits: int
    state_threshold: int
    size_sanity_enabled: bool
    min_width_ratio: float
    max_width_ratio: float
    min_height_ratio: float
    max_height_ratio: float
    min_area_ratio: float
    max_area_ratio: float
    min_aspect_ratio: float
    max_aspect_ratio: float


@dataclass(frozen=True)
class StreamConfig:
    jpeg_quality: int
    placeholder_width: int
    placeholder_height: int


@dataclass(frozen=True)
class OutputVideoConfig:
    enabled: bool
    path: str
    codec: str
    fps: float | None
    write_when_paused: bool


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    reload: bool


@dataclass(frozen=True)
class AppConfig:
    video: VideoConfig
    model: ModelConfig
    counter: CounterConfig
    stream: StreamConfig
    output_video: OutputVideoConfig
    server: ServerConfig


_DEFAULT_CONFIG: dict[str, Any] = {
    "video": {
        "path": "videos/Event20260123020157006.mp4",
        "loop": True,
    },
    "model": {
        "path": "weights/best_openvino_model",
        "classes": None,
        "task": "detect",
        "persist_tracking": True,
        "verbose": False,
        "skip_frame": 1,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
    },
    "counter": {
        "Polylines": [
            [[0.40, 0.10], [0.40, 0.90]],
            [[0.60, 0.10], [0.60, 0.90]],
        ],
        "CrossingDirections": ["left-to-right", "left-to-right"],
        "CrossingOrder": [1, 2],
        "Unit": "Normalized",
        "AnchorPoint": "TopCenter",
        "min_hits": 3,
        "state_threshold": 3,
        "size_sanity": {
            "enabled": True,
            "min_width_ratio": 0.10,
            "max_width_ratio": 0.409,
            "min_height_ratio": 0.057,
            "max_height_ratio": 0.422,
            "min_area_ratio": 0.006,
            "max_area_ratio": 0.12,
            "min_aspect_ratio": 0.56,
            "max_aspect_ratio": 2.23,
        },
    },
    "stream": {
        "jpeg_quality": 80,
        "placeholder_width": 640,
        "placeholder_height": 480,
    },
    "output_video": {
        "enabled": True,
        "path": "outputs/counting_result.mkv",
        "codec": "XVID",
        "fps": None,
        "write_when_paused": False,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": False,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two nested dicts, with override values taking precedence."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalise_classes(value: Any) -> list[int] | None:
    """Accept null, a list of ints, or a comma-separated string of class IDs."""
    if value in (None, "", []):
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        classes = [part.strip() for part in value.split(",") if part.strip()]
        return [int(part) for part in classes] or None
    if isinstance(value, list):
        return [int(item) for item in value] or None
    raise ValueError("model.classes must be null, an int, a list, or a comma-separated string")


def _normalise_gate_line_ratios(value: Any) -> list[float]:
    """Require exactly two ordered gate-line ratios in the open interval [0, 1]."""
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("counter.gate_line_ratios must be a list with exactly two entries")

    ratios = [float(item) for item in value]
    if not 0.0 <= ratios[0] < ratios[1] <= 1.0:
        raise ValueError("counter.gate_line_ratios must be ordered and within [0.0, 1.0]")
    return ratios


def _normalise_polylines(value: Any) -> list[list[tuple[float, float]]]:
    """Require a list of polylines, each with exactly two [x, y] points."""
    if not isinstance(value, list) or not value:
        raise ValueError("counter.Polylines must be a non-empty list")

    polylines: list[list[tuple[float, float]]] = []
    for polyline in value:
        if not isinstance(polyline, list) or len(polyline) != 2:
            raise ValueError("each counter polyline must contain exactly two points")

        points: list[tuple[float, float]] = []
        for point in polyline:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("each counter polyline point must contain two coordinates")
            x, y = float(point[0]), float(point[1])
            points.append((x, y))
        polylines.append(points)
    return polylines


def _normalise_crossing_order(value: Any, polyline_count: int) -> list[int]:
    """Require a non-empty ordered list of distinct 1-based polyline indices."""
    if not isinstance(value, list) or not value:
        raise ValueError("counter.CrossingOrder must contain at least one entry")

    order = [int(item) for item in value]
    if len(set(order)) != len(order):
        raise ValueError("counter.CrossingOrder must reference distinct polylines")
    if any(item < 1 or item > polyline_count for item in order):
        raise ValueError("counter.CrossingOrder references a polyline index outside the Polylines list")
    return order


def _normalise_crossing_directions(value: Any, polyline_count: int) -> list[str]:
    """Normalise per-polyline crossing motion directions."""
    if not isinstance(value, list) or len(value) != polyline_count:
        raise ValueError("counter.CrossingDirections must contain one direction per polyline")

    directions = [str(item).strip().lower() for item in value]
    allowed = {
        "left-to-right",
        "right-to-left",
        "top-to-bottom",
        "bottom-to-top",
        "any",
    }
    if any(item not in allowed for item in directions):
        raise ValueError("counter.CrossingDirections contains an unsupported direction")
    return directions


def _normalise_anchor_point(value: Any) -> str:
    """Normalise supported anchor point names."""
    anchor = str(value).strip().lower()
    allowed = {"bottomcenter", "bottomright", "topcenter"}
    if anchor not in allowed:
        raise ValueError("counter.AnchorPoint must be TopCenter, BottomCenter, or BottomRight")
    return anchor


def _normalise_unit(value: Any) -> str:
    """Only Normalized coordinates are supported for tripwire config."""
    unit = str(value).strip().lower()
    if unit != "normalized":
        raise ValueError("counter.Unit must be Normalized")
    return unit


def _normalise_skip_frame(value: Any) -> int:
    """Normalise model.skip_frame to an integer >= 1."""
    skip_frame = int(value)
    if skip_frame < 1:
        raise ValueError("model.skip_frame must be >= 1")
    return skip_frame


def _normalise_threshold(value: Any, field_name: str) -> float:
    """Normalise a confidence/IoU threshold into [0.0, 1.0]."""
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{field_name} must be within [0.0, 1.0]")
    return threshold


def _normalise_ratio(value: Any, field_name: str) -> float:
    """Normalise a ratio value into a finite float."""
    ratio = float(value)
    if ratio < 0.0:
        raise ValueError(f"{field_name} must be >= 0.0")
    return ratio


def _normalise_size_sanity(value: Any) -> dict[str, float | bool]:
    """Validate the size sanity thresholds used to filter counted objects."""
    if not isinstance(value, dict):
        raise ValueError("counter.size_sanity must be a mapping")

    result = {
        "enabled": bool(value.get("enabled", True)),
        "min_width_ratio": _normalise_ratio(value.get("min_width_ratio", 0.0), "counter.size_sanity.min_width_ratio"),
        "max_width_ratio": _normalise_ratio(value.get("max_width_ratio", 1.0), "counter.size_sanity.max_width_ratio"),
        "min_height_ratio": _normalise_ratio(value.get("min_height_ratio", 0.0), "counter.size_sanity.min_height_ratio"),
        "max_height_ratio": _normalise_ratio(value.get("max_height_ratio", 1.0), "counter.size_sanity.max_height_ratio"),
        "min_area_ratio": _normalise_ratio(value.get("min_area_ratio", 0.0), "counter.size_sanity.min_area_ratio"),
        "max_area_ratio": _normalise_ratio(value.get("max_area_ratio", 1.0), "counter.size_sanity.max_area_ratio"),
        "min_aspect_ratio": _normalise_ratio(value.get("min_aspect_ratio", 0.0), "counter.size_sanity.min_aspect_ratio"),
        "max_aspect_ratio": _normalise_ratio(value.get("max_aspect_ratio", 999.0), "counter.size_sanity.max_aspect_ratio"),
    }

    ordered_pairs = [
        ("width", result["min_width_ratio"], result["max_width_ratio"]),
        ("height", result["min_height_ratio"], result["max_height_ratio"]),
        ("area", result["min_area_ratio"], result["max_area_ratio"]),
        ("aspect", result["min_aspect_ratio"], result["max_aspect_ratio"]),
    ]
    for label, minimum, maximum in ordered_pairs:
        if minimum > maximum:
            raise ValueError(f"counter.size_sanity {label} min must be <= max")

    return result


def _config_path() -> Path:
    """Return the resolved YAML config path."""
    env_path = os.environ.get("APP_CONFIG_PATH")
    if env_path:
        return Path(env_path).resolve()

    cwd = Path.cwd()
    preferred = cwd / "config.yml"
    fallback = cwd / "config.yaml"
    return preferred.resolve() if preferred.exists() else fallback.resolve()


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """Load, validate, and cache the application configuration from YAML."""
    path = _config_path()
    raw_config: dict[str, Any] = {}

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
            if not isinstance(raw, dict):
                raise ValueError("config.yaml must contain a top-level mapping")
            raw_config = raw

    merged = _deep_merge(_DEFAULT_CONFIG, raw_config)
    polylines = _normalise_polylines(merged["counter"]["Polylines"])
    crossing_directions = _normalise_crossing_directions(merged["counter"]["CrossingDirections"], len(polylines))
    size_sanity = _normalise_size_sanity(merged["counter"].get("size_sanity", {}))

    return AppConfig(
        video=VideoConfig(
            path=str(merged["video"]["path"]),
            loop=bool(merged["video"]["loop"]),
        ),
        model=ModelConfig(
            path=str(merged["model"]["path"]),
            classes=_normalise_classes(merged["model"].get("classes")),
            task=(str(merged["model"]["task"]) if merged["model"].get("task") else None),
            persist_tracking=bool(merged["model"]["persist_tracking"]),
            verbose=bool(merged["model"]["verbose"]),
            skip_frame=_normalise_skip_frame(merged["model"].get("skip_frame", 1)),
            conf_threshold=_normalise_threshold(merged["model"].get("conf_threshold", 0.25), "model.conf_threshold"),
            iou_threshold=_normalise_threshold(merged["model"].get("iou_threshold", 0.45), "model.iou_threshold"),
        ),
        counter=CounterConfig(
            polylines=polylines,
            crossing_directions=crossing_directions,
            crossing_order=_normalise_crossing_order(merged["counter"]["CrossingOrder"], len(polylines)),
            unit=_normalise_unit(merged["counter"]["Unit"]),
            anchor_point=_normalise_anchor_point(merged["counter"]["AnchorPoint"]),
            min_hits=int(merged["counter"]["min_hits"]),
            state_threshold=int(merged["counter"]["state_threshold"]),
            size_sanity_enabled=bool(size_sanity["enabled"]),
            min_width_ratio=float(size_sanity["min_width_ratio"]),
            max_width_ratio=float(size_sanity["max_width_ratio"]),
            min_height_ratio=float(size_sanity["min_height_ratio"]),
            max_height_ratio=float(size_sanity["max_height_ratio"]),
            min_area_ratio=float(size_sanity["min_area_ratio"]),
            max_area_ratio=float(size_sanity["max_area_ratio"]),
            min_aspect_ratio=float(size_sanity["min_aspect_ratio"]),
            max_aspect_ratio=float(size_sanity["max_aspect_ratio"]),
        ),
        stream=StreamConfig(
            jpeg_quality=int(merged["stream"]["jpeg_quality"]),
            placeholder_width=int(merged["stream"]["placeholder_width"]),
            placeholder_height=int(merged["stream"]["placeholder_height"]),
        ),
        output_video=OutputVideoConfig(
            enabled=bool(merged["output_video"]["enabled"]),
            path=str(merged["output_video"]["path"]),
            codec=str(merged["output_video"]["codec"]),
            fps=(
                float(merged["output_video"]["fps"])
                if merged["output_video"].get("fps") not in (None, "")
                else None
            ),
            write_when_paused=bool(merged["output_video"]["write_when_paused"]),
        ),
        server=ServerConfig(
            host=str(merged["server"]["host"]),
            port=int(merged["server"]["port"]),
            reload=bool(merged["server"]["reload"]),
        ),
    )


def _debug_main() -> None:
    """Load config.yaml and print the resolved configuration as JSON."""
    config = load_config()
    print(
        json.dumps(
            {
                "video": config.video.__dict__,
                "model": config.model.__dict__,
                "counter": config.counter.__dict__,
                "stream": config.stream.__dict__,
                "output_video": config.output_video.__dict__,
                "server": config.server.__dict__,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _debug_main()
