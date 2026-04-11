from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any


DEFAULT_LOG_PATH = "outputs/sheep_counter.log"
_LOGGER_NAME = "sheep_counter.runtime"


def _resolve_log_path() -> Path:
    configured = os.environ.get("APP_LOG_PATH", DEFAULT_LOG_PATH).strip() or DEFAULT_LOG_PATH
    path = Path(configured)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_runtime_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(_resolve_log_path(), mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(event_type: str, **payload: Any) -> None:
    record = {"event": event_type, **payload}
    get_runtime_logger().info(json.dumps(record, ensure_ascii=True, default=str))