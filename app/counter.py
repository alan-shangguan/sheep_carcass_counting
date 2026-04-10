"""
app/counter.py
--------------
Directed polyline crossing counter.

Each polyline is treated as a line segment that splits the frame into two
sides. The configured CrossingDirections describe object motion across those
sides, not whether the polyline points themselves numerically increase from
left to right.

Looking from the polyline's first point toward its second point:

- the left-hand side of the line is the positive side
- the right-hand side of the line is the negative side

A track is considered to have crossed a polyline when its anchor point moves
from the left-hand side to the right-hand side for a left-to-right crossing,
or the reverse for right-to-left. A count is emitted only after the track
crosses all configured
polylines in CrossingOrder.

Per-track memory stores a UUID, one rolling 5-state side history per polyline,
and forward/reverse order progress. The special key `__last_event__` is
written when an ordered sequence completes.
"""

from __future__ import annotations

from uuid import uuid4


_SIDE_RIGHT = "right"
_SIDE_LEFT = "left"
_SIDE_ON = "on"


def _anchor_selector(anchor_point: str):
    if anchor_point == "bottomcenter":
        return lambda bbox: (((bbox[0] + bbox[2]) / 2.0), bbox[3])
    if anchor_point == "bottomright":
        return lambda bbox: (bbox[2], bbox[3])
    if anchor_point == "topcenter":
        return lambda bbox: (((bbox[0] + bbox[2]) / 2.0), bbox[1])
    raise ValueError("Unsupported anchor_point")


def _pixel_point(point: tuple[float, float], frame_w: int, frame_h: int) -> tuple[float, float]:
    return (point[0] * float(frame_w), point[1] * float(frame_h))


def _side_of_polyline(
    anchor: tuple[float, float],
    polyline: list[tuple[float, float]],
    frame_w: int,
    frame_h: int,
) -> str:
    start = _pixel_point(polyline[0], frame_w, frame_h)
    end = _pixel_point(polyline[1], frame_w, frame_h)
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    signed_area = (delta_x * (anchor[1] - start[1])) - (delta_y * (anchor[0] - start[0]))
    line_length = max((delta_x * delta_x + delta_y * delta_y) ** 0.5, 1.0)
    tolerance = line_length * 3.0

    if signed_area > tolerance:
        return _SIDE_LEFT
    if signed_area < -tolerance:
        return _SIDE_RIGHT
    return _SIDE_ON


def _line_crossing_delta(first_side: str, last_side: str) -> int:
    if first_side == _SIDE_LEFT and last_side == _SIDE_RIGHT:
        return 1
    if first_side == _SIDE_RIGHT and last_side == _SIDE_LEFT:
        return -1
    return 0


def _advance_progress(memory: dict, progress_key: str, ordered_lines: list[int], line_index: int) -> bool:
    progress = int(memory.get(progress_key, 0))
    expected_line = ordered_lines[progress] if progress < len(ordered_lines) else None

    if line_index == expected_line:
        progress += 1
    elif line_index == ordered_lines[0]:
        progress = 1
    else:
        progress = 0

    memory[progress_key] = progress
    if progress == len(ordered_lines):
        memory[progress_key] = 0
        return True
    return False


def _normalise_direction(direction: str) -> str:
    normalised = direction.replace("-", "_")
    if normalised not in {"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "any"}:
        raise ValueError("Unsupported tripwire direction")
    return normalised


def process_frame(
    tracks: list[dict],
    track_memory: dict,
    frame_w: int,
    frame_h: int,
    *,
    polylines: list[list[tuple[float, float]]],
    crossing_directions: list[str],
    crossing_order: list[int] | tuple[int, int],
    unit: str = "normalized",
    anchor_point: str = "topcenter",
    min_hits: int = 3,
) -> int:
    """
    Evaluate one video frame and return the signed count delta.

    min_hits is retained for backward compatibility and is intentionally
    ignored by the 5-state logic.
    """
    _ = min_hits

    if unit != "normalized":
        raise ValueError("Only normalized tripwire coordinates are supported")

    ordered_lines = [int(item) for item in crossing_order]
    reverse_order = list(reversed(ordered_lines))
    anchor_selector = _anchor_selector(anchor_point)
    normalised_directions = [_normalise_direction(direction) for direction in crossing_directions]

    count_increment = 0

    for track in tracks:
        tid = int(track["id"])
        x1, y1, x2, y2 = track["bbox"]
        anchor = anchor_selector((x1, y1, x2, y2))

        if tid not in track_memory:
            track_memory[tid] = {
                "object_uuid": str(uuid4()),
                "line_state_history": {},
                "forward_progress": 0,
                "reverse_progress": 0,
            }

        mem = track_memory[tid]
        line_state_history = mem.setdefault("line_state_history", {})
        line_events: list[dict] = []

        for line_index, polyline in enumerate(polylines, start=1):
            side = _side_of_polyline(anchor, polyline, frame_w, frame_h)
            side_history = line_state_history.setdefault(line_index, [])
            if side == _SIDE_ON:
                continue

            side_history.append(side)
            if len(side_history) > 5:
                side_history.pop(0)
            if len(side_history) < 5:
                continue

            first_side = side_history[0]
            last_side = side_history[-1]
            line_delta = _line_crossing_delta(first_side, last_side)
            if line_delta == 0:
                continue

            side_history[:] = [last_side]
            line_events.append(
                {
                    "line_index": line_index,
                    "line_delta": line_delta,
                    "direction": normalised_directions[line_index - 1],
                    "first_zone": first_side,
                    "last_zone": last_side,
                }
            )

        if not line_events:
            continue

        forward_position = {line_index: position for position, line_index in enumerate(ordered_lines)}
        reverse_position = {line_index: position for position, line_index in enumerate(reverse_order)}
        line_events.sort(
            key=lambda event: (
                0 if event["line_delta"] > 0 else 1,
                forward_position.get(event["line_index"], len(ordered_lines))
                if event["line_delta"] > 0
                else reverse_position.get(event["line_index"], len(reverse_order)),
            )
        )

        for event in line_events:
            if event["line_delta"] > 0:
                mem["reverse_progress"] = 0
                if _advance_progress(mem, "forward_progress", ordered_lines, event["line_index"]):
                    count_increment += 1
                    track_memory["__last_event__"] = {
                        "track_id": tid,
                        "object_uuid": mem["object_uuid"],
                        "delta": 1,
                        "direction": event["direction"],
                        "first_zone": event["first_zone"],
                        "last_zone": event["last_zone"],
                        "line_index": event["line_index"],
                        "crossing_order": ordered_lines,
                    }
            else:
                mem["forward_progress"] = 0
                if _advance_progress(mem, "reverse_progress", reverse_order, event["line_index"]):
                    count_increment -= 1
                    track_memory["__last_event__"] = {
                        "track_id": tid,
                        "object_uuid": mem["object_uuid"],
                        "delta": -1,
                        "direction": event["direction"],
                        "first_zone": event["first_zone"],
                        "last_zone": event["last_zone"],
                        "line_index": event["line_index"],
                        "crossing_order": reverse_order,
                    }

    return count_increment


def _debug_main() -> None:
    """Run a deterministic crossing simulation."""
    frame_w = 100
    frame_h = 100
    memory: dict = {}
    polylines = [
        [(0.0, 0.40), (1.0, 0.40)],
        [(0.0, 0.20), (1.0, 0.20)],
    ]
    crossing_directions = ["left-to-right", "left-to-right"]
    crossing_order = [1, 2]
    track_sequence = [
        {"id": 1, "bbox": (40, 55, 50, 65)},
        {"id": 1, "bbox": (40, 48, 50, 58)},
        {"id": 1, "bbox": (40, 38, 50, 48)},
        {"id": 1, "bbox": (40, 28, 50, 38)},
        {"id": 1, "bbox": (40, 18, 50, 28)},
        {"id": 1, "bbox": (40, 10, 50, 20)},
    ]

    total = 0
    for index, track in enumerate(track_sequence, start=1):
        increment = process_frame(
            [track],
            memory,
            frame_w,
            frame_h,
            polylines=polylines,
            crossing_directions=crossing_directions,
            crossing_order=crossing_order,
            unit="normalized",
            anchor_point="bottomcenter",
            min_hits=1,
        )
        total += increment
        print(f"frame={index} increment={increment} total={total} memory={memory}")


if __name__ == "__main__":
    _debug_main()
