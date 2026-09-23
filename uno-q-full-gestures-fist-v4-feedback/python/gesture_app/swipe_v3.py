"""Two-timepoint V-sign swipe tracking from built-in detector boxes.

Select camera timestamps roughly 300 ms apart, not a fixed frame count.
No additional neural network or landmark model is used.
"""

from collections import deque
from dataclasses import asdict, dataclass
import logging
import math

LOG = logging.getLogger(__name__)

DEFAULTS = {
    "enabled": False,
    "min_confidence": 0.25,
    "distance_fraction": 0.20,
    "target_pair_seconds": 0.30,
    "min_pair_seconds": 0.15,
    "max_pair_seconds": 0.30,
    "max_vertical_fraction": 0.08,
    "max_result_age_seconds": 0.75,
    "max_size_ratio": 1.6,
    "cooldown_seconds": 1.0,
}


@dataclass(frozen=True)
class SwipeEvent:
    duration_ms: int
    distance_fraction: float
    confidence: float
    direction: str
    source_pose: str = "Victory"

    def to_dict(self):
        return {"type": "swipe", "action": "skip", "gesture": "Swipe", **asdict(self)}


@dataclass(frozen=True)
class _Point:
    seconds: float
    x: float
    y: float
    width: float
    height: float
    confidence: float


def _overlap(a, b):
    """Intersection-over-union for two normalized xyxy boxes."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


class SwipeDetector:
    def __init__(self, options=None):
        self.options = {**DEFAULTS, **(options or {})}
        self._points = deque()
        self._last_timestamp = None
        self._cooldown_until = float("-inf")
        self._last_progress = float("-inf")
        self._last_warning = float("-inf")

    def _clear(self):
        self._points.clear()

    def _candidate(self, detections):
        candidates = []
        for detection in detections:
            box = detection.bounding_box
            if detection.confidence < self.options["min_confidence"] or box is None:
                continue
            if not all(math.isfinite(v) for v in box):
                continue
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                continue
            candidates.append(detection)
        if not candidates:
            return None, False
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        primary = candidates[0]
        # Duplicate labels on one box are not two hands. Distinct plausible
        # boxes are ambiguous, even if one is not labelled Victory.
        competing = any(
            _overlap(item.bounding_box, primary.bounding_box) < 0.50
            for item in candidates[1:]
        )
        return primary, competing

    def update(self, detections, timestamp_ms, now_ms=None):
        """Use captured-frame time and hand-box centers; return one Skip event."""
        now = timestamp_ms / 1000.0
        if self._last_timestamp is not None and now <= self._last_timestamp:
            return None
        self._last_timestamp = now
        if (
            now_ms is not None
            and now_ms - timestamp_ms > self.options["max_result_age_seconds"] * 1000
        ):
            self._clear()
            if now - self._last_warning >= 5:
                LOG.warning(
                    "Ignoring stale swipe frame: %.0fms old", now_ms - timestamp_ms
                )
                self._last_warning = now
            return None
        detection, competing = self._candidate(detections)
        if competing:
            self._clear()
            if now - self._last_warning >= 2:
                LOG.info("Swipe pair rejected: multiple plausible hands")
                self._last_warning = now
            return None
        if detection is None:
            if (
                self._points
                and now - self._points[-1].seconds > self.options["max_pair_seconds"]
            ):
                self._clear()
            return None
        if detection.gesture != "Victory":
            self._clear()
            return None
        if now < self._cooldown_until:
            self._clear()
            return None

        x1, y1, x2, y2 = detection.bounding_box
        current = _Point(
            now, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, detection.confidence
        )
        LOG.info(
            "V-sign point: t=%.3f x=%.0f%% y=%.0f%% box=%.0f%%x%.0f%% confidence=%.0f%%",
            now,
            current.x * 100,
            current.y * 100,
            current.width * 100,
            current.height * 100,
            current.confidence * 100,
        )
        self._points.append(current)
        while (
            self._points
            and now - self._points[0].seconds > self.options["max_pair_seconds"]
        ):
            self._points.popleft()

        # Pick one earlier endpoint closest to 300 ms, inside tolerance.
        # Intermediate frames may lack detections; they are never counted.
        pairs = [
            (abs(now - point.seconds - self.options["target_pair_seconds"]), point)
            for point in list(self._points)[:-1]
            if self.options["min_pair_seconds"]
            <= now - point.seconds
            <= self.options["max_pair_seconds"]
        ]
        if not pairs:
            if now - self._last_progress >= 1:
                LOG.info(
                    "V-sign candidate: x=%.0f%% y=%.0f%% confidence=%.0f%%; waiting for ~300ms pair",
                    current.x * 100,
                    current.y * 100,
                    current.confidence * 100,
                )
                self._last_progress = now
            return None
        limit = self.options["max_size_ratio"]
        for _timing_error, first in sorted(pairs, key=lambda item: item[0]):
            duration = now - first.seconds
            dx, dy = current.x - first.x, current.y - first.y
            width_ratio = current.width / first.width
            height_ratio = current.height / first.height
            if (
                abs(dx) < self.options["distance_fraction"]
                or abs(dy) > self.options["max_vertical_fraction"]
                or not 1 / limit <= width_ratio <= limit
                or not 1 / limit <= height_ratio <= limit
            ):
                continue
            event = SwipeEvent(
                round(duration * 1000),
                abs(dx),
                min(first.confidence, current.confidence),
                "right" if dx > 0 else "left",
            )
            self._clear()
            self._cooldown_until = now + self.options["cooldown_seconds"]
            return event
        first = min(pairs, key=lambda item: item[0])[1]
        duration = now - first.seconds
        dx, dy = current.x - first.x, current.y - first.y
        LOG.info(
            "V-sign pair: %.0fms apart, dx=%.0f%% dy=%.0f%% size=%.2fx/%.2fx; no swipe",
            duration * 1000,
            dx * 100,
            dy * 100,
            current.width / first.width,
            current.height / first.height,
        )
        return None
