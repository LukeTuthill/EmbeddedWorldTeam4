"""Two-finger V-sign swipes from the integrated model's existing detections."""

from collections import deque
from dataclasses import asdict, dataclass
import logging
import math

LOG = logging.getLogger(__name__)
DEFAULTS = {
    "enabled": False, "min_confidence": .45,
    "distance_fraction": .12,
    "max_step_fraction": .25, "max_duration_seconds": 1.0,
    "max_gap_seconds": .45, "max_result_age_seconds": .75,
    "release_seconds": .25, "min_samples": 3,
    "min_speed_fraction_per_second": .20,
    "min_directionality": .70,
    "other_pose_confidence": .50,
}


@dataclass(frozen=True)
class SwipeEvent:
    duration_ms: int
    distance_fraction: float
    confidence: float

    def to_dict(self):
        return {"type": "swipe", "action": "skip", "gesture": "Victory", **asdict(self)}


class SwipeDetector:
    def __init__(self, options=None):
        self.options = {**DEFAULTS, **(options or {})}
        self._points = deque()
        self._last_timestamp = None
        self._last_box = None
        self._release_since = None
        self._release_frames = 0
        self._last_warning = float("-inf")
        self._last_progress = float("-inf")
        self.locked = False

    def _clear_motion(self):
        self._points.clear()
        self._last_box = None

    def _release(self, now):
        self._clear_motion()
        if self._release_since is None:
            self._release_since = now
        self._release_frames += 1
        if self._release_frames >= 2 and now - self._release_since >= self.options["release_seconds"]:
            self.locked = False

    def update(self, detections, timestamp_ms, now_ms=None):
        """Use raw, fresh results, without waiting for static-gesture stability.

        Timestamps refer to camera capture. Brief classifier dropouts retain the
        path; confident other poses, multiple hands, jumps, and stale results do
        not. Observed absence of the V-sign rearms after an event.
        """
        now = timestamp_ms / 1000
        if self._last_timestamp is not None and now <= self._last_timestamp:
            return None
        if self._last_timestamp is not None and now - self._last_timestamp > self.options["max_gap_seconds"]:
            self._clear_motion()
            self._release_since = None
            self._release_frames = 0
        self._last_timestamp = now
        if now_ms is not None and now_ms - timestamp_ms > self.options["max_result_age_seconds"] * 1000:
            self._clear_motion()
            self._release_since = None
            self._release_frames = 0
            if now - self._last_warning >= 5:
                LOG.warning("Ignoring stale swipe position: frame_to_result=%.0fms", now_ms - timestamp_ms)
                self._last_warning = now
            return None
        eligible = [d for d in detections if d.confidence >= self.options["min_confidence"]]
        # Keep short missed frames only before an event. Confident other poses
        # and multiple hands must never contribute motion evidence.
        if len(eligible) != 1 or eligible[0].gesture != "Victory":
            if self.locked:
                self._release(now)
            elif len(eligible) > 1 or any(d.gesture != "Victory"
                                             and d.confidence >= self.options["other_pose_confidence"]
                                             for d in eligible):
                self._clear_motion()
                self._release_since = None
                self._release_frames = 0
            elif self._points and now - self._points[-1][0] > self.options["max_gap_seconds"]:
                self._clear_motion()
            return None
        detection = eligible[0]
        self._release_since = None
        self._release_frames = 0
        box = detection.bounding_box
        if box is None:
            self._clear_motion()
            if now - self._last_warning >= 5:
                LOG.warning("V-sign has no usable position: swipe needs bounding_box_xyxy from the integrated model; static feedback remains available")
                self._last_warning = now
            return None
        x1, y1, x2, y2 = box
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1
        if not all(math.isfinite(v) for v in box) or not (0 <= x <= 1 and 0 <= y <= 1 and width > 0 and height > 0):
            self._clear_motion()
            return None
        if self.locked:
            return None
        if self._points and now - self._points[-1][0] > self.options["max_gap_seconds"]:
            self._clear_motion()
        if self._last_box is not None:
            px, py, pw, ph = self._last_box
            # A sudden position/size jump can be another hand or a bad box.
            if (math.hypot(x - px, y - py) > self.options["max_step_fraction"]
                    or not .5 <= width / pw <= 2 or not .5 <= height / ph <= 2):
                self._clear_motion()
        self._last_box = (x, y, width, height)
        self._points.append((now, x, y, detection.confidence))
        while self._points and now - self._points[0][0] > self.options["max_duration_seconds"]:
            self._points.popleft()
        if len(self._points) < self.options["min_samples"]:
            return None
        points = list(self._points)
        # Try suffixes: a stationary V-sign before a swipe must not dilute
        # movement speed or consume the one-second motion window.
        for start in range(len(points) - self.options["min_samples"] + 1):
            track = points[start:]
            duration = now - track[0][0]
            if duration < .10:
                continue
            distance = math.hypot(x - track[0][1], y - track[0][2])
            steps = [math.hypot(b[1] - a[1], b[2] - a[2])
                     for a, b in zip(track, track[1:])]
            path = sum(steps)
            if (distance >= self.options["distance_fraction"]
                    and distance / duration >= self.options["min_speed_fraction_per_second"]
                    and path > 0
                    and distance / path >= self.options["min_directionality"]
                    and sum(step >= .02 for step in steps) >= 2):
                event = SwipeEvent(round(duration * 1000), distance,
                                   min(p[3] for p in track))
                self.locked = True
                self._clear_motion()
                return event
        if now - self._last_progress >= 1.0:
            distance = math.hypot(x - points[0][1], y - points[0][2])
            LOG.info("V-sign track: %d frames, %.0f%% displacement in %.2fs; waiting for deliberate movement",
                     len(points), distance * 100, now - points[0][0])
            self._last_progress = now
        return None
