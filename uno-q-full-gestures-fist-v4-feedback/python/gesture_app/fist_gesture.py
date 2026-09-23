"""Closed-fist Skip trigger using Arduino's built-in hand-gestures detector."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FistGestureEvent:
    confidence: float

    def to_dict(self):
        return {"type": "fist_gesture", "action": "skip", "gesture": "Closed_Fist",
                "confidence": self.confidence, "source": "arduino_hand_gestures"}


class FistGestureTrigger:
    """Emit once for two fresh fist detections; rearm after sustained absence."""

    def __init__(self, options):
        self.options = options
        self._last_timestamp = None
        self._last_fist = None
        self._streak = 0
        self._streak_confidence = 1.0
        self._absence_since = None
        self._armed = True

    def update(self, detections, timestamp_ms, now_ms=None):
        """Inspect fresh camera result; no bounding box or movement required."""
        now = timestamp_ms / 1000.0
        if self._last_timestamp is not None and now <= self._last_timestamp:
            return None
        self._last_timestamp = now
        if now_ms is not None and now_ms - timestamp_ms > self.options["max_result_age_seconds"] * 1000:
            self._streak = 0
            self._last_fist = None
            return None
        fist = max((item for item in detections
                    if item.gesture == "Closed_Fist" and item.confidence >= self.options["min_confidence"]),
                   key=lambda item: item.confidence, default=None)
        if fist is None:
            self._streak = 0
            self._last_fist = None
            if self._absence_since is None:
                self._absence_since = now
            if now - self._absence_since >= self.options["release_seconds"]:
                self._armed = True
            return None
        self._absence_since = None
        if self._last_fist is None or now - self._last_fist > self.options["max_gap_seconds"]:
            self._streak = 0
            self._streak_confidence = 1.0
        self._last_fist = now
        self._streak = min(self._streak + 1, self.options["stable_frames"])
        self._streak_confidence = min(self._streak_confidence, fist.confidence)
        if self._armed and self._streak >= self.options["stable_frames"]:
            self._armed = False
            return FistGestureEvent(self._streak_confidence)
        return None
