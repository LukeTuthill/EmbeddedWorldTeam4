"""Common results and a filter operating on fresh inference frames only."""

from dataclasses import asdict, dataclass
import math


def normalized_box(box, image_size):
    """Model-space xyxy box -> normalized coordinates; bad boxes can't swipe."""
    if box is None or image_size is None:
        return None
    try:
        x1, y1, x2, y2 = map(float, box)
        width, height = image_size
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2, width, height)):
            return None
        if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
            return None
        return (x1 / width, y1 / height, x2 / width, y2 / height)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Detection:
    gesture: str
    confidence: float
    hand: str | None = None
    raw_label: str | None = None
    bounding_box: tuple[float, float, float, float] | None = None

    def to_dict(self):
        return asdict(self)


class StableGesture:
    """Select the strongest gesture; optionally accept a strong first frame.

    Multi-hand results are available from the backends, but this LED demo has one
    active gesture. A low-confidence/empty frame immediately clears its state.
    """

    def __init__(self, min_confidence, stable_frames, fast_confidence=None):
        self.min_confidence = min_confidence
        self.stable_frames = stable_frames
        self.fast_confidence = fast_confidence
        self._candidate = None
        self._count = 0
        self._accepted = False

    def update(self, detections):
        eligible = [d for d in detections if d.gesture != "None" and d.confidence >= self.min_confidence]
        best = max(eligible, key=lambda d: d.confidence, default=None)
        key = (best.gesture, best.hand) if best else None
        if key is None:
            self._candidate, self._count = None, 0
            self._accepted = False
            return None
        if key != self._candidate:
            self._candidate, self._count = key, 1
            self._accepted = False
        else:
            self._count += 1
        # Once a high-confidence frame establishes a gesture, a following
        # eligible weaker frame must not clear it and replay the buzzer.
        self._accepted = self._accepted or self._count >= self.stable_frames or (
            self.fast_confidence is not None and best.confidence >= self.fast_confidence
        )
        return best if self._accepted else None

    @property
    def progress(self):
        return min(self._count, self.stable_frames)
