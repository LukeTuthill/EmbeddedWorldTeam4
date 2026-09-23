"""Periodic inference status, including frames rejected by the gesture filter."""

import logging
import time

LOG = logging.getLogger(__name__)


class RecognitionDiagnostics:
    def __init__(self, interval_seconds=2.0):
        self._interval = interval_seconds
        self._last_report = float("-inf")
        self._frames = 0

    def update(self, detections, stable, filtering, inference_seconds, now=None,
               frame_age_seconds=0, output_seconds=0):
        now = time.monotonic() if now is None else now
        self._frames += 1
        LOG.debug("Frame %s raw detections: %s", self._frames, [d.to_dict() for d in detections])
        if now - self._last_report < self._interval:
            return
        self._last_report = now
        top = max(detections, key=lambda d: d.confidence, default=None)
        if top is None:
            reason = "no detections from model; show the whole hand toward the camera"
        elif not any(d.gesture != "None" and d.confidence >= filtering.min_confidence for d in detections):
            reason = "no gesture above confidence threshold"
        elif stable is None:
            reason = "waiting for consecutive matching frames"
        elif filtering.progress < filtering.stable_frames:
            reason = "gesture accepted on high-confidence fast path"
        else:
            reason = "gesture accepted"
        LOG.info("Frame=%s inference=%.0fms detections=%s top=%s confidence=%.1f%% threshold=%.1f%% stable=%s/%s: %s",
                 self._frames, inference_seconds * 1000, len(detections), top.gesture if top else "None",
                 top.confidence * 100 if top else 0, filtering.min_confidence * 100,
                 filtering.progress, filtering.stable_frames, reason)
        LOG.info("Timing: frame_age=%.0fms inference=%.0fms feedback=%.0fms frame_to_output=%.0fms",
                 frame_age_seconds * 1000, inference_seconds * 1000, output_seconds * 1000,
                 (frame_age_seconds + inference_seconds + output_seconds) * 1000)
        if inference_seconds >= 1.5:
            LOG.warning("Inference took %.2fs. Increasing camera FPS cannot fix slow model inference; stale positions are ignored for swipe detection.", inference_seconds)
