"""Run Arduino's same hand-gestures EIM locally without App Lab services."""

from contextlib import contextmanager
import logging
import math
import os
import platform
import signal
import threading
import time

from .backends import INTEGRATED_LABELS
from .recognition import Detection
from .preprocessing import letterbox

LOG = logging.getLogger(__name__)


@contextmanager
def _deadline(seconds):
    # The Linux SDK can otherwise wait indefinitely for a model that failed to
    # start. In this app inference runs on the main thread of the Linux MPU.
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Edge Impulse inference must run on the main thread")

    def expired(signum, frame):
        raise TimeoutError("Edge Impulse model timed out; check its Linux dependencies and board architecture")

    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(.001, previous_timer[0] - (time.monotonic() - started)), previous_timer[1])


def edge_impulse_results(response):
    if not isinstance(response, dict) or "bounding_boxes" not in response.get("result", {}):
        raise RuntimeError("Edge Impulse did not return object detections; use the hand-gestures model")
    detections = []
    for item in response["result"]["bounding_boxes"]:
        score = float(item["value"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid Edge Impulse confidence: {score}")
        if score == 0:
            continue  # Some models emit empty zero-confidence bounding boxes.
        raw = item["label"]
        if raw not in INTEGRATED_LABELS:
            raise RuntimeError(f"Unexpected label {raw!r}; use the hand-gestures model")
        detections.append(Detection(INTEGRATED_LABELS[raw], score, raw_label=raw))
    return detections


class EdgeImpulseBackend:
    def __init__(self, config):
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise RuntimeError("The integrated EIM model requires 64-bit ARM Linux on the UNO Q. Run this app on the board.")
        model = config.integrated_model_path
        if not model.is_file():
            raise RuntimeError(f"Missing integrated model: {model}. Run scripts/download_ide_models.py.")
        if not os.access(model, os.X_OK):
            raise RuntimeError(f"Model is not executable: {model}. Run scripts/download_ide_models.py on the UNO Q to set its executable bit.")
        try:
            import cv2
            from edge_impulse_linux.image import ImageImpulseRunner
        except ImportError as exc:
            raise RuntimeError("Install arduino-ide/requirements.txt in the UNO Q's Python environment") from exc
        self._cv2 = cv2
        self._runner = ImageImpulseRunner(str(model))
        try:
            with _deadline(30):
                info = self._runner.init()
            labels = set(info["model_parameters"]["labels"])
            if labels != set(INTEGRATED_LABELS):
                raise RuntimeError(f"Wrong integrated model labels: {sorted(labels)}; expected hand-gestures")
            self._resize_mode = config.data.get("integrated", {}).get("resize_mode", "model")
            self._input_size = (info["model_parameters"].get("image_input_width", 0),
                                info["model_parameters"].get("image_input_height", 0))
            LOG.info("Integrated framing=%s input=%sx%s", self._resize_mode, *self._input_size)
        except BaseException:
            self.close()
            raise

    def recognize(self, bgr_frame, timestamp_ms):
        if self._resize_mode == "letterbox":
            bgr_frame = letterbox(bgr_frame, *self._input_size, self._cv2)
        rgb = self._cv2.cvtColor(bgr_frame, self._cv2.COLOR_BGR2RGB)
        with _deadline(30):
            features, _ = self._runner.get_features_from_image_auto_studio_settings(rgb)
            response = self._runner.classify(features)
        return edge_impulse_results(response)

    def close(self):
        runner, self._runner = self._runner, None
        if runner is not None:
            try:
                runner.stop()
            except (OSError, RuntimeError) as exc:
                LOG.warning("Could not fully stop the Edge Impulse runner: %s", exc)
