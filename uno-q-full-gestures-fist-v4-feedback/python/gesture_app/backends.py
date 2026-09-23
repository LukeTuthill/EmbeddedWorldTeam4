"""Backend adapters. Import only the selected inference library."""

import importlib
import logging
import math

from .recognition import Detection, normalized_box
from .preprocessing import letterbox

LOG = logging.getLogger(__name__)


def _import_runtime(module, label):
    """Preserve missing native-library and nested-import errors in terminal logs."""
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"{label} import failed in {module}: {type(exc).__name__}: {exc}. "
            "See the traceback for the missing library or incompatible dependency; "
            "this is a runtime import failure, not a missing gesture model."
        ) from exc


INTEGRATED_LABELS = {
    "five": "Open_Palm",
    "good": "Thumb_Up",
    "neut": "Closed_Fist",
    "peace": "Victory",
}


def integrated_results(result, image_size=None):
    # Arduino ObjectDetection returns percentage strings, including values < 1%.
    if not isinstance(result, dict) or not isinstance(result.get("detection"), list):
        raise RuntimeError(
            f"Unexpected Object Detection response (expected a detection list): {result!r}"
        )
    detections = []
    for item in result["detection"]:
        if (
            not isinstance(item, dict)
            or not {"class_name", "confidence"} <= item.keys()
        ):
            raise RuntimeError(f"Invalid Object Detection item: {item!r}")
        raw = item["class_name"]
        confidence = float(item["confidence"]) / 100.0
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid integrated confidence: {item['confidence']!r}")
        if confidence == 0:
            continue
        if raw not in INTEGRATED_LABELS:
            raise RuntimeError(
                f"Unexpected label {raw!r}. Select the hand-gestures model in app.yaml."
            )
        box = normalized_box(item.get("bounding_box_xyxy"), image_size)
        detections.append(
            Detection(
                INTEGRATED_LABELS[raw], confidence, raw_label=raw, bounding_box=box
            )
        )
    return detections


class IntegratedBackend:
    def __init__(self, config):
        try:
            from arduino.app_bricks.object_detection import ObjectDetection
        except ImportError as exc:
            raise RuntimeError(
                "Integrated mode requires Arduino App Lab on the UNO Q, with the Object Detection brick and hand-gestures model."
            ) from exc
        self._cv2 = _import_runtime("cv2", "OpenCV")
        # Keep positive low-confidence results visible to diagnostics. Apply the
        # user's acceptance threshold exactly once, in StableGesture.
        self._detector = ObjectDetection(confidence=0.0)
        info = self._detector.get_model_info()
        if info is None:
            raise RuntimeError(
                "Cannot read the integrated model information; check the inference service logs"
            )
        labels = set(info.labels)
        LOG.info(
            "Integrated model=%s labels=%s input=%sx%s",
            info.name,
            sorted(labels),
            info.image_input_width,
            info.image_input_height,
        )
        if labels != set(INTEGRATED_LABELS):
            raise RuntimeError(
                f"Wrong integrated model labels: {sorted(labels)}. Select model: hand-gestures in app.yaml and restart/rebuild the app."
            )
        self._input_size = (info.image_input_width, info.image_input_height)
        options = config.data.get("integrated", {})
        self._resize_mode = options.get("resize_mode", "model")
        self._jpeg_quality = options.get("jpeg_quality", 90)
        LOG.info(
            "Integrated framing=%s JPEG quality=%s",
            self._resize_mode,
            self._jpeg_quality,
        )

    def recognize(self, bgr_frame, timestamp_ms):
        if self._resize_mode == "letterbox":
            bgr_frame = letterbox(bgr_frame, *self._input_size, self._cv2)
        ok, encoded = self._cv2.imencode(
            ".jpg", bgr_frame, [self._cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not ok:
            raise RuntimeError("Could not encode the camera frame")
        result = self._detector.detect(encoded.tobytes(), image_type="jpg")
        if result is None:
            raise RuntimeError(
                "Arduino inference returned no result. Check the Object Detection service logs in App Lab."
            )
        return integrated_results(result, self._input_size)

    def close(self):
        pass  # The App Lab service lifecycle belongs to App Lab.


def create_backend(config):
    if config.backend != "integrated":
        raise ValueError(f"Unknown backend: {config.backend}")
    if config.runtime == "standalone":
        from .edge_impulse_backend import EdgeImpulseBackend

        return EdgeImpulseBackend(config)
    return IntegratedBackend(config)
