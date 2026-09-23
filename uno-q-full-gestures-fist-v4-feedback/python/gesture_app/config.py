"""Strict configuration loading, with paths relative to the config file."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .swipe_v3 import DEFAULTS as SWIPE_DEFAULTS


class ConfigError(ValueError):
    pass


def _object(value, name, keys, optional=()):
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be an object")
    unknown = value.keys() - set(keys) - set(optional)
    missing = set(keys) - value.keys()
    if unknown or missing:
        raise ConfigError(f"{name}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return value


def _number(value, name, minimum, maximum, integer=False):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be {'an integer' if integer else 'a number'} in [{minimum}, {maximum}]")


def _boolean(value, name):
    if type(value) is not bool:
        raise ConfigError(f"{name} must be true or false")


@dataclass(frozen=True)
class Config:
    path: Path
    data: dict

    @property
    def backend(self):
        return self.data["backend"]

    @property
    def runtime(self):
        return self.data.get("runtime", "app_lab")

    @property
    def integrated_model_path(self):
        model = self.data.get("integrated", {}).get("model_path", "models/hand-gestures.eim")
        return (self.path.parent / model).resolve()

def load_config(path):
    path = Path(path).resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    _object(data, "config", ["backend", "camera", "recognition", "led", "buzzer"], ["runtime", "integrated", "diagnostics", "swipe", "v_gesture", "fist_gesture"])
    if data["backend"] != "integrated":
        raise ConfigError('backend must be "integrated"')
    if data.get("runtime", "app_lab") not in ("app_lab", "standalone"):
        raise ConfigError('runtime must be "app_lab" or "standalone"')
    if "integrated" in data:
        integrated = _object(data["integrated"], "integrated", [], ["model_path", "resize_mode", "jpeg_quality"])
        if "model_path" in integrated and (not isinstance(integrated["model_path"], str) or not integrated["model_path"].strip()):
            raise ConfigError("integrated.model_path must be a nonempty path")
        if integrated.get("resize_mode", "model") not in ("model", "letterbox"):
            raise ConfigError('integrated.resize_mode must be "model" or "letterbox"')
        _number(integrated.get("jpeg_quality", 90), "integrated.jpeg_quality", 50, 100, integer=True)

    camera = _object(data["camera"], "camera", ["device", "width", "height", "fps", "mirror", "timeout_seconds"])
    device = camera["device"]
    if not ((type(device) is int and device >= 0) or (isinstance(device, str) and device.startswith(("/dev/video", "/dev/v4l/")))):
        raise ConfigError("camera.device must be a nonnegative integer or a Linux /dev/video or /dev/v4l/ device path")
    for key in ("width", "height"):
        _number(camera[key], f"camera.{key}", 16, 4096, integer=True)
    _number(camera["fps"], "camera.fps", 1, 60, integer=True)
    _number(camera["timeout_seconds"], "camera.timeout_seconds", 0.1, 60)
    _boolean(camera["mirror"], "camera.mirror")

    recognition = _object(data["recognition"], "recognition", ["min_confidence", "stable_frames", "repeat_seconds"], ["fast_confidence"])
    _number(recognition["min_confidence"], "recognition.min_confidence", 0, 1)
    _number(recognition["stable_frames"], "recognition.stable_frames", 1, 100, integer=True)
    _number(recognition["repeat_seconds"], "recognition.repeat_seconds", 0.1, 3600)
    if "fast_confidence" in recognition and recognition["fast_confidence"] is not None:
        _number(recognition["fast_confidence"], "recognition.fast_confidence", recognition["min_confidence"], 1)

    swipe = _object(data.get("swipe", {}), "swipe", [], SWIPE_DEFAULTS)
    swipe = data["swipe"] = {**SWIPE_DEFAULTS, **swipe}
    for key in ("enabled",):
        _boolean(swipe[key], f"swipe.{key}")
    for key in ("min_confidence", "distance_fraction", "max_vertical_fraction"):
        _number(swipe[key], f"swipe.{key}", .01, 1)
    for key in ("target_pair_seconds", "min_pair_seconds", "max_pair_seconds",
                "cooldown_seconds", "max_result_age_seconds"):
        _number(swipe[key], f"swipe.{key}", .05, 10)
    _number(swipe["max_size_ratio"], "swipe.max_size_ratio", 1, 4)
    if not swipe["min_pair_seconds"] <= swipe["target_pair_seconds"] <= swipe["max_pair_seconds"]:
        raise ConfigError("swipe pair timing must satisfy min <= target <= max")
    if swipe["enabled"] and data.get("runtime", "app_lab") != "app_lab":
        raise ConfigError("Swipe tracking currently requires the integrated App Lab runtime")

    v_gesture = data.setdefault("v_gesture", {"enabled": False, "min_confidence": 0.5,
        "stable_frames": 2, "max_gap_seconds": 0.4, "release_seconds": 0.4,
        "max_result_age_seconds": 0.75})
    _object(v_gesture, "v_gesture", ["enabled", "min_confidence", "stable_frames",
            "max_gap_seconds", "release_seconds", "max_result_age_seconds"])
    _boolean(v_gesture["enabled"], "v_gesture.enabled")
    _number(v_gesture["min_confidence"], "v_gesture.min_confidence", 0, 1)
    _number(v_gesture["stable_frames"], "v_gesture.stable_frames", 1, 10, integer=True)
    for key in ("max_gap_seconds", "release_seconds", "max_result_age_seconds"):
        _number(v_gesture[key], f"v_gesture.{key}", 0.05, 10)
    if v_gesture["enabled"] and data.get("runtime", "app_lab") != "app_lab":
        raise ConfigError("V-sign feedback requires the App Lab runtime")

    fist_gesture = data.setdefault("fist_gesture", {"enabled": False, "min_confidence": 0.5,
        "stable_frames": 2, "max_gap_seconds": 0.4, "release_seconds": 0.4,
        "max_result_age_seconds": 0.75})
    _object(fist_gesture, "fist_gesture", ["enabled", "min_confidence", "stable_frames",
            "max_gap_seconds", "release_seconds", "max_result_age_seconds"])
    _boolean(fist_gesture["enabled"], "fist_gesture.enabled")
    _number(fist_gesture["min_confidence"], "fist_gesture.min_confidence", 0, 1)
    _number(fist_gesture["stable_frames"], "fist_gesture.stable_frames", 1, 10, integer=True)
    for key in ("max_gap_seconds", "release_seconds", "max_result_age_seconds"):
        _number(fist_gesture[key], f"fist_gesture.{key}", 0.05, 10)
    if fist_gesture["enabled"] and data.get("runtime", "app_lab") != "app_lab":
        raise ConfigError("Closed-fist feedback requires the App Lab runtime")
    if sum((swipe["enabled"], v_gesture["enabled"], fist_gesture["enabled"])) > 1:
        raise ConfigError("Only one Skip trigger may be enabled")

    led = _object(data["led"], "led", ["enabled", "gesture_colors", "heartbeat_seconds"])
    _boolean(led["enabled"], "led.enabled")
    colors = led["gesture_colors"]
    if not isinstance(colors, dict) or not all(
        isinstance(gesture, str) and gesture and gesture != "None" and color in ("red", "green")
        for gesture, color in colors.items()
    ):
        raise ConfigError('led.gesture_colors must map gesture names to "red" or "green"')
    _number(led["heartbeat_seconds"], "led.heartbeat_seconds", 0.1, 0.75)
    buzzer = _object(data["buzzer"], "buzzer", ["enabled"])
    _boolean(buzzer["enabled"], "buzzer.enabled")
    diagnostics = data.setdefault("diagnostics", {"startup_feedback_test": False, "interval_seconds": 2.0})
    _object(diagnostics, "diagnostics", ["startup_feedback_test", "interval_seconds"])
    _boolean(diagnostics["startup_feedback_test"], "diagnostics.startup_feedback_test")
    _number(diagnostics["interval_seconds"], "diagnostics.interval_seconds", 0.1, 60)
    return Config(path, data)
