"""UNO Q Full Gestures: integrated static recognition and closed-fist Skip."""

import argparse
from contextlib import ExitStack
import logging
from pathlib import Path
import signal
import sys
import time

from gesture_app.config import ConfigError, load_config

LOG = logging.getLogger("gesture-app")
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def _stop_on_signal(signum, frame):
    raise KeyboardInterrupt


def run(config, camera_test=False, feedback_test=False):
    from gesture_app.camera import Camera

    # Raise out of blocking inference so cleanup also runs on App Lab Stop.
    previous = signal.signal(signal.SIGTERM, _stop_on_signal)
    try:
        with ExitStack() as stack:
            if not camera_test:
                from gesture_app.backends import create_backend
                from gesture_app.output import Outputs, connect_bridge
                from gesture_app.linux_leds import connect_linux_leds
                from gesture_app.recognition import StableGesture
                from gesture_app.diagnostics import RecognitionDiagnostics

                linux_leds = connect_linux_leds(config)
                if linux_leds is not None:
                    stack.callback(linux_leds.close)
                outputs = Outputs(config, connect_bridge(config), linux_leds=linux_leds)
                stack.callback(outputs.close)
                diagnostic_options = config.data.get("diagnostics", {})
                if feedback_test or diagnostic_options.get("startup_feedback_test", False):
                    outputs.self_test()
                if feedback_test:
                    return
                LOG.info("Loading %s recognition backend", config.backend)
                backend = create_backend(config)
                stack.callback(backend.close)
                recognition_options = config.data["recognition"]
                filtering = StableGesture(recognition_options["min_confidence"], recognition_options["stable_frames"],
                                          recognition_options.get("fast_confidence"))
                diagnostics = RecognitionDiagnostics(diagnostic_options.get("interval_seconds", 2.0))
            camera = Camera(config.data["camera"])
            stack.callback(camera.close)
            if camera_test:
                frame, _ = camera.read()
                LOG.info("USB camera OK: received %sx%s pixels", frame.shape[1], frame.shape[0])
                return
            LOG.info("Running %s backend; raised palm = red, thumbs up = green; buzzer enabled=%s. Ctrl+C stops.", config.backend, config.data["buzzer"]["enabled"])
            fist_trigger = None
            if config.data["fist_gesture"]["enabled"]:
                from gesture_app.fist_gesture import FistGestureTrigger
                outputs.check_swipe_firmware()
                fist_trigger = FistGestureTrigger(config.data["fist_gesture"])
                LOG.info("Closed-fist Skip: two fresh built-in Closed_Fist detections; no movement requirement.")
            while True:
                frame, timestamp_ms = camera.read()
                inference_started = time.monotonic()
                frame_age_seconds = max(0, inference_started - timestamp_ms / 1000)
                LOG.debug("Starting inference on camera timestamp=%s", timestamp_ms)
                detections = backend.recognize(frame, timestamp_ms)
                inference_seconds = time.monotonic() - inference_started
                output_started = time.monotonic()
                event = fist_trigger.update(detections, timestamp_ms, now_ms=output_started * 1000) if fist_trigger else None
                stable = filtering.update(detections)
                if event is not None:
                    outputs.fist(event)
                elif stable is not None and stable.gesture == "Closed_Fist":
                    outputs.update(None)
                else:
                    outputs.update(stable)
                output_seconds = time.monotonic() - output_started
                diagnostics.update(detections, stable, filtering, inference_seconds,
                                   frame_age_seconds=frame_age_seconds, output_seconds=output_seconds)
                # Camera.read waits for a new frame. An extra rate-limit sleep
                # adds latency when the camera delivers faster than requested.
    finally:
        signal.signal(signal.SIGTERM, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-config", action="store_true", help="Validate JSON without camera, Arduino, or inference imports")
    test_mode = parser.add_mutually_exclusive_group()
    test_mode.add_argument("--camera-test", action="store_true", help="Capture one frame without starting inference or feedback")
    test_mode.add_argument("--feedback-test", action="store_true", help="Test red/green LEDs and beeps without a camera or inference")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO", help="Terminal logging verbosity (default INFO)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    try:
        config = load_config(args.config)
        if args.check_config:
            LOG.info("Config OK: backend=%s, camera=%r, LED=%s", config.backend, config.data["camera"]["device"], config.data["led"]["enabled"])
            return 0
        run(config, camera_test=args.camera_test, feedback_test=args.feedback_test)
        return 0
    except KeyboardInterrupt:
        LOG.info("Stopped")
        return 0
    except ConfigError as exc:
        LOG.error("%s", exc)
        return 1
    except (RuntimeError, OSError, ImportError, ValueError) as exc:
        LOG.exception("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
