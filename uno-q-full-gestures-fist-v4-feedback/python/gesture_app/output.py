"""Terminal logs, JSON events, RGB LEDs, and Modulino buzzer feedback."""

import json
import logging
import time

LOG = logging.getLogger(__name__)
COLOR_CODES = {"off": 0, "red": 1, "green": 2}


class Outputs:
    def __init__(self, config, bridge=None, emit=print, linux_leds=None):
        self._backend = config.backend
        self._owns_bridge = config.runtime == "standalone"
        self._led = config.data["led"]
        self._buzzer = config.data["buzzer"]["enabled"]
        self._linux_leds = linux_leds
        self._repeat = config.data["recognition"]["repeat_seconds"]
        self._bridge, self._emit = bridge, emit
        self._last_key = None
        self._last_event = float("-inf")
        self._last_led_time = float("-inf")
        self._last_led_state = None
        self._bridge_warning_time = float("-inf")
        self._linux_warning_time = float("-inf")
        self._hardware_status = None
        self._animation_until = float("-inf")

    def check_swipe_firmware(self):
        if self._bridge is not None:
            try:
                version = self._bridge.call("gesture_feedback_version", timeout=1)
                if type(version) is not int or version < 3:
                    raise ValueError(f"feedback version {version!r}")
            except Exception as exc:
                raise RuntimeError("Skip feedback requires the matching sketch. Update both Python and sketch, then restart the App Lab app to flash it.") from exc

    def swipe(self, event, now=None):
        now = time.monotonic() if now is None else now
        payload = {"backend": self._backend, "timestamp": time.time(), **event.to_dict(),
                   "led_color": "blue" if self._led["enabled"] else "off"}
        self._emit(json.dumps(payload), flush=True)
        LOG.info("Gesture=%s action=skip confidence=%.0f%%",
                 payload["gesture"], event.confidence * 100)
        if self._bridge is not None:
            try:
                status = self._bridge.call("show_skip", self._led["enabled"], self._buzzer, timeout=.5)
                self._report_hardware_status(status)
            except Exception as exc:
                LOG.warning("Skip feedback Bridge call failed: %s", exc)
        if self._linux_leds is not None:
            try:
                self._linux_leds.set_color(payload["led_color"])
            except OSError as exc:
                LOG.warning("Linux RGB Skip update failed: %s", exc)
        self._animation_until = now + 1.0
        self._last_led_state = None
        self._last_led_time = float("-inf")

    def victory(self, event, now=None):
        """V-sign Skip feedback; legacy swipe method retains hardware protocol."""
        self.swipe(event, now=now)

    def fist(self, event, now=None):
        """Closed-fist Skip feedback; reuse installed MCU show_skip protocol."""
        self.swipe(event, now=now)

    def _report_hardware_status(self, status):
        if type(status) is int and status != self._hardware_status:
            LOG.info("Modulino detection at startup: Pixels=%s Buzzer=%s", "found" if status & 1 else "missing", "found" if status & 2 else "missing")
            if self._led["enabled"] and not status & 1:
                LOG.warning("Modulino Pixels missing: check its Qwiic cable and restart the sketch")
            if self._buzzer and not status & 2:
                LOG.warning("Modulino Buzzer missing: check its Qwiic cable and restart the sketch")
            self._hardware_status = status

    def self_test(self):
        """Exercise feedback without camera/inference; never emit fake gestures."""
        if self._bridge is None and self._linux_leds is None:
            LOG.info("Feedback self-test skipped: hardware output disabled")
            return
        try:
            # Reset first so a previously held gesture cannot suppress its beep.
            if self._bridge is not None:
                status = self._bridge.call("set_gesture_feedback", 0, False, False, timeout=1)
                self._report_hardware_status(status)
            for color in ("red", "green"):
                LOG.info("SELF-TEST: LEDs=%s buzzer=%s (no camera/inference)",
                         color if self._led["enabled"] else "disabled",
                         ("two beeps" if color == "red" else "one beep") if self._buzzer else "disabled")
                if self._bridge is not None:
                    self._bridge.call("set_gesture_feedback", COLOR_CODES[color], self._led["enabled"], self._buzzer, timeout=1)
                if self._linux_leds is not None:
                    self._linux_leds.set_color(color if self._led["enabled"] else "off")
                time.sleep(.6)
        finally:
            try:
                if self._bridge is not None:
                    self._bridge.call("set_gesture_feedback", 0, False, False, timeout=1)
            finally:
                if self._linux_leds is not None:
                    self._linux_leds.set_color("off")
        LOG.info("Feedback self-test commands completed; LEDs off and buzzer silent")

    def update(self, detection, now=None):
        now = time.monotonic() if now is None else now
        if now < self._animation_until:
            if detection is None or detection.gesture not in ("Open_Palm", "Thumb_Up"):
                return  # Ordinary heartbeats must not cancel the MCU animation.
            # Accepted Stop/Thumb_Up keeps its normal response time even during
            # a swipe animation. The MCU's static RPC cancels the animation.
            self._animation_until = float("-inf")
        key = (detection.gesture, detection.hand) if detection else None
        color = self._led["gesture_colors"].get(detection.gesture, "off") if detection else "off"
        led_color = color if self._led["enabled"] else "off"
        if key != self._last_key or (detection and now - self._last_event >= self._repeat):
            event = {"backend": self._backend, "timestamp": time.time(), "gesture": "None", "confidence": 0.0, "hand": None, "raw_label": None}
            if detection:
                event.update(detection.to_dict())
            event["led_color"] = led_color
            self._emit(json.dumps(event), flush=True)
            LOG.info("Gesture=%s confidence=%.0f%% hand=%s LEDs=%s",
                     event["gesture"], event["confidence"] * 100, event["hand"] or "unknown", led_color)
            self._last_event = now
        self._last_key = key
        changed = color != self._last_led_state
        if changed or now - self._last_led_time >= self._led["heartbeat_seconds"]:
            if self._bridge is not None:
                try:
                    status = self._bridge.call("set_gesture_feedback", COLOR_CODES[color], self._led["enabled"], self._buzzer, timeout=0.5)
                    self._report_hardware_status(status)
                    if changed and color != "off" and self._buzzer:
                        LOG.info("Buzzer requested: %s short beep(s)", 2 if color == "red" else 1)
                except Exception as exc:
                    if now - self._bridge_warning_time >= 5:
                        LOG.warning("Feedback Bridge call failed (recognition continues): %s", exc)
                        self._bridge_warning_time = now
            if self._linux_leds is not None:
                try:
                    self._linux_leds.set_color(led_color)
                except OSError as exc:
                    if now - self._linux_warning_time >= 5:
                        LOG.warning("Linux RGB LED update failed: %s", exc)
                        self._linux_warning_time = now
            self._last_led_state = color
            self._last_led_time = now

    def close(self):
        if self._bridge is not None:
            try:
                self._bridge.call("set_gesture_feedback", 0, False, False, timeout=0.5)
            except Exception:
                LOG.warning("Could not clear MCU LEDs/Pixels/buzzer through Bridge; the sketch watchdog will clear them")
            finally:
                if self._owns_bridge:
                    self._bridge.disconnect()


def connect_bridge(config):
    if not config.data["led"]["enabled"] and not config.data["buzzer"]["enabled"]:
        return None
    if config.runtime == "standalone":
        try:
            from arduino.router_bridge import Bridge as RouterBridge
        except ImportError as exc:
            raise RuntimeError("Install arduino-ide/requirements.txt for the standalone Arduino Router Bridge") from exc
        bridge = RouterBridge("unix:///var/run/arduino-router.sock")
        try:
            if not bridge.connect(timeout=5):
                raise RuntimeError("Cannot connect to arduino-router. Run this program on the UNO Q and check: systemctl status arduino-router")
            # Verify the matching MCU sketch before starting the camera.
            bridge.call("set_gesture_feedback", 0, False, False, timeout=1)
            return bridge
        except BaseException:
            bridge.disconnect()
            raise
    try:
        from arduino.app_utils import Bridge
        return Bridge
    except ImportError as exc:
        raise RuntimeError("Feedback requires Arduino App Lab and the supplied MCU sketch. Use the integrated app on the UNO Q.") from exc
