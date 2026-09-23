"""One webcam owner; keep only its newest frame when inference is slower."""

import logging
import sys
import threading
import time

LOG = logging.getLogger(__name__)


class Camera:
    def __init__(self, options):
        import cv2

        self._options, self._cv2 = options, cv2
        self._condition = threading.Condition()
        self._stopped = threading.Event()
        self._latest = None
        self._sequence = 0
        self._consumed = 0
        self._error = None
        api = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(options["device"], api)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Cannot open USB camera {options['device']!r}. Check the powered hub, device path, permissions, and other camera apps.")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, options["width"])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, options["height"])
        self._capture.set(cv2.CAP_PROP_FPS, options["fps"])
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        LOG.info("Camera requested %sx%s @ %s fps; reported %sx%s @ %.1f fps",
                 options["width"], options["height"], options["fps"],
                 int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), self._capture.get(cv2.CAP_PROP_FPS))
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="usb-camera")
        self._thread.start()

    def _read_loop(self):
        try:
            while not self._stopped.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    raise RuntimeError("USB camera stopped delivering frames; reconnect it and restart the app")
                timestamp_ms = time.monotonic_ns() // 1_000_000
                if self._options["mirror"]:
                    frame = self._cv2.flip(frame, 1)
                with self._condition:
                    self._latest = (frame, timestamp_ms)
                    self._sequence += 1
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            # Release on the capture thread; never race VideoCapture.read().
            self._capture.release()

    def read(self):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._sequence > self._consumed or self._error is not None or self._stopped.is_set(),
                timeout=self._options["timeout_seconds"],
            )
            if self._error is not None:
                raise RuntimeError(str(self._error)) from self._error
            if self._stopped.is_set():
                raise RuntimeError("Camera is closed")
            if not ready:
                raise TimeoutError("Timed out waiting for a fresh USB camera frame")
            self._consumed = self._sequence
            return self._latest

    def close(self):
        self._stopped.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            LOG.warning("Camera driver is blocked in read(); it will be released when the process exits")
