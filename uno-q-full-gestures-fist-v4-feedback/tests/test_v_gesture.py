"""Built-in Victory pose triggers Skip without motion or bounding box."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from gesture_app.config import load_config
from gesture_app.output import Outputs
from gesture_app.recognition import Detection
from gesture_app.v_gesture import VGestureEvent, VGestureTrigger


CONFIG = Path(__file__).parents[1] / "python/config.json"


def victory(confidence=.7):
    return Detection("Victory", confidence)


class VGestureTests(unittest.TestCase):
    def trigger(self):
        return VGestureTrigger(load_config(CONFIG).data["v_gesture"])

    def test_stationary_v_sign_triggers_once_without_box(self):
        trigger = self.trigger()
        self.assertIsNone(trigger.update([victory()], 1000, now_ms=1000))
        event = trigger.update([victory(.65)], 1150, now_ms=1150)
        self.assertEqual(event.to_dict()["gesture"], "Victory")
        self.assertEqual(event.to_dict()["action"], "skip")
        self.assertEqual(event.confidence, .65)
        for timestamp in (1300, 1450, 1600):
            self.assertIsNone(trigger.update([victory()], timestamp, now_ms=timestamp))

    def test_low_confidence_or_other_pose_does_not_trigger(self):
        trigger = self.trigger()
        for timestamp, detections in ((1000, [victory(.49)]),
                                      (1150, [Detection("Open_Palm", .9)]),
                                      (1300, [victory(.8)])):
            self.assertIsNone(trigger.update(detections, timestamp, now_ms=timestamp))

    def test_rearms_after_absence(self):
        trigger = self.trigger()
        times = [(1000, [victory()]), (1150, [victory()]),
                 (1300, []), (1750, []),
                 (1900, [victory()]), (2050, [victory()])]
        events = [trigger.update(detections, timestamp, now_ms=timestamp)
                  for timestamp, detections in times]
        self.assertEqual(sum(event is not None for event in events), 2)

    def test_stale_or_duplicate_frame_never_counts(self):
        trigger = self.trigger()
        self.assertIsNone(trigger.update([victory()], 1000, now_ms=1000))
        self.assertIsNone(trigger.update([victory()], 1000, now_ms=1000))
        self.assertIsNone(trigger.update([victory()], 1150, now_ms=2000))
        self.assertIsNone(trigger.update([victory()], 1300, now_ms=1300))
        self.assertIsNotNone(trigger.update([victory()], 1450, now_ms=1450))

    def test_output_requests_blue_skip_feedback(self):
        class Bridge:
            def __init__(self):
                self.calls = []

            def call(self, name, *args, **kwargs):
                self.calls.append((name, args))
                return 3

        bridge = Bridge()
        emitted = []
        outputs = Outputs(load_config(CONFIG), bridge=bridge,
                          emit=lambda payload, **_kwargs: emitted.append(payload))
        outputs.victory(VGestureEvent(.7))
        self.assertIn(("show_skip", (True, True)), bridge.calls)
        self.assertIn('"gesture": "Victory"', emitted[0])


if __name__ == "__main__":
    unittest.main()
