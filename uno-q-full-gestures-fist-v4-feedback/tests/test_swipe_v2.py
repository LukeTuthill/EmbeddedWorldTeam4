"""Trajectory checks for the isolated board app; no webcam needed."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from gesture_app.backends import integrated_results
from gesture_app.config import load_config
from gesture_app.output import Outputs
from gesture_app.recognition import Detection, StableGesture
from gesture_app.swipe import SwipeDetector, SwipeEvent


def detected(gesture, x, y=.5, confidence=.55, width=.12, height=.18):
    return Detection(gesture, confidence,
                     bounding_box=(x - width / 2, y - height / 2,
                                   x + width / 2, y + height / 2))


class SwipeV2Tests(unittest.TestCase):
    def replay(self, frames):
        detector = SwipeDetector()
        events = []
        for timestamp_ms, detections in frames:
            event = detector.update(detections, timestamp_ms, now_ms=timestamp_ms)
            if event is not None:
                events.append(event)
        return events, detector

    def test_gesture_in_any_image_direction(self):
        for axis, positions in (("right", [.20, .28, .36, .44]),
                                ("left", [.80, .72, .64, .56]),
                                ("down", [.20, .28, .36, .44])):
            frames = []
            for index, p in enumerate(positions):
                x, y = (.50, p) if axis == "down" else (p, .50)
                frames.append((index * 120, [detected("Victory", x, y)]))
            events, _ = self.replay(frames)
            self.assertEqual(len(events), 1, axis)
            self.assertEqual(events[0].to_dict()["action"], "skip")

    def test_brief_model_dropout_during_v_sign_motion(self):
        frames = [(0, [detected("Victory", .20)]),
                  (100, [detected("Victory", .27)]),
                  (200, []),
                  (300, [detected("Victory", .36)]),
                  (400, [detected("Victory", .44)])]
        events, _ = self.replay(frames)
        self.assertEqual(len(events), 1)

    def test_board_observed_confidence_below_old_threshold(self):
        frames = [(i * 120, [detected("Victory", x, confidence=.52)])
                  for i, x in enumerate([.20, .28, .36, .44])]
        self.assertEqual(len(self.replay(frames)[0]), 1)

    def test_stationary_hold_then_short_swipe(self):
        frames = [(i * 100, [detected("Victory", x)])
                  for i, x in enumerate([.20] * 7 + [.27, .34, .42])]
        self.assertEqual(len(self.replay(frames)[0]), 1)

    def test_stationary_lighting_dropout_and_jitter(self):
        frames = [(0, [detected("Victory", .50)]),
                  (100, [detected("Victory", .52)]),
                  (200, []),
                  (300, [detected("Victory", .49)]),
                  (400, [detected("Victory", .51)]),
                  (500, [detected("Victory", .50)])]
        self.assertEqual(self.replay(frames)[0], [])

    def test_box_jump_or_size_change_resets_track(self):
        for jump in ([detected("Victory", .70)],
                     [detected("Victory", .30, width=.32)]):
            frames = [(0, [detected("Victory", .20)]),
                      (100, [detected("Victory", .27)]),
                      (200, jump),
                      (300, [detected("Victory", .70 if jump[0].bounding_box[0] > .5 else .30)]),
                      (400, [detected("Victory", .70 if jump[0].bounding_box[0] > .5 else .30)])]
            self.assertEqual(self.replay(frames)[0], [])

    def test_zigzag_and_slow_repositioning_are_not_swipes(self):
        zigzag = [(i * 100, [detected("Victory", x)])
                  for i, x in enumerate([.50, .57, .50, .57, .50, .57])]
        slow = [(i * 350, [detected("Victory", x)])
                for i, x in enumerate([.20, .26, .32, .38])]
        self.assertEqual(self.replay(zigzag)[0], [])
        self.assertEqual(self.replay(slow)[0], [])

    def test_confident_stop_or_second_hand_breaks_motion(self):
        stop = [(0, [detected("Victory", .20)]),
                (100, [detected("Victory", .28)]),
                (200, [detected("Open_Palm", .34, confidence=.8)]),
                (300, [detected("Victory", .36)]),
                (400, [detected("Victory", .44)])]
        second_hand = [(0, [detected("Victory", .20)]),
                       (100, [detected("Victory", .28)]),
                       (200, [detected("Victory", .36), detected("Victory", .78)]),
                       (300, [detected("Victory", .44)])]
        self.assertEqual(self.replay(stop)[0], [])
        self.assertEqual(self.replay(second_hand)[0], [])

    def test_stale_frame_and_duplicate_timestamp_do_not_add_motion(self):
        frames = [(0, [detected("Victory", .20)]),
                  (100, [detected("Victory", .28)]),
                  (100, [detected("Victory", .44)])]
        detector = SwipeDetector()
        events = []
        for timestamp_ms, detections in frames:
            result = detector.update(detections, timestamp_ms, now_ms=timestamp_ms)
            if result:
                events.append(result)
        self.assertEqual(events, [])
        detector = SwipeDetector()
        self.assertIsNone(detector.update([detected("Victory", .20)], 0, now_ms=800))

    def test_one_event_until_v_sign_released(self):
        outbound = [(i * 100, [detected("Victory", x)])
                    for i, x in enumerate([.20, .28, .36, .44])]
        return_path = [(400 + i * 100, [detected("Victory", x)])
                       for i, x in enumerate([.44, .36, .28, .20])]
        self.assertEqual(len(self.replay(outbound + return_path)[0]), 1)

    def test_lowered_fingers_rearm_after_release(self):
        outbound = [(i * 100, [detected("Victory", x)])
                    for i, x in enumerate([.20, .28, .36, .44])]
        lowered = [(400, [detected("Open_Palm", .44, confidence=.8)]),
                   (700, [detected("Open_Palm", .44, confidence=.8)])]
        second = [(800 + i * 100, [detected("Victory", x)])
                  for i, x in enumerate([.20, .28, .36, .44])]
        self.assertEqual(len(self.replay(outbound + lowered + second)[0]), 2)

    def test_confident_unknown_pose_breaks_motion(self):
        frames = [(0, [detected("Victory", .20)]),
                  (100, [detected("Victory", .28)]),
                  (200, [detected("Closed_Fist", .34, confidence=.8)]),
                  (300, [detected("Victory", .36)]),
                  (400, [detected("Victory", .44)])]
        self.assertEqual(self.replay(frames)[0], [])

    def test_existing_static_acceptance_still_works(self):
        static = StableGesture(.5, 2, .85)
        self.assertIsNone(static.update([detected("Open_Palm", .5, confidence=.6)]))
        self.assertEqual(static.update([detected("Open_Palm", .5, confidence=.6)]).gesture,
                         "Open_Palm")
        self.assertEqual(static.update([detected("Thumb_Up", .5, confidence=.9)]).gesture,
                         "Thumb_Up")

    def test_integrated_bbox_is_normalized_for_swipe(self):
        detections = integrated_results({"detection": [{"class_name": "peace",
            "confidence": "52.0", "bounding_box_xyxy": [64, 96, 128, 160]}]},
            image_size=(320, 320))
        self.assertEqual(detections[0].gesture, "Victory")
        self.assertEqual(detections[0].bounding_box, (.2, .3, .4, .5))

    def test_sound_mode_requests_buzzer(self):
        config = load_config(Path(__file__).parents[1] / "python/config.json")
        self.assertTrue(config.data["buzzer"]["enabled"])
        self.assertTrue(config.data["diagnostics"]["startup_feedback_test"])

        class RecordingBridge:
            def __init__(self):
                self.calls = []

            def call(self, name, *args, **kwargs):
                self.calls.append((name, args))
                return 3

        bridge = RecordingBridge()
        outputs = Outputs(config, bridge=bridge, emit=lambda *args, **kwargs: None)
        outputs.swipe(SwipeEvent(300, .2, .52), now=1.0)
        outputs.update(detected("Open_Palm", .5, confidence=.8), now=2.0)
        self.assertIn(("show_skip", (True, True)), bridge.calls)
        self.assertIn(("set_gesture_feedback", (1, True, True)), bridge.calls)


if __name__ == "__main__":
    unittest.main()
