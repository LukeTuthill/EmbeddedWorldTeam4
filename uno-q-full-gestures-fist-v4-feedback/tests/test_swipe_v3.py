"""Timestamp-pair swipe tests; no webcam or board required."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from gesture_app.config import load_config
from gesture_app.recognition import Detection
from gesture_app.swipe_v3 import SwipeDetector


def hand(x, y=0.5, confidence=0.35, width=0.14, height=0.18, pose="Victory"):
    return Detection(pose, confidence, bounding_box=(
        x - width / 2, y - height / 2, x + width / 2, y + height / 2))


class SwipeV3Tests(unittest.TestCase):
    def detector(self):
        return SwipeDetector(load_config(Path(__file__).parents[1] / "python/config.json").data["swipe"])

    @staticmethod
    def feed(detector, timestamps_and_detections):
        events = []
        for timestamp, detections in timestamps_and_detections:
            event = detector.update(detections, timestamp, now_ms=timestamp)
            if event is not None:
                events.append(event)
        return events

    def test_left_and_right_from_two_frames_300ms_apart(self):
        for start, end, direction in ((.20, .43, "right"), (.78, .54, "left")):
            with self.subTest(direction=direction):
                events = self.feed(self.detector(), [
                    (1000, [hand(start, confidence=.31)]),
                    (1300, [hand(end, confidence=.33)]),
                ])
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0].direction, direction)
                self.assertEqual(events[0].duration_ms, 300)
                self.assertEqual(events[0].to_dict()["gesture"], "Swipe")

    def test_timestamps_not_frame_counts_and_dropouts_allowed(self):
        frames = [(1000, [hand(.20)])]
        frames += [(timestamp, []) for timestamp in range(1033, 1300, 33)]
        frames.append((1300, [hand(.42)]))
        self.assertEqual(len(self.feed(self.detector(), frames)), 1)

    def test_pair_must_be_near_300ms_and_move_twenty_percent(self):
        for end_time, end_x in ((1150, .45), (1600, .45), (1300, .39)):
            with self.subTest(end_time=end_time, end_x=end_x):
                self.assertEqual(self.feed(self.detector(), [
                    (1000, [hand(.20)]), (end_time, [hand(end_x)])
                ]), [])

    def test_check_other_timed_pairs_when_closest_pair_has_short_travel(self):
        frames = [(1000, [hand(.20)]), (1130, [hand(.30)]),
                  (1390, [hand(.45)])]
        events = self.feed(self.detector(), frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].duration_ms, 390)

    def test_vertical_motion_and_drift_do_not_count(self):
        for end_x, end_y in ((.21, .25), (.44, .61)):
            self.assertEqual(self.feed(self.detector(), [
                (1000, [hand(.20, .50)]), (1300, [hand(end_x, end_y)])
            ]), [])

    def test_large_box_size_change_rejected(self):
        self.assertEqual(self.feed(self.detector(), [
            (1000, [hand(.20, width=.12)]),
            (1300, [hand(.45, width=.25)]),
        ]), [])

    def test_multiple_plausible_hands_reject_either_endpoint(self):
        for first, second in (
            ([hand(.20), hand(.75)], [hand(.45)]),
            ([hand(.20)], [hand(.45), hand(.80)]),
        ):
            self.assertEqual(self.feed(self.detector(), [
                (1000, first), (1300, second)
            ]), [])

    def test_other_pose_and_low_confidence_do_not_establish_pair(self):
        self.assertEqual(self.feed(self.detector(), [
            (1000, [hand(.20, pose="Open_Palm")]),
            (1300, [hand(.45)]),
        ]), [])
        self.assertEqual(self.feed(self.detector(), [
            (1000, [hand(.20, confidence=.24)]),
            (1300, [hand(.45)]),
        ]), [])

    def test_confident_other_pose_between_endpoints_breaks_pair(self):
        self.assertEqual(self.feed(self.detector(), [
            (1000, [hand(.20)]),
            (1150, [hand(.32, pose="Thumb_Up")]),
            (1300, [hand(.45)]),
        ]), [])

    def test_stationary_jitter_and_brightness_dropout(self):
        frames = [(1000, [hand(.50)]), (1100, [hand(.52)]),
                  (1200, []), (1300, [hand(.49)]),
                  (1400, [hand(.51)])]
        self.assertEqual(self.feed(self.detector(), frames), [])

    def test_stale_or_duplicate_frames_ignored(self):
        detector = self.detector()
        self.assertIsNone(detector.update([hand(.20)], 1000, now_ms=1000))
        self.assertIsNone(detector.update([hand(.45)], 1000, now_ms=1000))
        self.assertIsNone(detector.update([hand(.45)], 1300, now_ms=2200))

    def test_cooldown_blocks_repeat(self):
        frames = [(1000, [hand(.20)]), (1300, [hand(.45)]),
                  (1500, [hand(.20)]), (1800, [hand(.45)]),
                  (2400, [hand(.20)]), (2700, [hand(.45)])]
        events = self.feed(self.detector(), frames)
        self.assertEqual(len(events), 2)


if __name__ == "__main__":
    unittest.main()
