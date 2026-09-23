# UNO Q V4 static fist with feedback

Recognition uses V4 Python unchanged: built-in `hand-gestures` model maps `neut` to `Closed_Fist`; two fresh fist detections at confidence at least 0.50 and within 0.40 seconds trigger Skip. No hand movement or second-frame pixel matching required. Open Palm and Thumb Up recognition stays unchanged.

Feedback uses V6's merged MCU sketch: red pulse for Open Palm, green fade for Thumb Up, blue trail for Skip, gesture sounds when enabled, and an 8×13 LED matrix counter. Existing Bridge calls (`set_gesture_feedback`, `show_skip`, `gesture_feedback_version`) remain compatible. Counter increments when a gesture starts and caps at 999; resets on MCU reboot. Supplied animation source remains in `sketch/sketch_1.reference.txt`, not as a second compiled `.ino`.

App path on UNO Q: `/home/arduino/ArduinoApps/uno-q-full-gestures-fist-v4-feedback`. Original V4 and V6 apps remain intact. Focused V4 test: `.venv/bin/python -m unittest discover -s uno-q-full-gestures-fist-v4-feedback/tests -p test_fist_gesture.py -v`; configuration test: `.venv/bin/python uno-q-full-gestures-fist-v4-feedback/python/main.py --check-config`. All five active fist tests pass. Five legacy V3 swipe tests in the full suite fail identically in original V4; that trigger is disabled here.

Deployed 2026-09-23: V6 stopped; V4-feedback flashed and started. Live logs show camera 640×480 at 30 frames per second, healthy inference service, Modulino Pixels/Buzzer found, and `fist_gesture` Skip events. Physical pixel, buzzer, and matrix-counter effects still require visual/audible confirmation.

Physical checks: hold fist steady for two inference results; expect `fist_gesture` Skip, blue trail, buzzer, and matrix count increase. Continue holding: no repeat. Remove fist for at least 0.40 seconds, then retry. Test Open Palm and Thumb Up, sudden doorway lighting change, and hand absence for false triggers. Runtime logs cannot verify actual light or sound output.
