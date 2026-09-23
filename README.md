# UNO Q gesture feedback app

# Starlight: On-Device Gesture Recognition on the Arduino UNO Q
**Embedded World North America 2026 Edge AI Hackathon, Team 4 (Starlight)**
Anaheim, CA - September 22–23, 2026
Hackaday.io project: https://hackaday.io/project/206735-ewna-hackathon-team-4

## The challenge
A warehouse worker whose hands are full needs to pause, confirm, or skip a step on a nearby
screen without touching it. Our system watches the scene with a webcam, recognizes three
gestures, and triggers a different action for each within a few hundred milliseconds. All
inference runs on the board, with no network connection.

## Team
- Luke Tuthill (Purdue)
- Jingbin Lin (UCSD)
- Yufan Wang (UCSD)
- Eduardo Hideki Sakamoto (CSUN)

Use [`uno-q-gesture-integrated/`](uno-q-gesture-integrated/) as the Arduino App Lab app. It runs the bundled `hand-gestures` model on the UNO Q, detects open palm and thumbs-up, derives horizontal swipe from multiple consecutive palm detections, then drives onboard RGB indicators, Modulino Pixels, and Modulino Buzzer.

1. Connect UNO Q to App Lab on your PC.
2. Import/open `uno-q-gesture-integrated/` as the App. Its `app.yaml`, Python entry point, and matching deployable MCU sketch are all in that directory.
3. Plug a USB webcam into UNO Q through a powered USB-C hub; camera must connect to board, not PC. Check `/dev/video*` on board; adjust `python/config.json` camera path if needed.
4. Connect Modulino Pixels and Modulino Buzzer to UNO Q Qwiic. In App Lab, press **Run** to start inference and flash sketch.

Gesture feedback: open palm = red/two beeps; thumbs-up = green/one beep; horizontal swipe = blue/three beeps. Swipe is a multi-frame heuristic, not a trained class. Run on-device with several lighting conditions before relying on it.

Standalone checks from repository root:

```sh
python3 uno-q-gesture-integrated/python/main.py --check-config
python3 -m unittest discover -s tests -v
```
