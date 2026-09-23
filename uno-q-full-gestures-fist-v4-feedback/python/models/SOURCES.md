# Integrated model source

Full Gestures uses only Arduino's `hand-gestures` model through the App Lab
`arduino:object_detection` brick. The app reuses the `five`, `good`, `neut`, and
`peace` labels and the detected positions; it does not run a second model.

Arduino's [Object Detection implementation](https://github.com/arduino/app-bricks-py/blob/main/src/arduino/app_bricks/object_detection/__init__.py)
returns `class_name`, percentage `confidence`, and `bounding_box_xyxy`. The adapter
normalizes those boxes using the model input dimensions. Full Gestures uses the
`peace` (V-sign) detections for direction-independent motion-to-Skip decisions.

The optional standalone IDE package embeds `hand-gestures.eim` from
[Arduino's repository at revision 75e2cd6a184175f3e9a6fa2ae9bf7d6059d669c4](https://github.com/arduino/app-bricks-py/blob/75e2cd6a184175f3e9a6fa2ae9bf7d6059d669c4/containers/ai/ei-models-runner/models/ei-ootb-models/arm64/hand-gestures.eim).
SHA-256: `89b9db101535b8c638d6cc1c281707d7d40c102b51fc482b82b7fef45025442c`.
Arduino's catalog associates it with [Edge Impulse project 842271](https://studio.edgeimpulse.com/public/842271/live).
Consult the publisher for model terms and provenance. The project does not train
or modify these model bytes.

The native MediaPipe runtime, its `.task` file, OpenCV Zoo ONNX hand models, and
their vendored adapters have been removed from this project. They are not part of
the Full Gestures package.
