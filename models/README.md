# Models

Put MediaPipe task model files here before running the app.

Pose Landmarker recommended file name:

```text
models/pose_landmarker_full.task
```

Pose Landmarker official model download URL:

```text
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

Hand Landmarker is optional. It is required only when running realtime detection with `--show-hands` to supplement the pose landmarks with five-finger overlays. The app displays the 20 finger points and excludes the wrist point.

Hand Landmarker recommended file name:

```text
models/hand_landmarker.task
```

Hand Landmarker official model download URL:

```text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
