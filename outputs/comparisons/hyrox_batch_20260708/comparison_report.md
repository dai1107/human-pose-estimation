# HYROX ??????

- ??????8
- ?????24
- ?????`outputs\comparisons\hyrox_batch_20260708`
- ?? CSV?`outputs\comparisons\hyrox_batch_20260708\summary.csv`
- ?????`outputs\comparisons\hyrox_batch_20260708\scheme_averages.csv`
- ??????`outputs\comparisons\hyrox_batch_20260708\best_by_video.csv`

## ??????

| scheme | success | missing | keypoint_jitter | angle_jitter | avg_infer_ms | p95_infer_ms | e2e_ms | roi_success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mediapipe | 0.999 | 0.016 | 0.00836 | 2.423 | 15.6 | 21.2 | 16.6 | 0.000 |
| yolo_pose | 0.997 | 0.015 | 0.01001 | 2.261 | 30.8 | 22.5 | 31.9 | 0.000 |
| yolo_roi_mediapipe | 0.997 | 0.022 | 0.01099 | 3.210 | 18.9 | 27.5 | 45.2 | 0.926 |

## ??????

| video | stability_best | speed_best | recommended | success | missing | keypoint_jitter | angle_jitter |
|---|---|---|---|---:|---:|---:|---:|
| 农夫行走.mp4 | mediapipe | mediapipe | mediapipe | 1.000 | 0.000 | 0.00289 | 1.274 |
| 划船机.mp4 | yolo_pose | yolo_pose | yolo_pose | 1.000 | 0.000 | 0.00943 | 2.088 |
| 投掷药球.mp4 | mediapipe | mediapipe | mediapipe | 1.000 | 0.000 | 0.00690 | 3.238 |
| 拉雪橇.mp4 | yolo_pose | mediapipe | yolo_pose | 1.000 | 0.000 | 0.00901 | 2.147 |
| 推雪橇.mp4 | mediapipe | mediapipe | mediapipe | 1.000 | 0.000 | 0.00639 | 1.480 |
| 波比跳远.mp4 | mediapipe | mediapipe | mediapipe | 0.992 | 0.082 | 0.00850 | 3.105 |
| 滑雪机.mp4 | yolo_pose | mediapipe | yolo_pose | 1.000 | 0.000 | 0.01079 | 3.192 |
| 负重箭步蹲.mp4 | mediapipe | mediapipe | mediapipe | 1.000 | 0.000 | 0.01079 | 3.085 |

## ??

- ??????????????mediapipe=5, yolo_pose=3
- ????????`mediapipe`?
- ???????`mediapipe`?
- ???YOLO Pose ? COCO 17 ??MediaPipe / YOLO ROI + MediaPipe ? 33 ????????????????????????????????????????
