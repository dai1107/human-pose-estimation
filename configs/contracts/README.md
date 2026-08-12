# Round 10–12 product contracts

These strict YAML files version the additive product surface introduced in Round 10 and retained through the realtime optimization work:

- `action_gating_v1.yaml`: 8 actions plus idle, transition and unknown/OOD, with entry/exit hysteresis and cooldown.
- `scoring_correction_v1.yaml`: VALID/NO_REP/UNSURE, unobservable scoring, traceable corrections and suppression.
- `coordinate_spaces_v1.yaml`: image 2D, relative monocular 3D, estimated camera rays and calibrated metric depth without mixing thresholds.
- `oni_research_v1.yaml`: offline Depth/IR subject proposals and modality evidence gates; RGB–Depth registration, phone–ONI pairing and phone frame labels are prohibited.
- `realtime_latency_v1.yaml`: latest-frame admission, stale suppression and display-only prediction. Runtime defaults additionally live in `configs/product_pose.yaml`, including queue size 1, target/max pose FPS 15/20, warning/max pose age 80/120 ms and inference-only adaptive resolution.

`python -m src.doctor` validates all five files. `python -m tools.run_round10_shadow` rebuilds the current readiness, A-F ablation and failure-pool reports. Automatic action gating remains disabled by default.

The desktop `RealtimeBudgetController` consumes rolling P95 inference time, pose-result age and queue saturation. Its fixed degradation order is pose FPS, inference-only resolution, then optional analysis frequency. It does not modify capture, playback or render clocks. The browser uses its own one-pending-frame Worker and Full/Lite benchmark while following the same stale-result and display-only-prediction boundaries.

The Logistic Regression shadow sidecar is available only after a reviewed model artifact exists:

```powershell
python tools/replay_hyrox_video.py `
  --video path\to\video.mp4 `
  --hyrox-action lunge `
  --auto-action-shadow `
  --auto-action-model datasets\hyrox\models\round10_action_gate_logreg_v1.json `
  --save-shadow-json outputs\round10_shadow.json
```

The manually selected analyzer is still authoritative in this mode. A shadow prediction never switches or resets it. The formal `--hyrox-action auto` entry is intentionally unavailable until continuous-switch, unknown rejection, latency and human-ground-truth gates pass.
