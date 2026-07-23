# Changelog

All notable changes are recorded here. Versions follow Semantic Versioning;
development builds use the `X.Y.Z.devN` form.

## [Unreleased]

### Added

- Added browser camera track/settings diagnostics for actual presented FPS,
  frame-interval P50/P95 and instability, low light, and duplicate frames.
- Added `pose-camera-benchmark` for explicit on-device default/DSHOW/MSMF,
  MJPG/YUY2 benchmarking and an exact configuration backend cache. Physical
  sensor-to-photon values remain null unless externally measured.
- Added an isolated browser `DisplayPosePredictor` for 0–45 ms
  expected-display-time compensation with smoothed velocity, confidence and
  stale-gap gates, body-scale displacement limits, reversal damping, and
  support-foot constraints. Predicted landmarks remain display-only.
- Added fingerprint-validated pose and hand-keypoint caches for all eight fixed
  web samples. Demo playback now performs zero model inference while offline
  golden validation continues to execute the real backends.

- Added MediaPipe world-landmark 3D shadow kinematics, stateful reliability
  gates, per-angle 2D/3D comparison, and grouped availability/failure reports
  without changing HYROX decision inputs.
- Added confidence-only 3D Assist Mode for mapped knee, hip, elbow, and shoulder
  rules. Reliable agreement can raise rule confidence, severe 2D/3D conflict
  downgrades the candidate to `UNSURE`, and unavailable 3D falls back to the
  unchanged 2D decision path.
- Added a strict MediaPipe-only product backend configuration, shared backend
  support tiers, p50/p95 latency metrics, a synchronous baseline tool, and a
  round-one architecture audit report.
- Added fixed-interval golden regression for all eight bundled HYROX videos and
  versioned reports through the `pose-golden` CLI.
- Added short-smoke and formal 30/60 minute endurance validation with FPS, P95
  latency, process-memory growth, read-failure, and output-integrity metrics
  through the `pose-endurance` CLI.

### Changed

- Closed the local-first architecture boundary: browser workers send only raw
  landmarks to Python HYROX analysis, constant-velocity predictions are Canvas
  only, server fallback is configuration-controlled, and report protocol
  whitelists reject prediction fields. Neural/temporal models and training
  flows remain unimplemented.
- Desktop camera startup now uses the device cache in `auto` mode and otherwise
  tries the OpenCV default backend before safe platform fallbacks; DSHOW is no
  longer permanently hard-coded.
- Web angle overlays now show quality-gated MediaPipe world-landmark 3D joint
  angles and label them `3D`; unavailable or unreliable 3D measurements are
  omitted instead of being silently replaced by 2D display values. Bundled
  sample caches were upgraded to v2 to include world landmarks.
- File-backed web playback is now paced at the video's encoded frame rate.
  Cached samples still analyze every frame, but no longer play faster than the
  source video when cached pose lookup finishes ahead of realtime.
- Replaced WebSocket result polling with an event-driven sender thread and made
  optional finger tracking off by default to reduce realtime camera latency.

- Promoted the formal product configuration from 3D shadow collection to
  `assist`; selected angles, phase thresholds, contact, floor, takeoff,
  landing, wrist-timing, step, and distance rules remain 2D.
- Reworked realtime One Euro smoothing to use observation capture timestamps,
  selectable stable/balanced/responsive profiles, body-region response scales,
  independent image/world landmark state, and a configurable 250 ms gap reset.
- Product `auto` now resolves only to MediaPipe; the web UI exposes only
  MediaPipe Pose, while YOLO/RTMW remain available solely through explicit
  experimental and offline-comparison paths.
- Split the desktop entry point into runtime, CLI, capture, backend, display,
  recording, session, and HYROX-analysis components. `main.py` is now a thin
  stable launcher.
- Retired the independent `src.realtime_pose` execution loop; its compatibility
  facade now translates legacy arguments and forwards to the maintained desktop
  runtime.

### Performance

- Reduced browser main-thread rendering work with fixed pose-coordinate
  buffers, cached connections/fonts/video transforms, 12 FPS angle labels,
  5 FPS metrics, 3 FPS statistics, and content-sensitive feedback DOM
  updates. Latency audits now include render-loop, Canvas, and DOM P95 plus
  Long Task phase attribution.
- Realtime web results are now pushed as soon as inference completes. On the
  current 30-frame local protocol probe at a 640 px long edge and JPEG quality
  0.65, round-trip latency improved from about 58 ms to 18.7 ms p50 and 35.5 ms
  p95; server processing was 17.2 ms p50, pose inference 14.4 ms p50, and pose
  age 16.0 ms p50. These are machine-specific baselines, not hardware-neutral
  guarantees.
- Without presentation pacing, cached processing of the 133-frame lunge sample
  required about 1.1 seconds with 0.0 ms runtime model inference, compared with
  about 15.7 seconds on the former inference path. Web presentation is now
  intentionally capped at the source frame rate, so this 30 FPS sample plays
  for its original duration of about 4.43 seconds.

### Validation

- The current full suite passes 530 Python tests and 16 Node tests. Full-model
  golden replay passes all 8/8 HYROX videos; Doctor, no-camera smoke,
  compileall, text-format, diff, and package-build checks also pass. Real
  camera backend and physical sensor-to-photon results remain device-site
  acceptance work and are not synthesized by automated tests.

## [0.1.0.dev0] - 2026-07-20

### Added

- Added strict schema validation for HYROX action, shared contact/foot-event,
  observability, and personal-reference YAML; `pose-doctor` now validates all
  configuration groups and output writability.
- Added rolling logs, classified `CFG/SRC/BCK/OUT/RUN/REC` errors, stable exit
  codes, optional debug tracebacks, safe resource cleanup, and partial-session
  recovery metadata.
- Added versioned JSON/CSV artifacts, legacy-schema compatibility, future
  schema rejection, configurable web output roots, and preview-first
  retention/quota cleanup through `pose-clean`.
- Added reproducible core/YOLO/RTMW/development dependency groups, ten
  installable CLI entry points, wheel/sdist packaging, release guidance,
  no-camera smoke tests, and Windows/Linux CI on Python 3.10 and 3.12.

### Changed

- Split the web model choices into explicit `纯 MediaPipe` and
  `YOLO + MediaPipe` pipelines. Explicit MediaPipe never loads YOLO; explicit
  YOLO + MediaPipe uses identity-checked dual-model fusion for every action.
- Kept automatic Lunge analysis on the YOLO + MediaPipe identity-lock path
  while retaining pure YOLO Pose for internal automatic/desktop compatibility.
- Changed Sled Pull analysis-cycle completion from
  `reach → pull → recover` to `reach → pull → recover → reach`, so the
  forward return belongs to the same cycle.
- Versioned session, web, replay, metrics, doctor, multicamera, reference, and
  comparison outputs with common program and artifact identity fields.
- Updated README, desktop/web/model/configuration guides, maturity status, and
  release documentation to match the current behavior and 400-test baseline.

### Included

- Eight HYROX analyzers, web realtime analysis, MediaPipe, optional YOLO/RTMW
  backends, personal-reference DTW comparison, local/public web launchers,
  session reports, and multi-camera synchronization checks.
