# Changelog

All notable changes are recorded here. Versions follow Semantic Versioning;
development builds use the `X.Y.Z.devN` form.

## [Unreleased]

### Added

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

- Realtime web results are now pushed as soon as inference completes. On the
  current 30-frame local protocol probe at a 640 px long edge and JPEG quality
  0.65, round-trip latency improved from about 58 ms to 18.7 ms p50 and 35.5 ms
  p95; server processing was 17.2 ms p50, pose inference 14.4 ms p50, and pose
  age 16.0 ms p50. These are machine-specific baselines, not hardware-neutral
  guarantees.
- Final API verification of the cached 133-frame lunge sample completed in
  about 1.1 seconds with 0.0 ms runtime model inference, compared with about
  15.7 seconds on the former inference path. The rowing sample benchmark
  improved from about 14.3 seconds to 6.1 seconds; remaining time is video
  decode, rule evaluation, drawing, and JPEG encoding.

### Validation

- The 3D Assist completion baseline passed 479 tests. The final focused suite
  covering sample caches, web APIs, backend safety, product configuration, and
  realtime protocol behavior passed 126 tests; online lunge verification also
  confirmed cached hand landmarks without starting the hand model.

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
