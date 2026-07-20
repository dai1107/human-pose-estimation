# Changelog

All notable changes are recorded here. Versions follow Semantic Versioning;
development builds use the `X.Y.Z.devN` form.

## [Unreleased]

- No unreleased changes recorded yet.

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
