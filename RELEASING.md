# Release and upgrade guide

## Version rules

- `src/version.py` is the single version source.
- Use Semantic Versioning: patch for compatible fixes, minor for compatible
  features, and major for incompatible CLI or output-schema changes.
- Development builds use `X.Y.Z.devN`. A release removes the `.devN` suffix.
- Every release updates `CHANGELOG.md`; output-schema changes also update the
  schema documentation and compatibility tests.

## Build and verify

From a clean checkout with Python 3.10–3.12:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m src.smoke_test
python -m build
python -m pip install --force-reinstall --no-deps dist\pose_estimation_hyrox-*.whl
pose-doctor --json
```

The build must produce both a wheel and source archive in `dist/`. CI performs
the import, compile, text-format, unit-test, no-camera smoke, and package-build
checks on Windows and Linux.

## Upgrade behavior

- Back up or export important `outputs/references` before upgrading.
- JSON/CSV outputs include `schema_version` and `program_version`. Legacy
  outputs without `schema_version` are read as version `0`; files newer than
  the installed reader are rejected with `SCH001` instead of being guessed.
- Run `pose-doctor --json` after upgrading.
- Preview retention cleanup with `pose-clean --json`; deletion occurs only
  when `--apply` is explicitly supplied.
- Models and default configuration are bundled in release artifacts. Explicit
  `--model` and `--hyrox-config` paths continue to override bundled assets.
- Web clients should use `yolo-mediapipe` for the explicit hybrid pipeline.
  The legacy `yolo-pose` request remains accepted for compatibility, but it is
  no longer presented as a manual web option. Explicit `mediapipe` is always
  the pure MediaPipe pose pipeline.
- Sled Pull cycle boundaries now require the return to `reach`. Comparisons
  across versions should account for this counting-boundary change.
- `POSE_OUTPUT_DIR` controls web artifacts plus doctor/cleanup defaults.
  Desktop and reference CLI output paths remain controlled by their explicit
  `--save-dir`, `--log-dir`, `--output-dir`, or `--root` options.
