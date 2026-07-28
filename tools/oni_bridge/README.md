# Offline ONI inspection bridge

This directory is deliberately separate from the MediaPipe product runtime.
It opens recorded ONI files only and ships only the OpenNI2 `OniFile` driver.

```powershell
tools\oni_bridge\oni-inspect.exe input.oni --output oni_inventory.json
python tools\audit_oni_dataset.py

tools\oni_bridge\oni-export.exe input.oni --output extracted\record_id
python tools\export_oni_dataset.py
python tools\synchronize_oni_dataset.py
```

The JSON report records file-open and complete-playback status, stream modes,
frame counts, indices, timestamps, interval P50/P95, anomalies, depth validity,
center-region depth distribution, and the A–E classification defined in
`完整实施清单.md`.

Batch output is written below `datasets/hyrox/reports/oni_audit/`: one JSON
per manifest record plus `batch_report.json`, `invalid_records.json`, and
`audit_summary.md`. Use `--reuse-existing` to regenerate aggregates without
replaying ONI files.

The bundled OpenNI file driver is not Unicode-path safe. For a non-ASCII input
name, the bridge creates a process-scoped ASCII hard link in the Windows temp
directory, opens that link, and removes only that link with POSIX delete
semantics. The original ONI remains read-only and unchanged.

The complete validation scan opened and replayed all 32 files. Every
record contains Depth + IR and no Color (`A=0, B=32, C=0, D=0, E=0`), with no
decode or timeline anomalies. This is a stream-integrity result only; it does
not confirm recording-intent labels or target-athlete identity.

`oni-export.exe` writes source-indexed lossless frames. Depth and GRAY16 IR
are little-endian uint16 NPY files; RGB888 Color is exported as lossless PPM.
The Python batch wrapper validates every index against the audit, fingerprints
all frame contents, and creates derivative MP4 previews. The current full run
exported all 32 records with 18,709 Depth and 18,713 IR frames. No Color
stream exists, so no current `color.mp4` is expected.

`synchronize_oni_dataset.py` performs only ONI-internal Color/Depth pairing.
It uses frame indices only after explicit capture-sync confirmation; otherwise
it uses nearest timestamps and quality gates. Current records have no Color,
so all 32 reports contain zero pairs and are `video_level_only`. IR and future
phone data are never used to fabricate RGB/Depth pairs.

Rebuild with:

```powershell
powershell -ExecutionPolicy Bypass -File tools\oni_bridge\build_oni_tools.ps1
```

Build-only dependencies are kept below `vendor/` and are ignored by Git. The
prebuilt executable, OpenNI2 runtime, file driver, license, and hash manifest
remain isolated in this directory. The upstream OpenNI binaries additionally
require the Microsoft Visual C++ 2013 x64 runtime (`MSVCR120.dll`), recorded in
`runtime_manifest.json`; it was present on the audit host.
