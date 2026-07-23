import assert from "node:assert/strict";
import test from "node:test";

import { CameraDiagnostics } from "../../webui/static/workers/camera_diagnostics.mjs";

test("camera diagnostics preserves actual track settings", () => {
  const diagnostics = new CameraDiagnostics();
  diagnostics.setSettings({
    width: 640,
    height: 480,
    frameRate: 59.94,
    deviceId: "camera-1",
    resizeMode: "none",
    facingMode: "user",
  });
  const result = diagnostics.snapshot();
  assert.equal(result.settings.width, 640);
  assert.equal(result.settings.frameRate, 59.94);
  assert.equal(result.settings.deviceId, "camera-1");
});

test("healthy 60 FPS stream produces no warnings", () => {
  const diagnostics = new CameraDiagnostics({ preferred_fps: 60 });
  for (let index = 0; index < 60; index += 1) {
    diagnostics.observeFrame(index * (1000 / 60));
    if (index % 10 === 0) diagnostics.observeImage(110, false);
  }
  const result = diagnostics.snapshot();
  assert.ok(result.actualPresentedFps > 59);
  assert.deepEqual(result.warnings, []);
});

test("low light, low FPS, interval anomalies and duplicate frames are reported", () => {
  const diagnostics = new CameraDiagnostics({ preferred_fps: 60 });
  let now = 0;
  for (let index = 0; index < 30; index += 1) {
    now += index % 5 === 0 ? 100 : 40;
    diagnostics.observeFrame(now);
    diagnostics.observeImage(20, index > 0);
  }
  const result = diagnostics.snapshot();
  assert.ok(result.warnings.includes("fps_below_requested"));
  assert.ok(result.warnings.includes("low_light"));
  assert.ok(result.warnings.includes("frame_interval_unstable"));
  assert.ok(result.warnings.includes("duplicate_frames"));
});
