import assert from "node:assert/strict";
import test from "node:test";

import { BackgroundCameraMotionEstimator, CameraDiagnostics } from "../../webui/static/workers/camera_diagnostics.mjs";

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

test("background camera motion ignores the person region and reports global translation", () => {
  const estimator = new BackgroundCameraMotionEstimator();
  const width = 32;
  const height = 18;
  const frame = (dx = 0) => {
    const pixels = new Uint8Array(width * height);
    for (let y = 2; y < height - 2; y += 4) {
      for (let x = 2; x < width - 4; x += 5) pixels[y * width + x + dx] = 255;
    }
    return pixels;
  };
  estimator.observe(frame(), width, height, { x1: 0.4, y1: 0.2, x2: 0.6, y2: 0.9 });
  const result = estimator.observe(frame(2), width, height, { x1: 0.4, y1: 0.2, x2: 0.6, y2: 0.9 });
  assert.equal(result.modifies_body_3d, false);
  assert.equal(result.formal_rule_replacement_allowed, false);
  assert.ok(result.camera_motion_score > 0);
  assert.notEqual(result.state, "camera_static");
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
