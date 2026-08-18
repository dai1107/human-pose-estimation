import assert from "node:assert/strict";
import test from "node:test";

import {
  DisplayPoseController,
  DISPLAY_TRACKING_STATE,
} from "../../webui/static/workers/display_pose_controller.mjs";

function result(wristX = 0.4, detected = true, frameId = 1) {
  return {
    frame_id: frameId,
    pose_detected: detected,
    keypoints: detected ? [
      { name: "left_shoulder", x: 0.4, y: 0.3, z: 0, visibility: 1, presence: 1 },
      { name: "right_shoulder", x: 0.6, y: 0.3, z: 0, visibility: 1, presence: 1 },
      { name: "left_hip", x: 0.42, y: 0.6, z: 0, visibility: 1, presence: 1 },
      { name: "right_hip", x: 0.58, y: 0.6, z: 0, visibility: 1, presence: 1 },
      { name: "left_wrist", x: wristX, y: 0.45, z: 0, visibility: 1, presence: 1 },
    ] : [],
  };
}

function controller() {
  return new DisplayPoseController({
    analysis_max_pose_age_ms: 120,
    display_prediction_ms: 45,
    display_hold_ms: 250,
    display_fade_ms: 150,
    display: {
      landmark_enter_confidence: 0.50,
      landmark_exit_confidence: 0.30,
    },
    prediction: { enabled: true },
  });
}

test("short MediaPipe misses retain the last good display pose", () => {
  const display = controller();
  const lastGood = result(0.5, true, 2);
  display.update(result(0.4, true, 1), 0, "session-1");
  display.update(lastGood, 20, "session-1");

  assert.equal(display.update(result(0, false, 3), 50, "session-1"), false);
  const held = display.resolve(220, {}, "session-1");
  assert.equal(held.state, DISPLAY_TRACKING_STATE.DEGRADED);
  assert.equal(held.result, lastGood);
  assert.equal(held.shouldRender, true);
  assert.equal(held.opacity, 1);
  assert.equal(held.displayOnly, true);
});

test("tracking degrades, fades, then becomes lost at configured ages", () => {
  const display = controller();
  display.update(result(), 0, "session-1");

  assert.equal(display.resolve(120).state, DISPLAY_TRACKING_STATE.TRACKING);
  assert.equal(display.resolve(121).state, DISPLAY_TRACKING_STATE.DEGRADED);
  assert.equal(display.resolve(250).opacity, 1);
  assert.equal(display.resolve(325).opacity, 0.5);
  assert.equal(display.resolve(400).shouldRender, false);
  assert.equal(display.resolve(401).state, DISPLAY_TRACKING_STATE.LOST);
  assert.equal(display.resolve(401).result, null);
});

test("prediction stops at 45 ms and the predicted display pose is held", () => {
  const display = controller();
  display.update(result(0.4, true, 1), 0, "session-1");
  const raw = result(0.5, true, 2);
  display.update(raw, 20, "session-1");

  const predicted = display.resolve(65, {}, "session-1");
  const held = display.resolve(250, {}, "session-1");
  assert.equal(predicted.predictionHorizonMs, 45);
  assert.equal(held.predictionHorizonMs, 45);
  assert.equal(held.landmarks[4].x, predicted.landmarks[4].x);
  assert.equal(raw.keypoints[4].x, 0.5);
});

test("identity changes clear held poses and a fresh pose reacquires tracking", () => {
  const display = controller();
  display.update(result(), 0, "session-1");

  assert.equal(display.resolve(50, {}, "session-2").state, DISPLAY_TRACKING_STATE.LOST);
  display.update(result(0.7, true, 3), 60, "session-2");
  const reacquired = display.resolve(70, {}, "session-2");
  assert.equal(reacquired.state, DISPLAY_TRACKING_STATE.TRACKING);
  assert.equal(reacquired.result.frame_id, 3);
});

test("a pose without a source timestamp cannot replace the last good display pose", () => {
  const display = controller();
  const lastGood = result(0.4, true, 1);
  display.update(lastGood, 10, "session-1");

  assert.equal(display.update(result(0.9, true, 2), null, "session-1"), false);
  assert.equal(display.resolve(20, {}, "session-1").result, lastGood);
});

test("per-landmark hysteresis prevents threshold flicker and isolates invalid joints", () => {
  const display = controller();
  const wrist = confidence => {
    const current = result(0.5, true, confidence * 100);
    current.keypoints[4].visibility = confidence;
    current.keypoints[4].presence = confidence;
    return current;
  };

  display.update(wrist(0.48), 10, "session-1");
  let resolved = display.resolve(10, {}, "session-1");
  assert.equal(resolved.landmarks[4].displayValid, false);
  assert.equal(resolved.landmarks[0].displayValid, true);

  display.update(wrist(0.52), 20, "session-1");
  assert.equal(display.resolve(20).landmarks[4].displayValid, true);
  display.update(wrist(0.49), 30, "session-1");
  assert.equal(display.resolve(30).landmarks[4].displayValid, true);
  display.update(wrist(0.30), 40, "session-1");
  assert.equal(display.resolve(40).landmarks[4].displayValid, true);
  display.update(wrist(0.29), 50, "session-1");
  resolved = display.resolve(50);
  assert.equal(resolved.landmarks[4].displayValid, true);
  assert.equal(resolved.landmarks[4].displayHeld, true);

  display.update(wrist(0.29), 271, "session-1");
  resolved = display.resolve(271);
  assert.equal(resolved.landmarks[4].displayValid, false);
  assert.equal(resolved.metrics.valid_landmark_count, 4);
  assert.equal(wrist(0.29).keypoints[4].displayValid, undefined);
});

test("tracking metrics report missing bursts, flickers, and reacquisition latency", () => {
  const display = controller();
  display.update(result(0.4, true, 1), 0, "session-1");
  display.update(result(0, false, 2), 40, "session-1");
  display.update(result(0, false, 3), 70, "session-1");
  let metrics = display.resolve(100).metrics;
  assert.equal(metrics.sample_count, 3);
  assert.equal(metrics.pose_detection_rate, 1 / 3);
  assert.ok(Math.abs(metrics.pose_missing_rate - 2 / 3) < 1e-12);
  assert.equal(metrics.consecutive_missing_frames, 2);
  assert.equal(metrics.consecutive_missing_ms, 60);

  display.update(result(0.5, true, 4), 140, "session-1");
  metrics = display.resolve(140).metrics;
  assert.equal(metrics.flicker_count, 1);
  assert.equal(metrics.reacquisition_ms, 100);
  assert.equal(metrics.reacquisition_count, 1);
  assert.equal(metrics.consecutive_missing_ms, 0);

  display.update(result(0, false, 5), 200, "session-1");
  assert.equal(display.resolve(601).state, DISPLAY_TRACKING_STATE.LOST);
  display.update(result(0.6, true, 6), 650, "session-1");
  metrics = display.resolve(650).metrics;
  assert.equal(metrics.reacquisition_ms, 450);
  assert.equal(metrics.reacquisition_count, 2);
  assert.equal(metrics.flicker_count, 1);
});
