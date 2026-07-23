import assert from "node:assert/strict";
import test from "node:test";

import { DisplayPosePredictor } from "../../webui/static/workers/display_pose_predictor.mjs";

function pose(wristX = 0.4, visibility = 1) {
  return [
    { name: "left_shoulder", x: 0.4, y: 0.3, z: 0, visibility, presence: visibility },
    { name: "right_shoulder", x: 0.6, y: 0.3, z: 0, visibility, presence: visibility },
    { name: "left_hip", x: 0.42, y: 0.6, z: 0, visibility, presence: visibility },
    { name: "right_hip", x: 0.58, y: 0.6, z: 0, visibility, presence: visibility },
    { name: "left_wrist", x: wristX, y: 0.45, z: 0, visibility, presence: visibility },
    { name: "left_ankle", x: 0.45, y: 0.9, z: 0, visibility, presence: visibility },
  ];
}

test("prediction horizon follows pose age and is capped at 45 ms", () => {
  const predictor = new DisplayPosePredictor();
  predictor.update(pose(0.4), 0, "person-1");
  predictor.update(pose(0.5), 20, "person-1");

  const result = predictor.predict(90);
  assert.equal(result.horizonMs, 45);
  assert.ok(result.landmarks[4].x > 0.5);
  assert.ok(result.predictedPointCount > 0);
});

test("static and low visibility points do not drift", () => {
  const staticPredictor = new DisplayPosePredictor();
  staticPredictor.update(pose(0.4), 0, "person-1");
  staticPredictor.update(pose(0.4), 20, "person-1");
  assert.equal(staticPredictor.predict(50).landmarks[4].x, 0.4);

  const hiddenPredictor = new DisplayPosePredictor();
  hiddenPredictor.update(pose(0.4, 0.6), 0, "person-1");
  hiddenPredictor.update(pose(0.6, 0.6), 20, "person-1");
  assert.equal(hiddenPredictor.predict(50).landmarks[4].x, 0.6);
});

test("body scale displacement cap and support foot lock are enforced", () => {
  const predictor = new DisplayPosePredictor();
  const first = pose(0.1);
  const second = pose(0.8);
  first[5].x = 0.40;
  second[5].x = 0.45;
  predictor.update(first, 0, "person-1");
  predictor.update(second, 10, "person-1");
  const result = predictor.predict(55, { phase: "standing" });
  const wristDisplacement = result.landmarks[4].x - 0.8;
  assert.ok(wristDisplacement <= 0.6 * 0.06 + 1e-9);
  assert.ok(result.clampedPointCount > 0);
  assert.equal(result.landmarks[5].x, 0.45);
});

test("action endpoint phases reduce overshoot without changing mid-phase response", () => {
  const predictor = new DisplayPosePredictor();
  predictor.update(pose(0.4), 0, "person-1");
  predictor.update(pose(0.5), 20, "person-1");

  const drive = predictor.predict(40, { action: "wall_ball", phase: "drive" });
  const driveWristX = drive.landmarks[4].x;
  const top = predictor.predict(40, { action: "wall_ball", phase: "top" });
  assert.ok(top.landmarks[4].x > 0.5);
  assert.ok(top.landmarks[4].x < driveWristX);
});

test("direction reversal damps prediction and gaps or identity switches reset it", () => {
  const predictor = new DisplayPosePredictor();
  predictor.update(pose(0.3), 0, "person-1");
  predictor.update(pose(0.5), 20, "person-1");
  predictor.update(pose(0.4), 40, "person-1");
  const reversed = predictor.predict(60);
  assert.ok(Math.abs(reversed.landmarks[4].x - 0.4) < 0.02);

  assert.equal(predictor.predict(200).horizonMs, 0);
  predictor.update(pose(0.8), 210, "person-2");
  assert.equal(predictor.predict(230).landmarks[4].x, 0.8);
});

test("prediction returns copies and never changes confidence", () => {
  const predictor = new DisplayPosePredictor();
  const current = pose(0.5, 0.9);
  predictor.update(pose(0.4, 0.9), 0, "person-1");
  predictor.update(current, 20, "person-1");
  const result = predictor.predict(40);

  assert.equal(current[4].x, 0.5);
  assert.equal(result.landmarks[4].visibility, 0.9);
  assert.equal(result.landmarks[4].presence, 0.9);
  assert.notEqual(result.landmarks[4], current[4]);
});
