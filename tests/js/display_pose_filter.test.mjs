import assert from "node:assert/strict";
import test from "node:test";

import { DisplayPoseFilter } from "../../webui/static/workers/display_pose_filter.mjs";

function pose(x = 0.5, visibility = 1) {
  return Array.from({ length: 33 }, () => ({
    x,
    y: 0.5,
    z: 0,
    visibility,
    presence: visibility,
  }));
}

test("fast extremities respond faster while core and face remain constrained", () => {
  const blended = new DisplayPoseFilter();
  const filteredOnly = new DisplayPoseFilter({ raw_blend_enabled: false });
  blended.applyImage(pose(0.4), 0);
  filteredOnly.applyImage(pose(0.4), 0);
  const blendedStep = blended.applyImage(pose(0.5), 33);
  const filteredStep = filteredOnly.applyImage(pose(0.5), 33);

  assert.ok(blendedStep.rawWeights[15] > 0);
  assert.ok(blendedStep.rawWeights[15] <= 0.45);
  assert.ok(blendedStep.rawWeights[11] < blendedStep.rawWeights[15]);
  assert.equal(blendedStep.rawWeights[0], 0);
  assert.ok(blendedStep.landmarks[15].x > filteredStep.landmarks[15].x);
  assert.ok(blendedStep.landmarks[15].x <= 0.5);
});

test("static jitter and low visibility do not enable raw blending", () => {
  const filter = new DisplayPoseFilter();
  filter.applyImage(pose(0.5), 0);
  const jitter = filter.applyImage(pose(0.502), 33);
  assert.equal(jitter.rawWeights[15], 0);
  assert.ok(Math.abs(jitter.landmarks[15].x - 0.5) < 0.002);

  const lowVisibility = pose(0.7, 0.6);
  const hidden = filter.applyImage(lowVisibility, 66);
  assert.equal(hidden.rawWeights[15], 0);
});

test("image and world state are independent and reset clears both", () => {
  const filter = new DisplayPoseFilter();
  filter.applyImage(pose(0.1), 0);
  filter.applyImage(pose(0.4), 33);
  const firstWorld = filter.applyWorld(pose(0.8), 33);
  assert.equal(firstWorld.landmarks[15].x, 0.8);
  assert.equal(firstWorld.rawWeights[15], 0);

  filter.reset();
  const resetImage = filter.applyImage(pose(0.75), 66);
  const resetWorld = filter.applyWorld(pose(0.25), 66);
  assert.equal(resetImage.landmarks[15].x, 0.75);
  assert.equal(resetWorld.landmarks[15].x, 0.25);
});

test("stage six exposes prediction capability and still caps raw weight", () => {
  const filter = new DisplayPoseFilter({ prediction_enabled: true, max_raw_weight: 0.9 });
  filter.applyImage(pose(0.1), 0);
  const result = filter.applyImage(pose(0.9), 33);
  const summary = filter.summary(result);

  assert.equal(summary.predictionEnabled, true);
  assert.ok(summary.maxRawWeight <= 0.45);
});
