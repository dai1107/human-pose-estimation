import assert from "node:assert/strict";
import test from "node:test";

import {
  compareUploadStabilizationStrategies,
  stabilizeUploadTimeline,
} from "../../webui/static/workers/upload_pose_stabilizer.mjs";

function frame(timestampMs, x, visibility = 1) {
  return {
    timestamp_ms: timestampMs,
    phase: "drive",
    keypoints: [{ name: "left_wrist", x, y: 0.4, z: 0, visibility }],
  };
}

test("short occlusions are interpolated without changing formal input frames", () => {
  const input = [frame(0, 0.2), frame(33, 0.95, 0.1), frame(66, 0.4)];
  const output = stabilizeUploadTimeline(input);

  assert.ok(Math.abs(output[1].keypoints[0].x - 0.3) < 1e-9);
  assert.equal(output[1].keypoints[0].displayInterpolated, true);
  assert.equal(output[1].keypoints[0].displayValid, true);
  assert.equal(input[1].keypoints[0].x, 0.95);
  assert.equal(input[1].keypoints[0].displayInterpolated, undefined);
  assert.equal(output[1].phase, "drive");
});

test("long unobservable gaps are hidden instead of drawing jumping joints", () => {
  const input = [frame(0, 0.2), frame(500, 0.95, 0.1), frame(1000, 0.4)];
  const output = stabilizeUploadTimeline(input);

  assert.equal(output[1].keypoints[0].displayValid, false);
});

test("reliable static jitter is reduced with a bounded correction", () => {
  const input = [
    frame(0, 0.50),
    frame(33, 0.515),
    frame(66, 0.49),
    frame(99, 0.512),
    frame(132, 0.495),
  ];
  const output = stabilizeUploadTimeline(input);
  const inputRange = Math.max(...input.map(item => item.keypoints[0].x))
    - Math.min(...input.map(item => item.keypoints[0].x));
  const outputRange = Math.max(...output.map(item => item.keypoints[0].x))
    - Math.min(...output.map(item => item.keypoints[0].x));

  assert.ok(outputRange < inputRange);
  assert.ok(output.every(item => item.display_stabilized));
});

test("ablation reports lookahead and proves formal fields are isolated", () => {
  const input = [
    frame(0, 0.50),
    frame(33, 0.515),
    frame(66, 0.49),
    frame(99, 0.512),
  ];
  input.forEach((item, index) => {
    item.reps = index > 1 ? 1 : 0;
    item.candidate_count = item.reps;
  });
  const report = compareUploadStabilizationStrategies(input);
  const forward = report.strategies.find(item => item.strategy === "forward");
  const bidirectional = report.strategies.find(item => item.strategy === "bidirectional");

  assert.equal(report.formal_fields_unchanged, true);
  assert.equal(forward.metrics.algorithmic_lookahead_ms, 0);
  assert.ok(bidirectional.metrics.algorithmic_lookahead_ms > 0);
});
