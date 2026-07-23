import assert from "node:assert/strict";
import test from "node:test";

import {
  DomUpdateScheduler,
  FixedSampleWindow,
  RenderPerformanceMonitor,
} from "../../webui/static/workers/render_performance.mjs";

test("fixed sample window is bounded and calculates p95", () => {
  const window = new FixedSampleWindow(30);
  for (let value = 1; value <= 60; value += 1) window.add(value);
  assert.equal(window.count, 30);
  assert.equal(window.percentile(0.95), 59);
});

test("DOM scheduler enforces independent metrics, stats and angle rates", () => {
  const scheduler = new DomUpdateScheduler({ metrics_fps: 5, stats_fps: 2, angle_text_fps: 10 });
  assert.equal(scheduler.due("metrics", 0), true);
  assert.equal(scheduler.due("metrics", 100), false);
  assert.equal(scheduler.due("metrics", 200), true);
  assert.equal(scheduler.due("stats", 200), true);
  assert.equal(scheduler.due("stats", 600), false);
  assert.equal(scheduler.due("angles", 0), true);
  assert.equal(scheduler.due("angles", 99), false);
  assert.equal(scheduler.due("angles", 100), true);
});

test("performance monitor reports p95 and attributes long tasks to a phase", () => {
  const monitor = new RenderPerformanceMonitor({ timing_sample_capacity: 30 });
  for (let index = 1; index <= 30; index += 1) {
    monitor.record("render_loop_ms", index);
    monitor.record("canvas_draw_ms", index / 2);
    monitor.record("dom_update_ms", index / 4);
  }
  monitor.recordPhaseSpan("render", 100, 160);
  assert.equal(monitor.recordLongTask(105, 50), "render");
  const snapshot = monitor.snapshot();
  assert.equal(snapshot.renderLoopP95Ms, 29);
  assert.equal(snapshot.canvasDrawP95Ms, 14.5);
  assert.equal(snapshot.domUpdateP95Ms, 7.25);
  assert.equal(snapshot.mainThreadLongTaskCount, 1);
  assert.deepEqual(snapshot.longTaskPhases, { render: 1 });
});
