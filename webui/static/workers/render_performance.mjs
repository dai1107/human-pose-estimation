export const DEFAULT_RENDERING_CONFIG = Object.freeze({
  angle_text_fps: 12,
  metrics_fps: 5,
  stats_fps: 3,
  timing_sample_capacity: 240,
});

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, Number(value)));
}

export class FixedSampleWindow {
  constructor(capacity = 240) {
    this.configure(capacity);
  }

  configure(capacity) {
    this.capacity = Math.round(clamp(capacity, 30, 2000));
    this.values = new Float64Array(this.capacity);
    this.count = 0;
    this.cursor = 0;
  }

  reset() {
    this.count = 0;
    this.cursor = 0;
  }

  add(value) {
    const number = Number(value);
    if (!Number.isFinite(number) || number < 0) return;
    this.values[this.cursor] = number;
    this.cursor = (this.cursor + 1) % this.capacity;
    this.count = Math.min(this.capacity, this.count + 1);
  }

  percentile(ratio) {
    if (!this.count) return 0;
    const ordered = Array.from(this.values.subarray(0, this.count)).sort((a, b) => a - b);
    const index = Math.min(ordered.length - 1, Math.max(0, Math.ceil(ratio * ordered.length) - 1));
    return ordered[index];
  }
}

export class DomUpdateScheduler {
  constructor(config = {}) {
    this.configure(config);
  }

  configure(config = {}) {
    this.config = {
      angle_text_fps: clamp(config.angle_text_fps ?? 12, 1, 30),
      metrics_fps: clamp(config.metrics_fps ?? 5, 1, 10),
      stats_fps: clamp(config.stats_fps ?? 3, 1, 10),
      timing_sample_capacity: Math.round(clamp(config.timing_sample_capacity ?? 240, 30, 2000)),
    };
    this.lastUpdate = new Map();
  }

  reset() {
    this.lastUpdate.clear();
  }

  interval(channel) {
    if (channel === "angles") return 1000 / this.config.angle_text_fps;
    if (channel === "stats") return 1000 / this.config.stats_fps;
    return 1000 / this.config.metrics_fps;
  }

  due(channel, nowMs) {
    const now = Number(nowMs);
    if (!Number.isFinite(now)) return true;
    const last = this.lastUpdate.get(channel);
    if (last !== undefined && now - last < this.interval(channel)) return false;
    this.lastUpdate.set(channel, now);
    return true;
  }
}

export class RenderPerformanceMonitor {
  constructor(config = {}) {
    this.configure(config);
  }

  configure(config = {}) {
    const capacity = Math.round(clamp(config.timing_sample_capacity ?? 240, 30, 2000));
    this.samples = {
      render_loop_ms: new FixedSampleWindow(capacity),
      canvas_draw_ms: new FixedSampleWindow(capacity),
      dom_update_ms: new FixedSampleWindow(capacity),
    };
    this.phaseNames = new Array(128);
    this.phaseStarts = new Float64Array(128);
    this.phaseEnds = new Float64Array(128);
    this.phaseCursor = 0;
    this.phaseCount = 0;
    this.reset();
  }

  reset() {
    Object.values(this.samples).forEach(window => window.reset());
    this.phaseCursor = 0;
    this.phaseCount = 0;
    this.longTaskCount = 0;
    this.longTaskTotalMs = 0;
    this.longTaskPhases = Object.create(null);
  }

  record(metric, durationMs) {
    this.samples[metric]?.add(durationMs);
  }

  recordPhaseSpan(phase, startMs, endMs) {
    const start = Number(startMs);
    const end = Number(endMs);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return;
    const index = this.phaseCursor;
    this.phaseNames[index] = String(phase || "other");
    this.phaseStarts[index] = start;
    this.phaseEnds[index] = end;
    this.phaseCursor = (index + 1) % this.phaseNames.length;
    this.phaseCount = Math.min(this.phaseNames.length, this.phaseCount + 1);
  }

  recordLongTask(startMs, durationMs) {
    const start = Number(startMs);
    const duration = Number(durationMs);
    if (!Number.isFinite(start) || !Number.isFinite(duration) || duration < 0) return "other";
    const end = start + duration;
    let selected = "other";
    let selectedOverlap = 0;
    for (let offset = 0; offset < this.phaseCount; offset += 1) {
      const index = (this.phaseCursor - 1 - offset + this.phaseNames.length) % this.phaseNames.length;
      const overlap = Math.max(0, Math.min(end, this.phaseEnds[index]) - Math.max(start, this.phaseStarts[index]));
      if (overlap > selectedOverlap) {
        selectedOverlap = overlap;
        selected = this.phaseNames[index] || "other";
      }
    }
    this.longTaskCount += 1;
    this.longTaskTotalMs += duration;
    this.longTaskPhases[selected] = (this.longTaskPhases[selected] || 0) + 1;
    return selected;
  }

  snapshot() {
    return {
      renderLoopP95Ms: this.samples.render_loop_ms.percentile(0.95),
      canvasDrawP95Ms: this.samples.canvas_draw_ms.percentile(0.95),
      domUpdateP95Ms: this.samples.dom_update_ms.percentile(0.95),
      mainThreadLongTaskCount: this.longTaskCount,
      mainThreadLongTaskTotalMs: this.longTaskTotalMs,
      longTaskPhases: { ...this.longTaskPhases },
    };
  }
}
