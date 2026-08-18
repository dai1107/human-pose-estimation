import { FixedSampleWindow } from "./render_performance.mjs";

export const DEFAULT_CAMERA_DIAGNOSTICS = Object.freeze({
  requested_fps: 60,
  minimum_fps: 30,
  sample_fps: 5,
  low_light_luma: 55,
  fps_warning_ratio: 0.80,
  interval_anomaly_ratio: 1.80,
  duplicate_warning_ratio: 0.20,
});

export class BackgroundCameraMotionEstimator {
  constructor() { this.reset(); }

  reset() {
    this.previous = null;
    this.width = 0;
    this.height = 0;
    this.latest = {
      schema_version: 1, available: false,
      method: "background_sparse_patch_flow",
      camera_motion_score: 0, state: "camera_static",
      tracked_background_points: 0,
      modifies_body_3d: false, formal_rule_replacement_allowed: false,
    };
  }

  observe(luminance, width, height, personBounds = null) {
    if (!(luminance instanceof Uint8Array) || luminance.length !== width * height) return this.latest;
    if (!this.previous || this.width !== width || this.height !== height) {
      this.previous = luminance.slice();
      this.width = width;
      this.height = height;
      return this.latest;
    }
    const features = [];
    const bounds = personBounds && Number.isFinite(personBounds.x1) ? personBounds : null;
    for (let y = 2; y < height - 2; y += 2) {
      for (let x = 2; x < width - 2; x += 2) {
        if (bounds && x / width >= bounds.x1 - 0.08 && x / width <= bounds.x2 + 0.08
          && y / height >= bounds.y1 - 0.08 && y / height <= bounds.y2 + 0.08) continue;
        const index = y * width + x;
        const gradient = Math.abs(this.previous[index + 1] - this.previous[index - 1])
          + Math.abs(this.previous[index + width] - this.previous[index - width]);
        if (gradient >= 18) features.push({ x, y, gradient });
      }
    }
    features.sort((a, b) => b.gradient - a.gradient);
    const flows = [];
    for (const feature of features.slice(0, 48)) {
      let best = { error: Infinity, dx: 0, dy: 0 };
      for (let dy = -3; dy <= 3; dy += 1) {
        for (let dx = -3; dx <= 3; dx += 1) {
          if (feature.x + dx < 1 || feature.x + dx >= width - 1
            || feature.y + dy < 1 || feature.y + dy >= height - 1) continue;
          let error = 0;
          for (let py = -1; py <= 1; py += 1) {
            for (let px = -1; px <= 1; px += 1) {
              const oldIndex = (feature.y + py) * width + feature.x + px;
              const newIndex = (feature.y + dy + py) * width + feature.x + dx + px;
              error += Math.abs(this.previous[oldIndex] - luminance[newIndex]);
            }
          }
          if (error < best.error) best = { error, dx, dy };
        }
      }
      if (best.error / 9 <= 28) flows.push(best);
    }
    const median = values => {
      if (!values.length) return 0;
      const ordered = [...values].sort((a, b) => a - b);
      return ordered[Math.floor(ordered.length / 2)];
    };
    const dx = median(flows.map(item => item.dx));
    const dy = median(flows.map(item => item.dy));
    const motionPixels = Math.hypot(dx, dy);
    const score = flows.length >= 6 ? Math.min(1, motionPixels / 4) : 0;
    const state = score >= 0.42 ? "camera_unstable" : score >= 0.12 ? "camera_small_motion" : "camera_static";
    this.latest = {
      schema_version: 1,
      available: flows.length >= 6,
      method: "background_sparse_patch_flow",
      camera_motion_score: score,
      state,
      translation_normalized: [dx / width, dy / height],
      tracked_background_points: flows.length,
      modifies_body_3d: false,
      formal_rule_replacement_allowed: false,
    };
    this.previous = luminance.slice();
    return this.latest;
  }
}

export class CameraDiagnostics {
  constructor(config = {}) {
    this.configure(config);
  }

  configure(config = {}) {
    this.config = {
      requested_fps: Number(config.preferred_fps || config.requested_fps || 60),
      minimum_fps: Number(config.fallback_fps || config.minimum_fps || 30),
      sample_fps: Number(config.diagnostic_sample_fps || config.sample_fps || 5),
      low_light_luma: Number(config.low_light_luma || 55),
      fps_warning_ratio: Number(config.fps_warning_ratio || 0.80),
      interval_anomaly_ratio: Number(config.interval_anomaly_ratio || 1.80),
      duplicate_warning_ratio: Number(config.duplicate_warning_ratio || 0.20),
    };
    this.intervals = new FixedSampleWindow(240);
    this.cameraMotion = new BackgroundCameraMotionEstimator();
    this.reset();
  }

  reset() {
    this.settings = {};
    this.intervals.reset();
    this.lastPresentationTime = null;
    this.presentedFrames = 0;
    this.intervalAnomalies = 0;
    this.imageSamples = 0;
    this.duplicateSamples = 0;
    this.brightnessTotal = 0;
    this.cameraMotion.reset();
  }

  setSettings(settings = {}) {
    this.settings = {
      width: Number(settings.width || 0),
      height: Number(settings.height || 0),
      frameRate: Number(settings.frameRate || 0),
      deviceId: String(settings.deviceId || ""),
      resizeMode: String(settings.resizeMode || ""),
      facingMode: String(settings.facingMode || ""),
    };
  }

  observeFrame(presentationTime) {
    const now = Number(presentationTime);
    if (!Number.isFinite(now)) return;
    if (this.lastPresentationTime !== null) {
      const interval = now - this.lastPresentationTime;
      if (interval > 0 && interval < 1000) {
        this.intervals.add(interval);
        const median = this.intervals.percentile(0.50);
        if (this.intervals.count >= 10 && median > 0 && interval > median * this.config.interval_anomaly_ratio) {
          this.intervalAnomalies += 1;
        }
      }
    }
    this.lastPresentationTime = now;
    this.presentedFrames += 1;
  }

  observeImage(meanLuma, repeated, luminancePixels = null, width = 0, height = 0, personBounds = null) {
    const luminance = Number(meanLuma);
    if (!Number.isFinite(luminance)) return;
    this.imageSamples += 1;
    this.brightnessTotal += luminance;
    this.duplicateSamples += Number(Boolean(repeated));
    if (luminancePixels) this.cameraMotion.observe(luminancePixels, width, height, personBounds);
  }

  snapshot() {
    const intervalP50 = this.intervals.percentile(0.50);
    const actualPresentedFps = intervalP50 > 0 ? 1000 / intervalP50 : 0;
    const brightnessMean = this.imageSamples ? this.brightnessTotal / this.imageSamples : 0;
    const duplicateFrameRatio = this.imageSamples > 1
      ? this.duplicateSamples / (this.imageSamples - 1)
      : 0;
    const frameIntervalAnomalyRatio = this.intervals.count
      ? this.intervalAnomalies / this.intervals.count
      : 0;
    const requestedFps = this.config.requested_fps;
    const warnings = [];
    if (this.intervals.count >= 20 && actualPresentedFps < requestedFps * this.config.fps_warning_ratio) {
      warnings.push("fps_below_requested");
    }
    if (this.imageSamples >= 5 && brightnessMean < this.config.low_light_luma) {
      warnings.push("low_light");
    }
    if (this.intervals.count >= 20 && frameIntervalAnomalyRatio > 0.10) {
      warnings.push("frame_interval_unstable");
    }
    if (this.imageSamples >= 5 && duplicateFrameRatio > this.config.duplicate_warning_ratio) {
      warnings.push("duplicate_frames");
    }
    return {
      settings: { ...this.settings },
      requestedFps,
      actualPresentedFps,
      frameIntervalP50Ms: intervalP50,
      frameIntervalP95Ms: this.intervals.percentile(0.95),
      frameIntervalAnomalyRatio,
      brightnessMean,
      duplicateFrameRatio,
      sampleCount: this.imageSamples,
      warnings,
      cameraMotion: { ...this.cameraMotion.latest },
    };
  }
}
