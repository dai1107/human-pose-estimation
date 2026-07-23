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

  observeImage(meanLuma, repeated) {
    const luminance = Number(meanLuma);
    if (!Number.isFinite(luminance)) return;
    this.imageSamples += 1;
    this.brightnessTotal += luminance;
    this.duplicateSamples += Number(Boolean(repeated));
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
    };
  }
}
