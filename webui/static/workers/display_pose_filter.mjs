export const DEFAULT_DISPLAY_SMOOTHING = Object.freeze({
  profile: "ultra_responsive",
  prediction_enabled: true,
  max_gap_ms_before_reset: 250,
  min_cutoff: 2.2,
  beta: 0.12,
  d_cutoff: 1.0,
  raw_blend_enabled: true,
  max_raw_weight: 0.45,
  minimum_visibility: 0.70,
  slow_speed: 0.15,
  fast_speed: 1.20,
  extremity_raw_weight_scale: 1.0,
  core_raw_weight_scale: 0.35,
  face_raw_weight_scale: 0.0,
  world_speed_scale: 1.25,
});

export const FACE_LANDMARKS = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
export const CORE_LANDMARKS = new Set([11, 12, 23, 24]);
export const EXTREMITY_LANDMARKS = new Set([15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32]);

function clamp(value, low = 0, high = 1) {
  return Math.max(low, Math.min(high, Number(value)));
}

class LowPassFilter {
  constructor() { this.value = null; }
  apply(value, alpha) {
    this.value = this.value === null ? value : alpha * value + (1 - alpha) * this.value;
    return this.value;
  }
  reset() { this.value = null; }
}

class OneEuroFilter {
  constructor(minCutoff, beta, derivativeCutoff, maxGapMs) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.derivativeCutoff = derivativeCutoff;
    this.maxGapMs = maxGapMs;
    this.valueFilter = new LowPassFilter();
    this.derivativeFilter = new LowPassFilter();
    this.lastRaw = null;
    this.lastTimestampMs = null;
  }
  alpha(cutoff, dtSeconds) {
    const tau = 1 / (2 * Math.PI * Math.max(1e-6, cutoff));
    return 1 / (1 + tau / dtSeconds);
  }
  apply(value, timestampMs) {
    if (!Number.isFinite(value)) return value;
    if (this.lastTimestampMs === null || timestampMs <= this.lastTimestampMs || timestampMs - this.lastTimestampMs > this.maxGapMs) {
      this.reset();
      this.lastRaw = value;
      this.lastTimestampMs = timestampMs;
      return this.valueFilter.apply(value, 1);
    }
    const dt = (timestampMs - this.lastTimestampMs) / 1000;
    const derivative = (value - this.lastRaw) / dt;
    const filteredDerivative = this.derivativeFilter.apply(derivative, this.alpha(this.derivativeCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(filteredDerivative);
    const filtered = this.valueFilter.apply(value, this.alpha(cutoff, dt));
    this.lastRaw = value;
    this.lastTimestampMs = timestampMs;
    return filtered;
  }
  reset() {
    this.valueFilter.reset();
    this.derivativeFilter.reset();
    this.lastRaw = null;
    this.lastTimestampMs = null;
  }
}

export class DisplayPoseFilter {
  constructor(config = {}) {
    this.config = { ...DEFAULT_DISPLAY_SMOOTHING };
    this.configure(config);
    this.reset();
  }

  configure(value = {}) {
    const candidate = value && typeof value === "object" ? value : {};
    const number = (name, fallback, low = 0, high = Number.POSITIVE_INFINITY) => {
      const parsed = Number(candidate[name]);
      return Number.isFinite(parsed) ? clamp(parsed, low, high) : fallback;
    };
    this.config = {
      ...DEFAULT_DISPLAY_SMOOTHING,
      profile: candidate.profile === "ultra_responsive" ? candidate.profile : "ultra_responsive",
      prediction_enabled: candidate.prediction_enabled !== false,
      max_gap_ms_before_reset: number("max_gap_ms_before_reset", 250, 1),
      min_cutoff: number("min_cutoff", 2.2, 0.01),
      beta: number("beta", 0.12, 0),
      d_cutoff: number("d_cutoff", 1.0, 0.01),
      raw_blend_enabled: candidate.raw_blend_enabled !== false,
      max_raw_weight: number("max_raw_weight", 0.45, 0, 0.45),
      minimum_visibility: number("minimum_visibility", 0.70, 0, 1),
      slow_speed: number("slow_speed", 0.15, 0),
      fast_speed: number("fast_speed", 1.20, 0.01),
      extremity_raw_weight_scale: number("extremity_raw_weight_scale", 1.0, 0, 1),
      core_raw_weight_scale: number("core_raw_weight_scale", 0.35, 0, 1),
      face_raw_weight_scale: number("face_raw_weight_scale", 0.0, 0, 1),
      world_speed_scale: number("world_speed_scale", 1.25, 0.01),
    };
    if (this.config.fast_speed <= this.config.slow_speed) {
      this.config.fast_speed = Math.max(1.20, this.config.slow_speed + 0.01);
    }
    this.reset();
  }

  reset() {
    this.imageFilters = [];
    this.worldFilters = [];
    this.imageRawHistory = [];
    this.worldRawHistory = [];
  }

  applyImage(landmarks, timestampMs) {
    return this.#filterLandmarks(landmarks, timestampMs, this.imageFilters, this.imageRawHistory, false);
  }

  applyWorld(landmarks, timestampMs) {
    return this.#filterLandmarks(landmarks, timestampMs, this.worldFilters, this.worldRawHistory, true);
  }

  summary(...results) {
    const rawWeights = results.flatMap(result => result?.rawWeights || []);
    const blendedWeights = rawWeights.filter(weight => weight > 0);
    return {
      profile: this.config.profile,
      predictionEnabled: this.config.prediction_enabled,
      rawBlendEnabled: this.config.raw_blend_enabled,
      blendedPointCount: blendedWeights.length,
      meanRawWeight: blendedWeights.length
        ? blendedWeights.reduce((total, weight) => total + weight, 0) / blendedWeights.length
        : 0,
      maxRawWeight: rawWeights.length ? Math.max(...rawWeights) : 0,
    };
  }

  #rawWeightScale(index) {
    if (FACE_LANDMARKS.has(index)) return this.config.face_raw_weight_scale;
    if (CORE_LANDMARKS.has(index)) return this.config.core_raw_weight_scale;
    if (EXTREMITY_LANDMARKS.has(index)) return this.config.extremity_raw_weight_scale;
    return 0.65;
  }

  #measuredSpeed(point, timestampMs, history, index, world) {
    const previous = history[index];
    const current = {
      x: Number(point.x),
      y: Number(point.y),
      z: Number(point.z || 0),
      timestampMs,
    };
    history[index] = current;
    if (!previous) return 0;
    const elapsedMs = timestampMs - previous.timestampMs;
    if (elapsedMs <= 0 || elapsedMs > this.config.max_gap_ms_before_reset) return 0;
    const inverseSeconds = 1000 / elapsedMs;
    const dx = current.x - previous.x;
    const dy = current.y - previous.y;
    const dz = current.z - previous.z;
    return Math.sqrt(dx * dx + dy * dy + (world ? dz * dz : 0)) * inverseSeconds;
  }

  #rawWeight(point, index, speed, world) {
    if (!this.config.raw_blend_enabled) return 0;
    const confidence = Math.min(Number(point.visibility ?? 1), Number(point.presence ?? 1));
    const visibilityWeight = clamp(
      (confidence - this.config.minimum_visibility)
        / Math.max(1e-6, 1 - this.config.minimum_visibility),
    );
    const speedScale = world ? this.config.world_speed_scale : 1;
    const slow = this.config.slow_speed * speedScale;
    const fast = this.config.fast_speed * speedScale;
    const speedRatio = clamp((speed - slow) / Math.max(1e-6, fast - slow));
    return clamp(
      speedRatio * visibilityWeight * this.config.max_raw_weight * this.#rawWeightScale(index),
      0,
      this.config.max_raw_weight,
    );
  }

  #filterLandmarks(landmarks, timestampMs, filters, history, world) {
    if (!Array.isArray(landmarks)) return { landmarks: [], rawWeights: [] };
    const rawWeights = [];
    const filteredLandmarks = landmarks.map((point, index) => {
      if (!filters[index]) {
        filters[index] = [
          new OneEuroFilter(this.config.min_cutoff, this.config.beta, this.config.d_cutoff, this.config.max_gap_ms_before_reset),
          new OneEuroFilter(this.config.min_cutoff, this.config.beta, this.config.d_cutoff, this.config.max_gap_ms_before_reset),
          new OneEuroFilter(this.config.min_cutoff, this.config.beta, this.config.d_cutoff, this.config.max_gap_ms_before_reset),
        ];
      }
      const rawX = Number(point.x);
      const rawY = Number(point.y);
      const rawZ = Number(point.z || 0);
      const filteredX = filters[index][0].apply(rawX, timestampMs);
      const filteredY = filters[index][1].apply(rawY, timestampMs);
      const filteredZ = filters[index][2].apply(rawZ, timestampMs);
      const speed = this.#measuredSpeed(point, timestampMs, history, index, world);
      const rawWeight = this.#rawWeight(point, index, speed, world);
      rawWeights.push(rawWeight);
      return {
        x: filteredX * (1 - rawWeight) + rawX * rawWeight,
        y: filteredY * (1 - rawWeight) + rawY * rawWeight,
        z: filteredZ * (1 - rawWeight) + rawZ * rawWeight,
        visibility: Number(point.visibility ?? 1),
        presence: Number(point.presence ?? 1),
      };
    });
    return { landmarks: filteredLandmarks, rawWeights };
  }
}
