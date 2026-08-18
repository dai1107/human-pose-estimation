import {
  DEFAULT_LANDMARK_QUALITY_GATE,
  LandmarkQualityGate,
} from "./landmark_quality_gate.mjs";

export const DEFAULT_DISPLAY_SMOOTHING = Object.freeze({
  profile: "ultra_responsive",
  prediction_enabled: true,
  max_gap_ms_before_reset: 250,
  min_cutoff: 1.4,
  beta: 0.08,
  d_cutoff: 1.0,
  raw_blend_enabled: true,
  max_raw_weight: 0.10,
  jitter_deadband: 0.0025,
  minimum_visibility: 0.70,
  slow_speed: 0.15,
  fast_speed: 1.20,
  extremity_raw_weight_scale: 1.0,
  core_raw_weight_scale: 0.35,
  face_raw_weight_scale: 0.0,
  world_speed_scale: 1.25,
  quality_gate_enabled: true,
  quality_minimum_confidence: 0.35,
  quality_max_speed_body_s: 18.0,
  quality_max_acceleration_body_s2: 180.0,
  quality_max_bone_length_change_ratio: 0.55,
  quality_identity_swap_margin: 0.12,
  occlusion_short_prediction_ms: 200,
  occlusion_hide_after_ms: 500,
  occlusion_reacquire_frames: 3,
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
    if (this.lastTimestampMs === null || timestampMs <= this.lastTimestampMs || timestampMs - this.lastTimestampMs >= this.maxGapMs) {
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
      min_cutoff: number("min_cutoff", 1.4, 0.01),
      beta: number("beta", 0.08, 0),
      d_cutoff: number("d_cutoff", 1.0, 0.01),
      raw_blend_enabled: candidate.raw_blend_enabled !== false,
      max_raw_weight: number("max_raw_weight", 0.10, 0, 0.45),
      jitter_deadband: number("jitter_deadband", 0.0025, 0, 0.02),
      minimum_visibility: number("minimum_visibility", 0.70, 0, 1),
      slow_speed: number("slow_speed", 0.15, 0),
      fast_speed: number("fast_speed", 1.20, 0.01),
      extremity_raw_weight_scale: number("extremity_raw_weight_scale", 1.0, 0, 1),
      core_raw_weight_scale: number("core_raw_weight_scale", 0.35, 0, 1),
      face_raw_weight_scale: number("face_raw_weight_scale", 0.0, 0, 1),
      world_speed_scale: number("world_speed_scale", 1.25, 0.01),
      quality_gate_enabled: candidate.quality_gate_enabled !== false,
      quality_minimum_confidence: number("quality_minimum_confidence", 0.35, 0, 1),
      quality_max_speed_body_s: number("quality_max_speed_body_s", 18, 0.1),
      quality_max_acceleration_body_s2: number("quality_max_acceleration_body_s2", 180, 0.1),
      quality_max_bone_length_change_ratio: number("quality_max_bone_length_change_ratio", 0.55, 0.01, 3),
      quality_identity_swap_margin: number("quality_identity_swap_margin", 0.12, 0, 1),
      occlusion_short_prediction_ms: number("occlusion_short_prediction_ms", 200, 0),
      occlusion_hide_after_ms: number("occlusion_hide_after_ms", 500, 1),
      occlusion_reacquire_frames: Math.max(1, Math.trunc(number("occlusion_reacquire_frames", 3, 1, 12))),
    };
    if (this.config.fast_speed <= this.config.slow_speed) {
      this.config.fast_speed = Math.max(1.20, this.config.slow_speed + 0.01);
    }
    if (this.config.occlusion_hide_after_ms < this.config.occlusion_short_prediction_ms) {
      this.config.occlusion_hide_after_ms = this.config.occlusion_short_prediction_ms;
    }
    this.imageQualityGate = new LandmarkQualityGate({
      ...DEFAULT_LANDMARK_QUALITY_GATE,
      enabled: this.config.quality_gate_enabled,
      minimum_confidence: this.config.quality_minimum_confidence,
      max_speed_body_s: this.config.quality_max_speed_body_s,
      max_acceleration_body_s2: this.config.quality_max_acceleration_body_s2,
      max_bone_length_change_ratio: this.config.quality_max_bone_length_change_ratio,
      identity_swap_margin: this.config.quality_identity_swap_margin,
      reacquire_frames: this.config.occlusion_reacquire_frames,
      max_gap_ms_before_reset: this.config.max_gap_ms_before_reset,
    });
    this.reset();
  }

  reset() {
    this.imageFilters = [];
    this.worldFilters = [];
    this.imageRawHistory = [];
    this.worldRawHistory = [];
    this.imageOutputHistory = [];
    this.imageMeasurementHistory = [];
    this.imageQualityGate?.reset();
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
      qualityGateEnabled: this.config.quality_gate_enabled,
      rejectedPointCount: results.reduce(
        (total, result) => total + Number(result?.quality?.rejectedPointCount || 0),
        0,
      ),
      occludedPointCount: results.reduce(
        (total, result) => total + Number(result?.quality?.occludedPointCount || 0),
        0,
      ),
      reacquiringPointCount: results.reduce(
        (total, result) => total + Number(result?.quality?.reacquiringPointCount || 0),
        0,
      ),
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
    if (elapsedMs <= 0 || elapsedMs >= this.config.max_gap_ms_before_reset) return 0;
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
    const qualityDecisions = world
      ? landmarks.map(() => null)
      : this.imageQualityGate.evaluateFrame(landmarks, timestampMs);
    const rawWeights = [];
    const filteredLandmarks = landmarks.map((point, index) => {
      const decision = qualityDecisions[index];
      if (!world && decision && !decision.accepted) {
        rawWeights.push(0);
        return this.#occludedLandmark(point, index, timestampMs, decision);
      }
      if (!filters[index]) {
        filters[index] = [
          new OneEuroFilter(this.config.min_cutoff, this.config.beta, this.config.d_cutoff, this.config.max_gap_ms_before_reset),
          new OneEuroFilter(this.config.min_cutoff, this.config.beta, this.config.d_cutoff, this.config.max_gap_ms_before_reset),
          new OneEuroFilter(this.config.min_cutoff, this.config.beta, this.config.d_cutoff, this.config.max_gap_ms_before_reset),
        ];
      }
      let measurement = point;
      if (!world && decision?.occlusion_state === "reacquiring") {
        const previous = this.imageOutputHistory[index];
        if (previous) {
          const blend = decision.reacquire_blend;
          measurement = {
            ...point,
            x: Number(previous.x) * (1 - blend) + Number(point.x) * blend,
            y: Number(previous.y) * (1 - blend) + Number(point.y) * blend,
            z: Number(previous.z || 0) * (1 - blend) + Number(point.z || 0) * blend,
          };
        }
      }
      const rawX = Number(measurement.x);
      const rawY = Number(measurement.y);
      const rawZ = Number(measurement.z || 0);
      const filteredX = filters[index][0].apply(rawX, timestampMs);
      const filteredY = filters[index][1].apply(rawY, timestampMs);
      const filteredZ = filters[index][2].apply(rawZ, timestampMs);
      const speed = this.#measuredSpeed(measurement, timestampMs, history, index, world);
      const rawWeight = this.#rawWeight(measurement, index, speed, world);
      rawWeights.push(rawWeight);
      let output = {
        x: filteredX * (1 - rawWeight) + rawX * rawWeight,
        y: filteredY * (1 - rawWeight) + rawY * rawWeight,
        z: filteredZ * (1 - rawWeight) + rawZ * rawWeight,
        visibility: Number(point.visibility ?? 1),
        presence: Number(point.presence ?? 1),
        displayValid: true,
        displayPredicted: false,
        displayMeasurementAccepted: decision?.accepted ?? true,
        displayQuality: decision?.quality ?? 1,
        displayReasonCodes: decision?.reason_codes || [],
        occlusionState: decision?.occlusion_state || "visible",
      };
      if (!world && this.config.jitter_deadband > 0) {
        const previous = this.imageOutputHistory[index];
        const pointConfidence = Math.min(output.visibility, output.presence);
        const threshold = this.config.jitter_deadband * (1 + (1 - clamp(pointConfidence)));
        if (previous && Math.hypot(output.x - previous.x, output.y - previous.y) < threshold) {
          output = { ...output, x: previous.x, y: previous.y };
        }
      }
      if (!world) {
        this.imageOutputHistory[index] = output;
        this.#recordAcceptedMeasurement(point, index, timestampMs);
      }
      return output;
    });
    const quality = world ? undefined : {
      rejectedPointCount: qualityDecisions.filter(item => item && !item.accepted).length,
      occludedPointCount: qualityDecisions.filter(item => item?.occlusion_state === "occluded").length,
      reacquiringPointCount: qualityDecisions.filter(item => item?.occlusion_state === "reacquiring").length,
      decisions: qualityDecisions,
    };
    return { landmarks: filteredLandmarks, rawWeights, quality };
  }

  #recordAcceptedMeasurement(point, index, timestampMs) {
    const previous = this.imageMeasurementHistory[index];
    let velocity = { x: 0, y: 0, z: 0 };
    if (previous) {
      const elapsedMs = timestampMs - previous.timestampMs;
      if (elapsedMs > 0 && elapsedMs <= this.config.max_gap_ms_before_reset) {
        const inverseMs = 1 / elapsedMs;
        velocity = {
          x: (Number(point.x) - previous.point.x) * inverseMs,
          y: (Number(point.y) - previous.point.y) * inverseMs,
          z: (Number(point.z || 0) - previous.point.z) * inverseMs,
        };
      }
    }
    this.imageMeasurementHistory[index] = {
      point: { x: Number(point.x), y: Number(point.y), z: Number(point.z || 0) },
      timestampMs,
      velocity,
      confidence: Math.min(Number(point.visibility ?? 1), Number(point.presence ?? 1)),
    };
  }

  #occludedLandmark(point, index, timestampMs, decision) {
    const track = this.imageMeasurementHistory[index];
    const previous = this.imageOutputHistory[index];
    if (!track || !previous) {
      return {
        ...point,
        visibility: 0,
        presence: 0,
        displayValid: false,
        displayPredicted: false,
        displayMeasurementAccepted: false,
        displayQuality: decision.quality,
        displayReasonCodes: decision.reason_codes,
        occlusionState: decision.occlusion_state,
      };
    }
    const elapsedMs = Math.max(0, timestampMs - track.timestampMs);
    const valid = elapsedMs < this.config.occlusion_hide_after_ms;
    const predictionMs = this.config.prediction_enabled
      ? Math.min(elapsedMs, this.config.occlusion_short_prediction_ms)
      : 0;
    const confidenceScale = valid
      ? Math.max(0.05, 1 - elapsedMs / this.config.occlusion_hide_after_ms)
      : 0;
    const output = {
      ...previous,
      x: previous.x + track.velocity.x * predictionMs,
      y: previous.y + track.velocity.y * predictionMs,
      z: previous.z + track.velocity.z * predictionMs,
      visibility: track.confidence * confidenceScale,
      presence: track.confidence * confidenceScale,
      displayValid: valid,
      displayPredicted: predictionMs > 0,
      displayMeasurementAccepted: false,
      displayQuality: decision.quality,
      displayReasonCodes: decision.reason_codes,
      occlusionState: decision.occlusion_state,
    };
    this.imageOutputHistory[index] = output;
    return output;
  }
}
