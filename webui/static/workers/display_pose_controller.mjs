import { DisplayPosePredictor } from "./display_pose_predictor.mjs";

export const DISPLAY_TRACKING_STATE = Object.freeze({
  TRACKING: "TRACKING",
  DEGRADED: "DEGRADED",
  LOST: "LOST",
});

export const DEFAULT_DISPLAY_POSE_CONFIG = Object.freeze({
  analysis_max_pose_age_ms: 120,
  display_prediction_ms: 45,
  display_hold_ms: 250,
  display_fade_ms: 150,
  landmark_enter_confidence: 0.50,
  landmark_exit_confidence: 0.30,
  landmark_hold_ms: 220,
});

const DETECTION_WINDOW_SAMPLES = 120;

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, Number(value)));
}

function finiteNumber(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function hasDisplayPose(result) {
  return Boolean(
    result?.pose_detected
    && Array.isArray(result.keypoints)
    && result.keypoints.length > 0,
  );
}

function pointKey(point, index) {
  return typeof point?.name === "string" && point.name ? point.name : String(index);
}

function pointConfidence(point) {
  return Math.min(
    clamp(finiteNumber(point?.visibility, 1), 0, 1),
    clamp(finiteNumber(point?.presence, 1), 0, 1),
  );
}

/**
 * Owns display-only pose continuity. Its output must only be consumed by Canvas
 * rendering; formal analysis continues to receive the current raw pose frame.
 */
export class DisplayPoseController {
  constructor(config = {}, predictor = null) {
    this.predictor = predictor || new DisplayPosePredictor();
    this.config = { ...DEFAULT_DISPLAY_POSE_CONFIG };
    this.configure(config);
  }

  configure(value = {}) {
    const candidate = value && typeof value === "object" ? value : {};
    const analysisMaxAge = finiteNumber(
      candidate.analysis_max_pose_age_ms ?? candidate.max_pose_age_ms,
      DEFAULT_DISPLAY_POSE_CONFIG.analysis_max_pose_age_ms,
    );
    const predictionMs = finiteNumber(
      candidate.display_prediction_ms,
      DEFAULT_DISPLAY_POSE_CONFIG.display_prediction_ms,
    );
    const holdMs = finiteNumber(
      candidate.display_hold_ms,
      DEFAULT_DISPLAY_POSE_CONFIG.display_hold_ms,
    );
    const fadeMs = finiteNumber(
      candidate.display_fade_ms,
      DEFAULT_DISPLAY_POSE_CONFIG.display_fade_ms,
    );
    const display = candidate.display && typeof candidate.display === "object"
      ? candidate.display
      : candidate;
    const enterConfidence = finiteNumber(
      display.landmark_enter_confidence,
      DEFAULT_DISPLAY_POSE_CONFIG.landmark_enter_confidence,
    );
    const exitConfidence = finiteNumber(
      display.landmark_exit_confidence,
      DEFAULT_DISPLAY_POSE_CONFIG.landmark_exit_confidence,
    );
    const landmarkHoldMs = finiteNumber(
      display.landmark_hold_ms,
      DEFAULT_DISPLAY_POSE_CONFIG.landmark_hold_ms,
    );
    this.config = {
      analysis_max_pose_age_ms: Math.max(1, analysisMaxAge),
      display_prediction_ms: clamp(predictionMs, 0, 60),
      display_hold_ms: Math.max(1, holdMs),
      display_fade_ms: Math.max(1, fadeMs),
      landmark_enter_confidence: clamp(enterConfidence, 0, 1),
      landmark_exit_confidence: clamp(exitConfidence, 0, 1),
      landmark_hold_ms: clamp(landmarkHoldMs, 0, 500),
    };
    this.config.display_hold_ms = Math.max(
      this.config.analysis_max_pose_age_ms,
      this.config.display_hold_ms,
    );
    this.config.landmark_exit_confidence = Math.min(
      this.config.landmark_exit_confidence,
      this.config.landmark_enter_confidence,
    );
    this.predictor.configure({
      ...(candidate.prediction || {}),
      max_horizon_ms: this.config.display_prediction_ms,
    });
    this.reset();
  }

  reset(identityKey = null) {
    this.identityKey = identityKey;
    this.#clearDisplayPose();
    this.#resetTrackingMetrics();
  }

  #clearDisplayPose() {
    this.lastGoodResult = null;
    this.lastGoodTimestampMs = null;
    this.state = DISPLAY_TRACKING_STATE.LOST;
    this.landmarkVisibility = new Map();
    this.landmarkStates = new Map();
    this.outputLandmarks = [];
    this.predictor.reset(this.identityKey);
  }

  #resetTrackingMetrics() {
    this.detectionSamples = [];
    this.lastObservationTimestampMs = null;
    this.missingSinceTimestampMs = null;
    this.consecutiveMissingFrames = 0;
    this.flickerCount = 0;
    this.reacquisitionMs = 0;
    this.reacquisitionCount = 0;
    this.everDetected = false;
  }

  /**
   * Observes a current inference result. Missing poses deliberately do not
   * erase the last good display pose or predictor/filter history.
   */
  update(result, timestampMs, identityKey = null) {
    const timestamp = timestampMs === null || timestampMs === undefined
      ? Number.NaN
      : Number(timestampMs);
    if (this.identityKey !== null && identityKey !== this.identityKey) {
      this.reset(identityKey);
    } else if (this.identityKey === null) {
      this.identityKey = identityKey;
    }
    if (!Number.isFinite(timestamp)) return false;
    const detected = hasDisplayPose(result);
    this.#observeDetection(detected, timestamp);
    if (!detected) return false;
    const displayPoints = this.#stabilizeLandmarks(result.keypoints, timestamp);
    if (!this.predictor.update(displayPoints, timestamp, identityKey)) return false;
    this.lastGoodResult = result;
    this.lastGoodTimestampMs = timestamp;
    this.state = DISPLAY_TRACKING_STATE.TRACKING;
    return true;
  }

  resolve(targetTimestampMs, context = {}, identityKey = this.identityKey) {
    const target = targetTimestampMs === null || targetTimestampMs === undefined
      ? Number.NaN
      : Number(targetTimestampMs);
    if (identityKey !== this.identityKey || !Number.isFinite(target) || this.lastGoodTimestampMs === null) {
      if (identityKey !== this.identityKey) this.reset(identityKey);
      return this.#lostResult(Number.POSITIVE_INFINITY, target);
    }

    const ageMs = Math.max(0, target - this.lastGoodTimestampMs);
    if (ageMs > this.config.analysis_max_pose_age_ms && this.missingSinceTimestampMs === null) {
      this.missingSinceTimestampMs = this.lastGoodTimestampMs;
      this.consecutiveMissingFrames = Math.max(1, this.consecutiveMissingFrames);
    }
    const lostAfterMs = this.config.display_hold_ms + this.config.display_fade_ms;
    if (ageMs > lostAfterMs) {
      this.#clearDisplayPose();
      return this.#lostResult(ageMs, target);
    }

    this.state = ageMs <= this.config.analysis_max_pose_age_ms
      ? DISPLAY_TRACKING_STATE.TRACKING
      : DISPLAY_TRACKING_STATE.DEGRADED;
    const predictionTarget = Math.min(
      target,
      this.lastGoodTimestampMs + this.config.display_prediction_ms,
    );
    const prediction = this.predictor.predict(predictionTarget, context);
    this.outputLandmarks = prediction.landmarks.map((point, index) => ({
      ...point,
      displayValid: this.landmarkVisibility.get(pointKey(point, index)) === true,
    }));
    const opacity = ageMs <= this.config.display_hold_ms
      ? 1
      : clamp(
        1 - (ageMs - this.config.display_hold_ms) / this.config.display_fade_ms,
        0,
        1,
      );
    return {
      state: this.state,
      result: this.lastGoodResult,
      landmarks: this.outputLandmarks,
      opacity,
      ageMs,
      predictionHorizonMs: prediction.horizonMs,
      shouldRender: opacity > 0,
      displayOnly: true,
      metrics: this.#trackingMetrics(target),
    };
  }

  #stabilizeLandmarks(points, timestampMs) {
    const observed = new Set();
    const stabilized = points.map((point, index) => {
      const key = pointKey(point, index);
      observed.add(key);
      const previous = this.landmarkStates.get(key) || {
        visible: false,
        lowSinceMs: null,
        lastReliable: null,
      };
      const wasVisible = previous.visible;
      const threshold = wasVisible
        ? this.config.landmark_exit_confidence
        : this.config.landmark_enter_confidence;
      const confidence = pointConfidence(point);
      let visible = wasVisible ? confidence >= threshold : confidence > threshold;
      let lowSinceMs = previous.lowSinceMs;
      let output = point;
      if (wasVisible && !visible) {
        if (lowSinceMs === null) lowSinceMs = timestampMs;
        const withinHold = timestampMs - lowSinceMs <= this.config.landmark_hold_ms;
        if (withinHold && previous.lastReliable) {
          visible = true;
          output = { ...previous.lastReliable, displayHeld: true };
        }
      } else {
        lowSinceMs = null;
      }
      const lastReliable = confidence >= this.config.landmark_exit_confidence
        ? { ...point }
        : previous.lastReliable;
      this.landmarkStates.set(key, { visible, lowSinceMs, lastReliable });
      this.landmarkVisibility.set(key, visible);
      return output;
    });
    for (const key of this.landmarkVisibility.keys()) {
      if (!observed.has(key)) {
        this.landmarkVisibility.set(key, false);
        this.landmarkStates.delete(key);
      }
    }
    return stabilized;
  }

  #observeDetection(detected, timestampMs) {
    if (
      this.lastObservationTimestampMs !== null
      && timestampMs <= this.lastObservationTimestampMs
    ) {
      return;
    }
    this.detectionSamples.push(Boolean(detected));
    if (this.detectionSamples.length > DETECTION_WINDOW_SAMPLES) this.detectionSamples.shift();
    if (detected) {
      if (this.missingSinceTimestampMs !== null && this.everDetected) {
        this.reacquisitionMs = Math.max(0, timestampMs - this.missingSinceTimestampMs);
        this.reacquisitionCount += 1;
        if (this.reacquisitionMs <= this.config.display_hold_ms) this.flickerCount += 1;
      }
      this.missingSinceTimestampMs = null;
      this.consecutiveMissingFrames = 0;
      this.everDetected = true;
    } else {
      if (this.missingSinceTimestampMs === null) this.missingSinceTimestampMs = timestampMs;
      this.consecutiveMissingFrames += 1;
    }
    this.lastObservationTimestampMs = timestampMs;
  }

  #trackingMetrics(targetTimestampMs) {
    const sampleCount = this.detectionSamples.length;
    const detectedCount = this.detectionSamples.filter(Boolean).length;
    const poseDetectionRate = sampleCount > 0 ? detectedCount / sampleCount : 0;
    const target = Number.isFinite(targetTimestampMs)
      ? targetTimestampMs
      : this.lastObservationTimestampMs;
    const consecutiveMissingMs = this.missingSinceTimestampMs !== null && Number.isFinite(target)
      ? Math.max(0, target - this.missingSinceTimestampMs)
      : 0;
    return {
      state: this.state,
      sample_count: sampleCount,
      pose_detection_rate: poseDetectionRate,
      pose_missing_rate: sampleCount > 0 ? 1 - poseDetectionRate : 0,
      consecutive_missing_frames: this.consecutiveMissingFrames,
      consecutive_missing_ms: consecutiveMissingMs,
      flicker_count: this.flickerCount,
      reacquisition_ms: this.reacquisitionMs,
      reacquisition_count: this.reacquisitionCount,
      valid_landmark_count: [...this.landmarkVisibility.values()].filter(Boolean).length,
      display_only: true,
    };
  }

  #lostResult(ageMs = Number.POSITIVE_INFINITY, targetTimestampMs = Number.NaN) {
    this.state = DISPLAY_TRACKING_STATE.LOST;
    return {
      state: this.state,
      result: null,
      landmarks: [],
      opacity: 0,
      ageMs,
      predictionHorizonMs: 0,
      shouldRender: false,
      displayOnly: true,
      metrics: this.#trackingMetrics(targetTimestampMs),
    };
  }
}
