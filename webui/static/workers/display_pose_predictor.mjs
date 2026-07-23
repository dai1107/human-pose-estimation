export const DEFAULT_DISPLAY_PREDICTION = Object.freeze({
  enabled: true,
  mode: "constant_velocity",
  max_horizon_ms: 45,
  maximum_body_scale_displacement: 0.06,
  minimum_visibility: 0.70,
  velocity_decay: 0.85,
  disable_after_gap_ms: 100,
  reversal_strength: 0.25,
  core_prediction_scale: 0.45,
  face_prediction_scale: 0.0,
  support_foot_horizontal_scale: 0.0,
});

const FACE_NAMES = new Set([
  "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
  "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
]);
const CORE_NAMES = new Set(["left_shoulder", "right_shoulder", "left_hip", "right_hip"]);
const EXTREMITY_NAMES = new Set([
  "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
  "left_thumb", "right_thumb", "left_ankle", "right_ankle", "left_heel", "right_heel",
  "left_foot_index", "right_foot_index",
]);
const FOOT_NAMES = new Set([
  "left_ankle", "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]);
const STABLE_PHASES = new Set(["idle", "ready", "stand", "standing"]);

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, Number(value)));
}

function finite(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function pointConfidence(point) {
  return Math.min(finite(point?.visibility, 1), finite(point?.presence, 1));
}

function pointKey(point, index) {
  return typeof point?.name === "string" && point.name ? point.name : String(index);
}

function clonePoint(point) {
  return {
    ...point,
    x: finite(point?.x),
    y: finite(point?.y),
    z: finite(point?.z),
  };
}

export class DisplayPosePredictor {
  constructor(config = {}) {
    this.config = { ...DEFAULT_DISPLAY_PREDICTION };
    this.configure(config);
  }

  configure(value = {}) {
    const candidate = value && typeof value === "object" ? value : {};
    const number = (name, fallback, low = 0, high = Number.POSITIVE_INFINITY) => {
      const parsed = Number(candidate[name]);
      return Number.isFinite(parsed) ? clamp(parsed, low, high) : fallback;
    };
    this.config = {
      ...DEFAULT_DISPLAY_PREDICTION,
      enabled: candidate.enabled !== false,
      mode: candidate.mode === "constant_velocity" ? candidate.mode : "constant_velocity",
      max_horizon_ms: number("max_horizon_ms", 45, 0, 60),
      maximum_body_scale_displacement: number("maximum_body_scale_displacement", 0.06, 0, 0.20),
      minimum_visibility: number("minimum_visibility", 0.70, 0, 1),
      velocity_decay: number("velocity_decay", 0.85, 0, 1),
      disable_after_gap_ms: number("disable_after_gap_ms", 100, 1),
      reversal_strength: number("reversal_strength", 0.25, 0, 1),
      core_prediction_scale: number("core_prediction_scale", 0.45, 0, 1),
      face_prediction_scale: number("face_prediction_scale", 0.0, 0, 1),
      support_foot_horizontal_scale: number("support_foot_horizontal_scale", 0.0, 0, 1),
    };
    this.reset();
  }

  reset(identityKey = null) {
    this.identityKey = identityKey;
    this.timestampMs = null;
    this.points = [];
    this.outputPoints = [];
    this.states = new Map();
    this.lastSummary = {
      landmarks: this.outputPoints,
      enabled: this.config.enabled,
      applied: false,
      horizonMs: 0,
      predictedPointCount: 0,
      clampedPointCount: 0,
    };
  }

  update(points, timestampMs, identityKey = null) {
    const time = Number(timestampMs);
    if (!Array.isArray(points) || !Number.isFinite(time)) {
      this.reset(identityKey);
      return false;
    }
    if (this.identityKey !== null && identityKey !== this.identityKey) this.reset(identityKey);
    else if (this.identityKey === null) this.identityKey = identityKey;

    const elapsedMs = this.timestampMs === null ? 0 : time - this.timestampMs;
    if (this.timestampMs !== null && (elapsedMs <= 0 || elapsedMs > this.config.disable_after_gap_ms)) {
      this.states.clear();
    }

    const nextStates = new Map();
    const nextPoints = points.map((input, index) => {
      const point = clonePoint(input);
      const key = pointKey(point, index);
      const previous = this.states.get(key);
      let velocity = { x: 0, y: 0, z: 0 };
      let reversed = false;
      if (previous && elapsedMs > 0 && elapsedMs <= this.config.disable_after_gap_ms) {
        const measured = {
          x: (point.x - previous.point.x) / elapsedMs,
          y: (point.y - previous.point.y) / elapsedMs,
          z: (point.z - previous.point.z) / elapsedMs,
        };
        const dot = measured.x * previous.velocity.x
          + measured.y * previous.velocity.y
          + measured.z * previous.velocity.z;
        const previousMagnitude = Math.hypot(
          previous.velocity.x,
          previous.velocity.y,
          previous.velocity.z,
        );
        const measuredMagnitude = Math.hypot(measured.x, measured.y, measured.z);
        reversed = dot < 0 && previousMagnitude > 1e-5 && measuredMagnitude > 1e-5;
        velocity = {
          x: 0.6 * previous.velocity.x + 0.4 * measured.x,
          y: 0.6 * previous.velocity.y + 0.4 * measured.y,
          z: 0.6 * previous.velocity.z + 0.4 * measured.z,
        };
      }
      nextStates.set(key, { point, velocity, reversed });
      return point;
    });
    this.states = nextStates;
    this.points = nextPoints;
    this.timestampMs = time;
    return true;
  }

  predict(targetTimestampMs, context = {}) {
    const target = Number(targetTimestampMs);
    if (
      !this.config.enabled
      || this.timestampMs === null
      || !Number.isFinite(target)
      || target <= this.timestampMs
    ) {
      this.#copyCurrentToOutput();
      return this.#result(0, 0, 0);
    }
    const rawAgeMs = target - this.timestampMs;
    if (rawAgeMs > this.config.disable_after_gap_ms) {
      this.#copyCurrentToOutput();
      return this.#result(0, 0, 0);
    }
    const horizonMs = clamp(rawAgeMs, 0, this.config.max_horizon_ms);
    const bodyScale = this.#bodyScale();
    const maximumDisplacement = bodyScale * this.config.maximum_body_scale_displacement;
    let predictedPointCount = 0;
    let clampedPointCount = 0;
    this.outputPoints.length = this.points.length;
    for (let index = 0; index < this.points.length; index += 1) {
      const point = this.points[index];
      const key = pointKey(point, index);
      const state = this.states.get(key);
      const name = point.name || "";
      const output = this.outputPoints[index] || {};
      Object.assign(output, point);
      this.outputPoints[index] = output;
      if (!state || pointConfidence(point) < this.config.minimum_visibility) continue;
      let strength = this.#predictionScale(name) * this.#phaseScale(name, context);
      if (state.reversed) strength *= this.config.reversal_strength;
      const supportFoot = FOOT_NAMES.has(name) && this.#isSupportFoot(name, context);
      const decay = this.config.velocity_decay;
      let dx = state.velocity.x * horizonMs * strength * decay;
      let dy = state.velocity.y * horizonMs * strength * decay;
      let dz = state.velocity.z * horizonMs * strength * decay;
      if (supportFoot) dx *= this.config.support_foot_horizontal_scale;
      const displacement = Math.hypot(dx, dy, dz);
      if (maximumDisplacement > 0 && displacement > maximumDisplacement) {
        const scale = maximumDisplacement / displacement;
        dx *= scale;
        dy *= scale;
        dz *= scale;
        clampedPointCount += 1;
      }
      if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) > 1e-12) predictedPointCount += 1;
      output.x = point.x + dx;
      output.y = point.y + dy;
      output.z = point.z + dz;
    }
    return this.#result(horizonMs, predictedPointCount, clampedPointCount);
  }

  #predictionScale(name) {
    if (FACE_NAMES.has(name)) return this.config.face_prediction_scale;
    if (CORE_NAMES.has(name)) return this.config.core_prediction_scale;
    if (EXTREMITY_NAMES.has(name)) return 1;
    return 0.70;
  }

  #isSupportFoot(name, context) {
    if (Array.isArray(context.supportFootNames) && context.supportFootNames.includes(name)) return true;
    const phase = String(context.phase || "").toLowerCase();
    const action = String(context.action || "").toLowerCase();
    if (STABLE_PHASES.has(phase)) return true;
    if (action === "wall_ball" && !["jump", "flight"].includes(phase)) return true;
    return ["landing", "landed", "contact"].includes(phase);
  }

  #phaseScale(name, context) {
    const phase = String(context.phase || "").toLowerCase();
    const action = String(context.action || "").toLowerCase();
    if (action === "lunge" && phase === "bottom" && /_(hip|knee|ankle)$/.test(name)) return 0.25;
    if (action === "wall_ball" && ["top", "finish"].includes(phase) && /_(elbow|wrist)$/.test(name)) {
      return 0.25;
    }
    if (
      ["rowing", "skierg"].includes(action)
      && ["catch", "finish", "top", "bottom"].includes(phase)
      && !FACE_NAMES.has(name)
    ) {
      return 0.35;
    }
    if (
      action === "burpee_broad_jump"
      && ["landing", "landed", "contact"].includes(phase)
      && FOOT_NAMES.has(name)
    ) {
      return 0.10;
    }
    return 1;
  }

  #bodyScale() {
    const confident = this.points.filter(point => pointConfidence(point) >= this.config.minimum_visibility);
    if (confident.length < 2) return 1;
    const xs = confident.map(point => point.x);
    const ys = confident.map(point => point.y);
    return Math.max(1e-3, Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  }

  #copyCurrentToOutput() {
    this.outputPoints.length = this.points.length;
    for (let index = 0; index < this.points.length; index += 1) {
      const output = this.outputPoints[index] || {};
      Object.assign(output, this.points[index]);
      this.outputPoints[index] = output;
    }
  }

  #result(horizonMs, predictedPointCount, clampedPointCount) {
    this.lastSummary.landmarks = this.outputPoints;
    this.lastSummary.enabled = this.config.enabled;
    this.lastSummary.applied = predictedPointCount > 0;
    this.lastSummary.horizonMs = horizonMs;
    this.lastSummary.predictedPointCount = predictedPointCount;
    this.lastSummary.clampedPointCount = clampedPointCount;
    return this.lastSummary;
  }
}
