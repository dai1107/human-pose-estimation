export const OCCLUSION_STATES = Object.freeze({
  VISIBLE: "VISIBLE",
  SUSPECTED: "SUSPECTED",
  OCCLUDED: "OCCLUDED",
  REACQUIRING: "REACQUIRING",
});

export const MEDIAPIPE_LANDMARK_NAMES = Object.freeze([
  "nose", "left_eye_inner", "left_eye", "left_eye_outer",
  "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
  "right_ear", "mouth_left", "mouth_right", "left_shoulder",
  "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
  "right_wrist", "left_pinky", "right_pinky", "left_index",
  "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
  "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
  "right_heel", "left_foot_index", "right_foot_index",
]);

export const DEFAULT_LANDMARK_QUALITY_GATE = Object.freeze({
  enabled: true,
  minimum_confidence: 0.35,
  max_speed_body_s: 18.0,
  max_acceleration_body_s2: 180.0,
  max_bone_length_change_ratio: 0.55,
  identity_swap_margin: 0.12,
  neighbor_motion_ratio: 10.0,
  occluded_after_rejections: 2,
  reacquire_frames: 3,
  max_gap_ms_before_reset: 500,
  bone_history_size: 21,
});

const PARENT_BY_NAME = Object.freeze({
  left_elbow: "left_shoulder", right_elbow: "right_shoulder",
  left_wrist: "left_elbow", right_wrist: "right_elbow",
  left_pinky: "left_wrist", right_pinky: "right_wrist",
  left_index: "left_wrist", right_index: "right_wrist",
  left_thumb: "left_wrist", right_thumb: "right_wrist",
  left_knee: "left_hip", right_knee: "right_hip",
  left_ankle: "left_knee", right_ankle: "right_knee",
  left_heel: "left_ankle", right_heel: "right_ankle",
  left_foot_index: "left_ankle", right_foot_index: "right_ankle",
});

function finite(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp(value, low = 0, high = 1) {
  return Math.max(low, Math.min(high, finite(value)));
}

function pointConfidence(point) {
  return Math.min(
    clamp(point?.visibility ?? point?.confidence ?? 0),
    clamp(point?.presence ?? point?.visibility ?? point?.confidence ?? 0),
  );
}

function usable(point) {
  return point
    && Number.isFinite(Number(point.x))
    && Number.isFinite(Number(point.y))
    && Number.isFinite(Number(point.z ?? 0));
}

function distance(first, second) {
  if (!usable(first) || !usable(second)) return Number.NaN;
  return Math.hypot(
    finite(first.x) - finite(second.x),
    finite(first.y) - finite(second.y),
    finite(first.z) - finite(second.z),
  );
}

function median(values) {
  const ordered = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!ordered.length) return Number.NaN;
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
}

function sideMate(name) {
  if (name.startsWith("left_")) return `right_${name.slice(5)}`;
  if (name.startsWith("right_")) return `left_${name.slice(6)}`;
  return "";
}

function normalizedConfig(value = {}) {
  const input = value && typeof value === "object" ? value : {};
  const number = (name, fallback, low, high = Number.POSITIVE_INFINITY) => {
    const parsed = Number(input[name]);
    return Number.isFinite(parsed) ? Math.max(low, Math.min(high, parsed)) : fallback;
  };
  return {
    enabled: input.enabled !== false,
    minimum_confidence: number("minimum_confidence", 0.35, 0, 1),
    max_speed_body_s: number("max_speed_body_s", 18, 0.1),
    max_acceleration_body_s2: number("max_acceleration_body_s2", 180, 0.1),
    max_bone_length_change_ratio: number("max_bone_length_change_ratio", 0.55, 0.01, 3),
    identity_swap_margin: number("identity_swap_margin", 0.12, 0, 1),
    neighbor_motion_ratio: number("neighbor_motion_ratio", 10, 0.1),
    occluded_after_rejections: Math.max(1, Math.trunc(number("occluded_after_rejections", 2, 1, 8))),
    reacquire_frames: Math.max(1, Math.trunc(number("reacquire_frames", 3, 1, 12))),
    max_gap_ms_before_reset: number("max_gap_ms_before_reset", 500, 1),
    bone_history_size: Math.max(3, Math.trunc(number("bone_history_size", 21, 3, 101))),
  };
}

/**
 * Display-only per-landmark measurement gate. It rejects implausible model
 * observations before smoothing and reports why; it never fabricates formal
 * analysis evidence.
 */
export class LandmarkQualityGate {
  constructor(config = {}) {
    this.configure(config);
  }

  configure(config = {}) {
    this.config = normalizedConfig({ ...DEFAULT_LANDMARK_QUALITY_GATE, ...config });
    this.reset();
  }

  reset() {
    this.tracks = new Map();
    this.boneHistory = new Map();
    this.lastBodyScale = 0.25;
    this.lastTimestampMs = null;
  }

  evaluateFrame(points, timestampMs, names = null) {
    const landmarks = Array.isArray(points) ? points : [];
    const resolvedTimestamp = finite(timestampMs, this.lastTimestampMs ?? 0);
    if (
      this.lastTimestampMs !== null
      && (resolvedTimestamp <= this.lastTimestampMs
        || resolvedTimestamp - this.lastTimestampMs >= this.config.max_gap_ms_before_reset)
    ) {
      this.reset();
    }
    this.lastTimestampMs = resolvedTimestamp;
    const resolvedNames = landmarks.map((point, index) => String(
      point?.name || names?.[index] || MEDIAPIPE_LANDMARK_NAMES[index] || index,
    ));
    const byName = new Map(resolvedNames.map((name, index) => [name, landmarks[index]]));
    const bodyScale = this.#bodyScale(byName);
    const preliminary = landmarks.map((point, index) => this.#measure(
      point,
      resolvedNames[index],
      resolvedTimestamp,
      bodyScale,
      byName,
    ));

    // A swap is only declared when both members cross toward the other side's
    // previous position. This avoids rejecting legitimate frontal crossings.
    const measuredByName = new Map(preliminary.map(item => [item.name, item]));
    for (const item of preliminary) {
      const mateName = sideMate(item.name);
      const mate = measuredByName.get(mateName);
      const ownTrack = this.tracks.get(item.name);
      const mateTrack = this.tracks.get(mateName);
      if (!mate || !ownTrack?.lastAccepted || !mateTrack?.lastAccepted) continue;
      const ownSame = distance(item.point, ownTrack.lastAccepted);
      const ownCross = distance(item.point, mateTrack.lastAccepted);
      const mateSame = distance(mate.point, mateTrack.lastAccepted);
      const mateCross = distance(mate.point, ownTrack.lastAccepted);
      const margin = this.config.identity_swap_margin * bodyScale;
      if (ownCross + margin < ownSame && mateCross + margin < mateSame) {
        item.reasons.add("left_right_identity_swap");
      }
    }

    return preliminary.map(item => this.#transition(item, resolvedTimestamp));
  }

  #bodyScale(byName) {
    const segments = [
      ["left_shoulder", "right_shoulder"],
      ["left_hip", "right_hip"],
      ["left_shoulder", "left_hip"],
      ["right_shoulder", "right_hip"],
    ].map(([first, second]) => distance(byName.get(first), byName.get(second)))
      .filter(value => Number.isFinite(value) && value > 0.01);
    const estimate = median(segments);
    if (Number.isFinite(estimate)) {
      this.lastBodyScale = clamp(estimate, 0.06, 1.5);
    }
    return this.lastBodyScale;
  }

  #measure(point, name, timestampMs, bodyScale, byName) {
    const reasons = new Set();
    const confidence = pointConfidence(point);
    if (!usable(point)) reasons.add("non_finite_measurement");
    if (confidence < this.config.minimum_confidence) reasons.add("low_confidence");
    const track = this.tracks.get(name);
    let speed = 0;
    let acceleration = 0;
    let velocity = { x: 0, y: 0, z: 0 };
    if (usable(point) && track?.lastAccepted) {
      const elapsedMs = timestampMs - track.lastAcceptedTimestampMs;
      if (elapsedMs > 0 && elapsedMs <= this.config.max_gap_ms_before_reset) {
        const inverseSeconds = 1000 / elapsedMs;
        velocity = {
          x: (finite(point.x) - finite(track.lastAccepted.x)) * inverseSeconds / bodyScale,
          y: (finite(point.y) - finite(track.lastAccepted.y)) * inverseSeconds / bodyScale,
          z: (finite(point.z) - finite(track.lastAccepted.z)) * inverseSeconds / bodyScale,
        };
        speed = Math.hypot(velocity.x, velocity.y, velocity.z);
        if (track.lastVelocity && track.acceptedCount >= 2) {
          acceleration = Math.hypot(
            velocity.x - track.lastVelocity.x,
            velocity.y - track.lastVelocity.y,
            velocity.z - track.lastVelocity.z,
          ) * inverseSeconds;
        }
      }
    }
    if (speed > this.config.max_speed_body_s) reasons.add("velocity_outlier");
    if (acceleration > this.config.max_acceleration_body_s2) reasons.add("acceleration_outlier");

    const parentName = PARENT_BY_NAME[name];
    const parent = parentName ? byName.get(parentName) : null;
    const boneLength = distance(point, parent);
    const history = this.boneHistory.get(name) || [];
    const referenceBoneLength = median(history);
    const boneChangeRatio = Number.isFinite(referenceBoneLength) && referenceBoneLength > 1e-6
      ? Math.abs(boneLength - referenceBoneLength) / referenceBoneLength
      : 0;
    if (
      history.length >= 3
      && Number.isFinite(boneLength)
      && boneChangeRatio > this.config.max_bone_length_change_ratio
    ) reasons.add("bone_length_inconsistent");

    if (parentName && track?.lastAccepted) {
      const parentTrack = this.tracks.get(parentName);
      if (parentTrack?.lastAccepted) {
        const jointMotion = distance(point, track.lastAccepted) / bodyScale;
        const parentMotion = distance(parent, parentTrack.lastAccepted) / bodyScale;
        if (
          Number.isFinite(jointMotion)
          && Number.isFinite(parentMotion)
          && jointMotion > 0.08
          && jointMotion > Math.max(0.02, parentMotion) * this.config.neighbor_motion_ratio
        ) reasons.add("neighbor_motion_inconsistent");
      }
    }

    let quality = confidence;
    if (reasons.has("velocity_outlier")) quality *= 0.35;
    if (reasons.has("acceleration_outlier")) quality *= 0.55;
    if (reasons.has("bone_length_inconsistent")) quality *= 0.35;
    if (reasons.has("neighbor_motion_inconsistent")) quality *= 0.50;
    return {
      point, name, confidence, quality: clamp(quality), reasons,
      velocity, speed, acceleration, boneLength, boneChangeRatio,
    };
  }

  #transition(item, timestampMs) {
    const track = this.tracks.get(item.name) || {
      state: OCCLUSION_STATES.VISIBLE,
      rejectionCount: 0,
      reacquireCount: 0,
      lastAccepted: null,
      lastAcceptedTimestampMs: null,
      lastVelocity: null,
      acceptedCount: 0,
    };
    const temporalOutlier = item.reasons.has("velocity_outlier")
      || (item.reasons.has("acceleration_outlier")
        && (item.reasons.has("bone_length_inconsistent")
          || item.reasons.has("neighbor_motion_inconsistent")));
    const structuralConsensus = item.reasons.has("bone_length_inconsistent")
      && item.reasons.has("neighbor_motion_inconsistent");
    const identityOrVisibilityFailure = item.reasons.has("non_finite_measurement")
      || item.reasons.has("low_confidence")
      || item.reasons.has("left_right_identity_swap");
    const acceptedMeasurement = this.config.enabled
      ? !(temporalOutlier || structuralConsensus || identityOrVisibilityFailure)
      : usable(item.point);
    if (!acceptedMeasurement) {
      track.reacquireCount = 0;
      track.rejectionCount += 1;
      track.state = track.rejectionCount >= this.config.occluded_after_rejections
        ? OCCLUSION_STATES.OCCLUDED
        : OCCLUSION_STATES.SUSPECTED;
    } else if (track.state === OCCLUSION_STATES.OCCLUDED) {
      track.rejectionCount = 0;
      track.reacquireCount = 1;
      track.state = OCCLUSION_STATES.REACQUIRING;
    } else if (track.state === OCCLUSION_STATES.REACQUIRING) {
      track.reacquireCount += 1;
      if (track.reacquireCount >= this.config.reacquire_frames) {
        track.state = OCCLUSION_STATES.VISIBLE;
        track.reacquireCount = 0;
      }
    } else {
      track.rejectionCount = 0;
      track.reacquireCount = 0;
      track.state = OCCLUSION_STATES.VISIBLE;
    }

    if (acceptedMeasurement) {
      track.lastAccepted = { ...item.point };
      track.lastAcceptedTimestampMs = timestampMs;
      track.lastVelocity = item.velocity;
      track.acceptedCount += 1;
      if (Number.isFinite(item.boneLength) && item.boneLength > 1e-6) {
        const history = this.boneHistory.get(item.name) || [];
        history.push(item.boneLength);
        if (history.length > this.config.bone_history_size) history.shift();
        this.boneHistory.set(item.name, history);
      }
    }
    this.tracks.set(item.name, track);
    return {
      accepted: acceptedMeasurement,
      quality: item.quality,
      reason_codes: [...item.reasons].sort(),
      occlusion_state: track.state.toLowerCase(),
      reacquire_blend: track.state === OCCLUSION_STATES.REACQUIRING
        ? clamp(track.reacquireCount / this.config.reacquire_frames, 0.15, 1)
        : 1,
      speed_body_s: item.speed,
      acceleration_body_s2: item.acceleration,
      bone_length: item.boneLength,
      bone_length_change_ratio: item.boneChangeRatio,
      elapsed_since_accepted_ms: track.lastAcceptedTimestampMs === null
        ? Number.POSITIVE_INFINITY
        : Math.max(0, timestampMs - track.lastAcceptedTimestampMs),
    };
  }
}
