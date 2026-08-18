import {
  DEFAULT_LANDMARK_QUALITY_GATE,
  LandmarkQualityGate,
} from "./landmark_quality_gate.mjs";

export const DEFAULT_UPLOAD_POSE_STABILIZATION = Object.freeze({
  reliable_confidence: 0.50,
  interpolation_gap_ms: 420,
  edge_hold_ms: 220,
  smoothing_radius: 2,
  smooth_blend: 0.62,
  maximum_correction: 0.035,
  smoothing_mode: "bidirectional",
  quality_gate_enabled: true,
  short_prediction_ms: 200,
  hide_after_ms: 500,
});

function finite(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, Number(value)));
}

function confidence(point) {
  return Math.min(
    clamp(finite(point?.visibility, point?.confidence ?? 0), 0, 1),
    clamp(finite(point?.presence, point?.visibility ?? point?.confidence ?? 0), 0, 1),
  );
}

function timestamp(frame, index) {
  return finite(frame?.timestamp_ms, index * (1000 / 30));
}

function pointMap(frame) {
  return new Map((frame?.keypoints || []).map(point => [String(point.name || ""), point]));
}

function lerpPoint(first, second, ratio) {
  const t = clamp(ratio, 0, 1);
  return {
    ...first,
    x: finite(first.x) + (finite(second.x) - finite(first.x)) * t,
    y: finite(first.y) + (finite(second.y) - finite(first.y)) * t,
    z: finite(first.z) + (finite(second.z) - finite(first.z)) * t,
    visibility: Math.max(confidence(first), confidence(second)),
    displayValid: true,
    displayInterpolated: true,
  };
}

function limitedCorrection(raw, target, maximum) {
  const dx = finite(target.x) - finite(raw.x);
  const dy = finite(target.y) - finite(raw.y);
  const distance = Math.hypot(dx, dy);
  if (distance <= maximum || distance <= 1e-12) return target;
  const scale = maximum / distance;
  return {
    ...target,
    x: finite(raw.x) + dx * scale,
    y: finite(raw.y) + dy * scale,
  };
}

function smoothReliablePoint(frames, maps, frameIndex, name, config) {
  const raw = maps[frameIndex].get(name);
  const samples = [];
  const firstIndex = config.smoothing_mode === "forward"
    ? Math.max(0, frameIndex - config.smoothing_radius)
    : Math.max(0, frameIndex - config.smoothing_radius);
  const lastIndex = config.smoothing_mode === "forward"
    ? frameIndex
    : Math.min(frames.length - 1, frameIndex + config.smoothing_radius);
  for (
    let index = firstIndex;
    index <= lastIndex;
    index += 1
  ) {
    const point = maps[index].get(name);
    if (!point || confidence(point) < config.reliable_confidence) continue;
    const distance = Math.abs(index - frameIndex);
    const weight = (config.smoothing_radius + 1 - distance) * Math.max(0.25, confidence(point));
    samples.push({ point, weight });
  }
  if (samples.length < 2) return { ...raw, displayValid: true };
  const total = samples.reduce((sum, sample) => sum + sample.weight, 0);
  const center = {
    ...raw,
    x: samples.reduce((sum, sample) => sum + finite(sample.point.x) * sample.weight, 0) / total,
    y: samples.reduce((sum, sample) => sum + finite(sample.point.y) * sample.weight, 0) / total,
    z: samples.reduce((sum, sample) => sum + finite(sample.point.z) * sample.weight, 0) / total,
  };
  const blended = {
    ...raw,
    x: finite(raw.x) * (1 - config.smooth_blend) + center.x * config.smooth_blend,
    y: finite(raw.y) * (1 - config.smooth_blend) + center.y * config.smooth_blend,
    z: finite(raw.z) * (1 - config.smooth_blend) + center.z * config.smooth_blend,
    displayValid: true,
    displaySmoothed: true,
  };
  return limitedCorrection(raw, blended, config.maximum_correction);
}

/**
 * Build a display-only upload timeline. Formal phases, rules, and report
 * keypoints remain untouched; only video playback consumes this result.
 */
export function stabilizeUploadTimeline(inputFrames, options = {}) {
  const frames = Array.isArray(inputFrames) ? inputFrames : [];
  if (!frames.length) return [];
  const config = {
    ...DEFAULT_UPLOAD_POSE_STABILIZATION,
    ...(options && typeof options === "object" ? options : {}),
  };
  config.reliable_confidence = clamp(config.reliable_confidence, 0, 1);
  config.smoothing_radius = Math.max(0, Math.min(5, Math.trunc(config.smoothing_radius)));
  config.smooth_blend = clamp(config.smooth_blend, 0, 1);
  config.maximum_correction = Math.max(0, finite(config.maximum_correction, 0.035));
  config.smoothing_mode = config.smoothing_mode === "forward" ? "forward" : "bidirectional";
  config.short_prediction_ms = Math.max(0, finite(config.short_prediction_ms, 200));
  config.hide_after_ms = Math.max(config.short_prediction_ms, finite(config.hide_after_ms, 500));
  const times = frames.map(timestamp);
  const qualityGate = new LandmarkQualityGate({
    ...DEFAULT_LANDMARK_QUALITY_GATE,
    enabled: config.quality_gate_enabled !== false,
    minimum_confidence: Math.min(config.reliable_confidence, 0.35),
    max_gap_ms_before_reset: Math.max(config.hide_after_ms, 500),
  });
  const decisions = frames.map((frame, index) => {
    const points = Array.isArray(frame?.keypoints) ? frame.keypoints : [];
    const evaluated = qualityGate.evaluateFrame(
      points,
      times[index],
      points.map(point => String(point?.name || "")),
    );
    return new Map(points.map((point, pointIndex) => [String(point.name || ""), evaluated[pointIndex]]));
  });
  const rawMaps = frames.map(pointMap);
  const maps = rawMaps.map((rawMap, frameIndex) => new Map(
    [...rawMap.entries()].filter(([name, point]) => {
      const decision = decisions[frameIndex].get(name);
      return confidence(point) >= config.reliable_confidence && decision?.accepted !== false;
    }),
  ));

  return frames.map((frame, frameIndex) => {
    const names = [...rawMaps[frameIndex].keys()];
    const keypoints = names.map(name => {
      const raw = rawMaps[frameIndex].get(name);
      const decision = decisions[frameIndex].get(name) || {
        accepted: confidence(raw) >= config.reliable_confidence,
        quality: confidence(raw),
        reason_codes: [],
        occlusion_state: "visible",
      };
      if (maps[frameIndex].has(name)) {
        return {
          ...smoothReliablePoint(frames, maps, frameIndex, name, config),
          displayQuality: decision.quality,
          displayMeasurementAccepted: decision.accepted,
          displayReasonCodes: decision.reason_codes,
          occlusionState: decision.occlusion_state,
        };
      }

      let previous = null;
      let next = null;
      for (let index = frameIndex - 1; index >= 0; index -= 1) {
        const candidate = maps[index].get(name);
        if (candidate && confidence(candidate) >= config.reliable_confidence) {
          previous = { point: candidate, time: times[index] };
          break;
        }
      }
      for (let index = frameIndex + 1; index < frames.length; index += 1) {
        const candidate = maps[index].get(name);
        if (candidate && confidence(candidate) >= config.reliable_confidence) {
          next = { point: candidate, time: times[index] };
          break;
        }
      }
      const currentTime = times[frameIndex];
      if (
        previous
        && next
        && next.time - previous.time <= config.interpolation_gap_ms
      ) {
        return {
          ...lerpPoint(
            previous.point,
            next.point,
            (currentTime - previous.time) / Math.max(1, next.time - previous.time),
          ),
          displayQuality: decision.quality,
          displayMeasurementAccepted: decision.accepted,
          displayReasonCodes: decision.reason_codes,
          occlusionState: decision.occlusion_state,
        };
      }
      if (previous && currentTime - previous.time <= config.edge_hold_ms) {
        return {
          ...previous.point, displayValid: true, displayHeld: true,
          displayQuality: decision.quality,
          displayMeasurementAccepted: decision.accepted,
          displayReasonCodes: decision.reason_codes,
          occlusionState: decision.occlusion_state,
        };
      }
      if (next && next.time - currentTime <= config.edge_hold_ms) {
        return {
          ...next.point, displayValid: true, displayHeld: true,
          displayQuality: decision.quality,
          displayMeasurementAccepted: decision.accepted,
          displayReasonCodes: decision.reason_codes,
          occlusionState: decision.occlusion_state,
        };
      }
      const elapsed = previous ? currentTime - previous.time : Number.POSITIVE_INFINITY;
      if (previous && elapsed < config.hide_after_ms) {
        const priorPrevious = (() => {
          for (let index = frameIndex - 2; index >= 0; index -= 1) {
            const candidate = maps[index].get(name);
            if (candidate) return { point: candidate, time: times[index] };
          }
          return null;
        })();
        let predicted = { ...previous.point };
        if (priorPrevious && elapsed <= config.short_prediction_ms) {
          const historyMs = Math.max(1, previous.time - priorPrevious.time);
          predicted = {
            ...predicted,
            x: finite(previous.point.x) + (finite(previous.point.x) - finite(priorPrevious.point.x)) * elapsed / historyMs,
            y: finite(previous.point.y) + (finite(previous.point.y) - finite(priorPrevious.point.y)) * elapsed / historyMs,
            z: finite(previous.point.z) + (finite(previous.point.z) - finite(priorPrevious.point.z)) * elapsed / historyMs,
          };
        }
        const fade = Math.max(0.05, 1 - elapsed / config.hide_after_ms);
        return {
          ...predicted,
          visibility: confidence(previous.point) * fade,
          presence: confidence(previous.point) * fade,
          displayValid: true,
          displayPredicted: elapsed <= config.short_prediction_ms,
          displayQuality: decision.quality,
          displayMeasurementAccepted: false,
          displayReasonCodes: decision.reason_codes,
          occlusionState: decision.occlusion_state,
        };
      }
      return {
        ...raw,
        displayValid: false,
        displayQuality: decision.quality,
        displayMeasurementAccepted: false,
        displayReasonCodes: decision.reason_codes,
        occlusionState: decision.occlusion_state,
      };
    });
    return {
      ...frame,
      keypoints,
      display_stabilized: true,
      display_smoothing_mode: config.smoothing_mode,
    };
  });
}

function percentile(values, ratio) {
  const ordered = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!ordered.length) return 0;
  return ordered[Math.min(ordered.length - 1, Math.max(0, Math.ceil(ordered.length * ratio) - 1))];
}

export function measureUploadTimelineStability(inputFrames, outputFrames, options = {}) {
  const inputs = Array.isArray(inputFrames) ? inputFrames : [];
  const outputs = Array.isArray(outputFrames) ? outputFrames : [];
  const outputMaps = outputs.map(pointMap);
  const inputMaps = inputs.map(pointMap);
  const times = outputs.map(timestamp);
  const accelerations = [];
  const corrections = [];
  let hiddenPoints = 0;
  let predictedPoints = 0;
  let rejectedPoints = 0;
  let flickers = 0;
  const reasonCodeCounts = {};
  const names = new Set(outputMaps.flatMap(map => [...map.keys()]));
  for (const name of names) {
    let previous = null;
    let previousVelocity = null;
    let previousValid = null;
    for (let index = 0; index < outputMaps.length; index += 1) {
      const point = outputMaps[index].get(name);
      if (!point) continue;
      const valid = point.displayValid !== false;
      if (previousValid === false && valid) flickers += 1;
      previousValid = valid;
      hiddenPoints += Number(!valid);
      predictedPoints += Number(Boolean(point.displayPredicted || point.displayInterpolated || point.displayHeld));
      rejectedPoints += Number(point.displayMeasurementAccepted === false);
      for (const reason of point.displayReasonCodes || []) {
        reasonCodeCounts[reason] = Number(reasonCodeCounts[reason] || 0) + 1;
      }
      const raw = inputMaps[index]?.get(name);
      if (raw && valid) corrections.push(Math.hypot(finite(point.x) - finite(raw.x), finite(point.y) - finite(raw.y)));
      if (!valid || !previous) {
        previous = valid ? { point, time: times[index] } : null;
        previousVelocity = null;
        continue;
      }
      const dt = (times[index] - previous.time) / 1000;
      if (dt > 0) {
        const velocity = {
          x: (finite(point.x) - finite(previous.point.x)) / dt,
          y: (finite(point.y) - finite(previous.point.y)) / dt,
        };
        if (previousVelocity) {
          accelerations.push(Math.hypot(
            velocity.x - previousVelocity.x,
            velocity.y - previousVelocity.y,
          ) / dt);
        }
        previousVelocity = velocity;
      }
      previous = { point, time: times[index] };
    }
  }
  const intervals = times.slice(1).map((value, index) => value - times[index]).filter(value => value > 0);
  const radius = Math.max(0, Math.trunc(finite(options.smoothing_radius, DEFAULT_UPLOAD_POSE_STABILIZATION.smoothing_radius)));
  return {
    frame_count: outputs.length,
    landmark_count: [...names].length,
    normalized_acceleration_p95: percentile(accelerations, 0.95),
    correction_p95: percentile(corrections, 0.95),
    hidden_point_count: hiddenPoints,
    predicted_or_filled_point_count: predictedPoints,
    rejected_point_count: rejectedPoints,
    reason_code_counts: Object.fromEntries(
      Object.entries(reasonCodeCounts).sort(([left], [right]) => left.localeCompare(right)),
    ),
    reacquisition_or_flicker_count: flickers,
    algorithmic_lookahead_ms: options.smoothing_mode === "forward"
      ? 0
      : radius * percentile(intervals, 0.5),
  };
}

export function compareUploadStabilizationStrategies(inputFrames, options = {}) {
  const strategies = [
    {
      name: "legacy_bidirectional",
      config: {
        ...options,
        smoothing_mode: "bidirectional",
        quality_gate_enabled: false,
        short_prediction_ms: 0,
        hide_after_ms: finite(options.edge_hold_ms, DEFAULT_UPLOAD_POSE_STABILIZATION.edge_hold_ms),
      },
    },
    { name: "forward", config: { ...options, smoothing_mode: "forward" } },
    { name: "bidirectional", config: { ...options, smoothing_mode: "bidirectional" } },
  ];
  const results = strategies.map(item => {
    const config = item.config;
    const frames = stabilizeUploadTimeline(inputFrames, config);
    return {
      strategy: item.name,
      metrics: measureUploadTimelineStability(inputFrames, frames, config),
    };
  });
  const formalFields = ["phase", "reps", "candidate_count", "pose_valid_rep_count", "no_rep_count", "unsure_count"];
  const formalFieldsUnchanged = results.every(result => result.metrics.frame_count === inputFrames.length)
    && results.every(result => {
      const strategy = strategies.find(item => item.name === result.strategy);
      const frames = stabilizeUploadTimeline(inputFrames, strategy.config);
      return frames.every((frame, index) => formalFields.every(field => frame[field] === inputFrames[index]?.[field]));
    });
  return {
    schema_version: 1,
    artifact_type: "display_stability_ablation",
    formal_fields_unchanged: formalFieldsUnchanged,
    strategies: results,
  };
}
