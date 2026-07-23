import {
  FilesetResolver,
  PoseLandmarker,
} from "../vendor/mediapipe/vision_bundle.mjs";
import { DisplayPoseFilter } from "./display_pose_filter.mjs";

let fileset = null;
let activeLandmarker = null;
let activeModel = "full";
let modelUrls = {};
let landmarkerOptions = {};
let landmarkers = new Map();
let lastTimestampMs = -1;
let switchingModel = false;
let benchmark = null;
const displayPoseFilter = new DisplayPoseFilter();

function resetFilters() {
  displayPoseFilter.reset();
}

function closeLandmarker(instance) {
  try { instance?.close?.(); } catch (_) { /* Ignore an already closed task. */ }
}

async function createLandmarker(model) {
  const url = modelUrls[model];
  if (!url) throw new Error(`Missing ${model} pose model URL`);
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: url },
    runningMode: "VIDEO",
    numPoses: 1,
    outputSegmentationMasks: false,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

async function ensureLandmarker(model) {
  if (!landmarkers.has(model)) landmarkerOptions[model] = createLandmarker(model);
  if (landmarkerOptions[model]) {
    try {
      const instance = await landmarkerOptions[model];
      landmarkers.set(model, instance);
    } finally {
      delete landmarkerOptions[model];
    }
  }
  return landmarkers.get(model);
}

function setActiveLandmarker(model) {
  const instance = landmarkers.get(model);
  if (!instance) throw new Error(`${model} pose model is not loaded`);
  activeModel = model;
  activeLandmarker = instance;
  resetFilters();
}

function releaseInactiveLandmarkers() {
  for (const [model, instance] of landmarkers.entries()) {
    if (model === activeModel) continue;
    closeLandmarker(instance);
    landmarkers.delete(model);
  }
}

function percentile(values, ratio) {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const index = Math.min(ordered.length - 1, Math.max(0, Math.ceil(ratio * ordered.length) - 1));
  return ordered[index];
}

function phaseStats(phase) {
  const samples = benchmark.samples[phase];
  const firstAt = benchmark.firstAt[phase];
  const lastAt = benchmark.lastAt[phase];
  const elapsedMs = Math.max(1, lastAt - firstAt);
  return {
    samples: samples.length,
    inferenceP50Ms: percentile(samples, 0.50),
    inferenceP95Ms: percentile(samples, 0.95),
    poseFps: Math.max(0, samples.length - 1) * 1000 / elapsedMs,
    detectionRate: samples.length ? benchmark.detected[phase] / samples.length : 0,
  };
}

function selectAutoModel(stats, liteAutoApproved) {
  const full = stats.full;
  const lite = stats.lite;
  const fullStable = full.inferenceP95Ms - full.inferenceP50Ms <= 12
    && full.inferenceP95Ms <= Math.max(20, full.inferenceP50Ms * 1.8);
  if (full.inferenceP95Ms <= 20) {
    return { model: "full", reason: "Full P95 不超过 20 ms" };
  }
  if (full.inferenceP95Ms <= 33 && fullStable) {
    return { model: "full", reason: "Full P95 不超过 33 ms 且运行稳定" };
  }
  const liteDetectionAcceptable = lite.detectionRate + 0.15 >= full.detectionRate;
  if (liteAutoApproved && lite.inferenceP95Ms < full.inferenceP95Ms && liteDetectionAcceptable) {
    return { model: "lite", reason: "Full P95 超过实时预算，Lite 更快且检出率稳定" };
  }
  return { model: "full", reason: liteAutoApproved
    ? "Lite 未在当前画面表现出更好的速度与检出率"
    : "Lite 尚未通过产品精度回归门" };
}

function startBenchmark(durationMs, liteAutoApproved) {
  benchmark = {
    active: true,
    phase: "full",
    phaseDurationMs: Math.max(1000, Number(durationMs || 3000) / 2),
    liteAutoApproved: Boolean(liteAutoApproved),
    samples: { full: [], lite: [] },
    detected: { full: 0, lite: 0 },
    firstAt: { full: 0, lite: 0 },
    lastAt: { full: 0, lite: 0 },
  };
}

function configureDisplaySmoothing(value) {
  displayPoseFilter.configure(value);
}

function resetActiveBenchmark() {
  if (!benchmark?.active) return;
  benchmark.phase = "full";
  benchmark.samples = { full: [], lite: [] };
  benchmark.detected = { full: 0, lite: 0 };
  benchmark.firstAt = { full: 0, lite: 0 };
  benchmark.lastAt = { full: 0, lite: 0 };
  setActiveLandmarker("full");
}

function recordBenchmarkSample(model, inferenceMs, detected, now) {
  if (!benchmark?.active || benchmark.phase !== model) return;
  if (!benchmark.firstAt[model]) benchmark.firstAt[model] = now;
  benchmark.lastAt[model] = now;
  benchmark.samples[model].push(inferenceMs);
  benchmark.detected[model] += Number(detected);
  const elapsed = now - benchmark.firstAt[model];
  if (elapsed < benchmark.phaseDurationMs || benchmark.samples[model].length < 4) return;
  if (model === "full") {
    benchmark.phase = "lite";
    setActiveLandmarker("lite");
    self.postMessage({ type: "benchmark_progress", completedModel: "full", stats: phaseStats("full"), nextModel: "lite" });
    return;
  }
  const stats = { full: phaseStats("full"), lite: phaseStats("lite") };
  const selection = selectAutoModel(stats, benchmark.liteAutoApproved);
  benchmark.active = false;
  setActiveLandmarker(selection.model);
  releaseInactiveLandmarkers();
  self.postMessage({
    type: "benchmark_complete",
    selectedModel: selection.model,
    reason: selection.reason,
    stats,
  });
}

async function initialize(message) {
  fileset = await FilesetResolver.forVisionTasks(message.wasmRoot);
  configureDisplaySmoothing(message.displaySmoothing);
  modelUrls = { ...message.modelUrls };
  const preference = ["auto", "lite", "full"].includes(message.modelPreference)
    ? message.modelPreference
    : "auto";
  if (preference === "auto") {
    await ensureLandmarker("full");
    await ensureLandmarker("lite");
    setActiveLandmarker("full");
    startBenchmark(message.benchmarkDurationMs, message.liteAutoApproved);
  } else {
    await ensureLandmarker(preference);
    setActiveLandmarker(preference);
  }
  self.postMessage({
    type: "ready",
    backend: "browser-mediapipe",
    model: activeModel,
    modelPreference: preference,
    benchmarking: Boolean(benchmark?.active),
  });
}

async function switchModel(message) {
  const model = message.model;
  if (!['lite', 'full'].includes(model)) throw new Error("Invalid pose model tier");
  switchingModel = true;
  try {
    await ensureLandmarker(model);
    setActiveLandmarker(model);
    releaseInactiveLandmarkers();
    self.postMessage({ type: "model_changed", model, reason: message.reason || "" });
  } finally {
    switchingModel = false;
  }
}

function closeInput(image) {
  try { image?.close?.(); } catch (_) { /* Transferred frame may already be closed. */ }
}

function normalizeInput(image, width, height) {
  if (typeof VideoFrame !== "undefined" && image instanceof VideoFrame) {
    if (typeof OffscreenCanvas === "undefined") throw new Error("OffscreenCanvas is required for VideoFrame input");
    const canvas = new OffscreenCanvas(Math.max(1, width), Math.max(1, height));
    canvas.getContext("2d", { alpha: false }).drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas;
  }
  return image;
}

function processFrame(message) {
  if (!activeLandmarker) throw new Error("Pose worker is not initialized");
  if (switchingModel) {
    closeInput(message.image);
    self.postMessage({ type: "frame_skipped", frameId: message.frameMeta?.frameId, reason: "model_switch" });
    return;
  }
  const timestampMs = Math.max(lastTimestampMs + 1, Math.round(Number(message.frameMeta.presentationTime)));
  lastTimestampMs = timestampMs;
  const model = activeModel;
  const wasBenchmarking = Boolean(benchmark?.active);
  const started = performance.now();
  const input = normalizeInput(message.image, message.frameMeta.width, message.frameMeta.height);
  try {
    const result = activeLandmarker.detectForVideo(input, timestampMs);
    const inferenceMs = performance.now() - started;
    const rawImageLandmarks = result.landmarks?.[0] || [];
    const rawWorldLandmarks = result.worldLandmarks?.[0] || [];
    const imageDisplay = displayPoseFilter.applyImage(rawImageLandmarks, timestampMs);
    const worldDisplay = displayPoseFilter.applyWorld(rawWorldLandmarks, timestampMs);
    self.postMessage({
      type: "result",
      frameMeta: message.frameMeta,
      imageLandmarks: imageDisplay.landmarks,
      worldLandmarks: worldDisplay.landmarks,
      rawImageLandmarks,
      rawWorldLandmarks,
      displayFilter: displayPoseFilter.summary(imageDisplay, worldDisplay),
      poseInferenceMs: inferenceMs,
      workerResultTimeMs: performance.now(),
      transferMode: message.transferMode,
      copyStartMs: message.copyStartMs,
      copyEndMs: message.copyEndMs,
      model,
      benchmarking: wasBenchmarking,
    });
    recordBenchmarkSample(model, inferenceMs, rawImageLandmarks.length > 0, performance.now());
  } finally {
    closeInput(message.image);
  }
}

self.onmessage = event => {
  const message = event.data || {};
  if (message.type === "init") {
    initialize(message).catch(error => {
      self.postMessage({ type: "init_error", message: error?.message || String(error) });
    });
    return;
  }
  if (message.type === "reset") {
    resetFilters();
    lastTimestampMs = -1;
    resetActiveBenchmark();
    return;
  }
  if (message.type === "switch_model") {
    switchModel(message).catch(error => {
      self.postMessage({ type: "model_change_error", message: error?.message || String(error) });
    });
    return;
  }
  if (message.type === "frame") {
    try {
      processFrame(message);
    } catch (error) {
      closeInput(message.image);
      self.postMessage({
        type: "frame_error",
        frameId: message.frameMeta?.frameId,
        message: error?.message || String(error),
      });
    }
  }
};
