import { DisplayPosePredictor } from "./workers/display_pose_predictor.mjs";
import {
  DomUpdateScheduler,
  RenderPerformanceMonitor,
} from "./workers/render_performance.mjs";
import { CameraDiagnostics } from "./workers/camera_diagnostics.mjs";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const ui = {
  sourceMode: "camera",
  uploadId: "",
  running: false,
  paused: false,
  stateTimer: null,
  lastStatus: "idle",
  csrfToken: "",
  realtimeConfig: {
    frame_width: 640,
    frame_height: 480,
    inference_long_edge: 640,
    jpeg_quality: 0.65,
    target_fps: 30,
    camera_fps: 60,
  },
  mediaStream: null,
  socket: null,
  requestTimeout: null,
  drawAnimation: null,
  reconnectTimer: null,
  reconnectAttempts: 0,
  manualStop: false,
  facingMode: "user",
  sequence: 0,
  pendingFrames: new Map(),
  requestInFlight: false,
  inFlightFrameId: -1,
  activeRealtimeSessionId: "",
  activeRealtimeRunId: "",
  lastRenderedPoseFrameId: -1,
  lastDiscardedFrameId: -1,
  staleResultCount: 0,
  latestResult: null,
  lastResultAt: 0,
  lastFrameSentAt: 0,
  captureFps: 30,
  samplesByAction: new Map(),
  standards: {},
  officialRules: {},
  recorder: null,
  recordingCanvas: null,
  recordingAnimation: null,
  recordingChunks: [],
  recordingUrl: "",
  voiceEnabled: true,
  voiceSupported: "speechSynthesis" in window && "SpeechSynthesisUtterance" in window,
  lastVoiceEventId: "",
  manualFloorPoints: [],
  floorCalibrationActive: false,
  videoFrameCallbackId: null,
  latestVideoFrameMeta: null,
  lastPresentedFrames: 0,
  presentedFrameSkipCount: 0,
  rvfcLateFrameCount: 0,
  rvfcLatenessMs: 0,
  longTaskCount: 0,
  longTaskDurationMs: 0,
  longTaskObserver: null,
  lastAuditedPoseFrameId: -1,
  lastFallbackMediaTime: -1,
  videoFrameIntervalMs: 1000 / 60,
  poseRuntimeMode: "initializing",
  poseWorker: null,
  poseWorkerReadyPromise: null,
  poseWorkerBusy: false,
  poseWorkerPending: null,
  poseWorkerFrameCreationBusy: false,
  poseWorkerRequestedMeta: null,
  poseWorkerLastCaptureAt: 0,
  poseWorkerFailureCount: 0,
  poseWorkerSlowFrameCount: 0,
  poseWorkerDroppedFrames: 0,
  poseWorkerTransferMode: "",
  poseWorkerActiveModel: "full",
  poseWorkerModelPreference: "auto",
  poseWorkerBenchmarking: false,
  poseWorkerBenchmarkReport: null,
  poseWorkerBenchmarkSent: false,
  poseWorkerBenchmarkLongTaskBaseline: 0,
  poseWorkerModelSwitching: false,
  latestAnalysisResult: null,
  lastFeedbackSignature: "",
};

const displayPosePredictor = new DisplayPosePredictor();
const domUpdateScheduler = new DomUpdateScheduler();
const renderPerformance = new RenderPerformanceMonitor();
const cameraDiagnostics = new CameraDiagnostics();
const domNodeCache = new Map();

const poseLandmarkNames = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
  "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
  "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
  "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
  "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
];
const fingerLandmarkSuffixes = [
  "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
  "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
  "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
  "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
  "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
];
const supplementalFingerLandmarkNames = ["left", "right"].flatMap(side =>
  fingerLandmarkSuffixes.map(suffix => `${side}_hand_${suffix}`),
);
const renderLandmarkNames = [...poseLandmarkNames, ...supplementalFingerLandmarkNames];
const poseConnectionIndexes = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[15,17],
  [15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],
  [23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32],
];
const poseConnectionNames = poseConnectionIndexes.map(
  ([start, end]) => [poseLandmarkNames[start], poseLandmarkNames[end]],
);
const poseNameToIndex = new Map(renderLandmarkNames.map((name, index) => [name, index]));
const drawingCache = {
  pointPresent: new Uint8Array(renderLandmarkNames.length),
  pointX: new Float32Array(renderLandmarkNames.length),
  pointY: new Float32Array(renderLandmarkNames.length),
  pointVisibility: new Float32Array(renderLandmarkNames.length),
  connectionSource: null,
  connectionPairs: new Int16Array(renderLandmarkNames.length * 2),
  connectionCount: 0,
  rectKey: "",
  rect: { drawWidth: 0, drawHeight: 0, offsetX: 0, offsetY: 0 },
  styleKey: "",
  style: { lineWidth: 2, pointRadius: 3, font: "11px Inter, sans-serif" },
  angleEntries: [],
  angleCount: 0,
  predictionContext: { action: "", phase: "", supportFootNames: [] },
  performanceSnapshot: {
    renderLoopP95Ms: 0,
    canvasDrawP95Ms: 0,
    domUpdateP95Ms: 0,
    mainThreadLongTaskCount: 0,
    mainThreadLongTaskTotalMs: 0,
    longTaskPhases: {},
  },
};
const cameraDiagnosticCache = {
  canvas: null,
  context: null,
  previousLuma: new Uint8Array(32 * 18),
  hasPreviousLuma: false,
  lastSampleAt: 0,
  lastReportAt: 0,
};

function cachedNode(selector) {
  if (!domNodeCache.has(selector)) domNodeCache.set(selector, document.querySelector(selector));
  return domNodeCache.get(selector);
}

function setTextIfChanged(selector, value) {
  const node = cachedNode(selector);
  const text = String(value);
  if (node && node.textContent !== text) node.textContent = text;
}

function setClassIfChanged(selector, value) {
  const node = cachedNode(selector);
  if (node && node.className !== value) node.className = value;
}

function setHiddenIfChanged(selector, hidden) {
  const node = cachedNode(selector);
  if (node && node.hidden !== Boolean(hidden)) node.hidden = Boolean(hidden);
}

function setWidthIfChanged(selector, value) {
  const node = cachedNode(selector);
  const width = String(value);
  if (node && node.style.width !== width) node.style.width = width;
}

function finiteNumber(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function deriveWebLatencies(timing) {
  const delta = (end, start, signed = false) => {
    const endValue = finiteNumber(timing[end]);
    const startValue = finiteNumber(timing[start]);
    if (endValue === null || startValue === null) return null;
    const value = endValue - startValue;
    return Math.round((signed ? value : Math.max(0, value)) * 1000) / 1000;
  };
  return {
    capture_to_submit_ms: delta("socket_send_ms", "camera_frame_presented_ms"),
    submit_to_result_ms: delta("client_result_receive_ms", "socket_send_ms"),
    result_to_render_ms: delta("pose_render_start_ms", "client_result_receive_ms"),
    render_to_expected_display_ms: delta("expected_display_time_ms", "pose_render_end_ms", true),
    pose_age_at_render_ms: delta("pose_render_start_ms", "camera_frame_presented_ms"),
    video_frame_age_at_render_ms: delta("pose_render_start_ms", "video_frame_presented_at_render_ms"),
    pose_video_age_difference_ms: delta("video_frame_presented_at_render_ms", "camera_frame_presented_ms"),
    render_time_ms: delta("pose_render_end_ms", "pose_render_start_ms"),
    render_loop_p95_ms: finiteNumber(timing.render_loop_p95_ms),
    canvas_draw_p95_ms: finiteNumber(timing.canvas_draw_p95_ms),
    dom_update_p95_ms: finiteNumber(timing.dom_update_p95_ms),
    main_thread_long_task_count: finiteNumber(timing.main_thread_long_task_count),
    main_thread_long_task_total_ms: finiteNumber(timing.main_thread_long_task_duration_ms),
    long_task_render_count: finiteNumber(timing.long_task_render_count),
    long_task_dom_update_count: finiteNumber(timing.long_task_dom_update_count),
    long_task_frame_copy_count: finiteNumber(timing.long_task_frame_copy_count),
    long_task_encode_count: finiteNumber(timing.long_task_encode_count),
    long_task_pose_transfer_count: finiteNumber(timing.long_task_pose_transfer_count),
    long_task_other_count: finiteNumber(timing.long_task_other_count),
  };
}

function timingForVideoFrame(frameMeta) {
  return {
    camera_frame_presented_ms: frameMeta.presentationTime,
    expected_display_time_ms: frameMeta.expectedDisplayTime,
    capture_time_ms: frameMeta.captureTime,
    media_time_s: frameMeta.mediaTime,
    presented_frames: frameMeta.presentedFrames,
  };
}

function buildVideoFrameMeta(now, metadata = {}, fallback = false) {
  const video = $("#localVideo");
  const presentedFrames = finiteNumber(metadata.presentedFrames, ui.lastPresentedFrames + 1);
  if (ui.lastPresentedFrames > 0 && presentedFrames > ui.lastPresentedFrames + 1) {
    ui.presentedFrameSkipCount += presentedFrames - ui.lastPresentedFrames - 1;
  }
  const presentationTime = finiteNumber(metadata.presentationTime, now);
  if (ui.latestVideoFrameMeta) {
    const interval = presentationTime - ui.latestVideoFrameMeta.presentationTime;
    if (interval > 0 && interval < 250) ui.videoFrameIntervalMs = ui.videoFrameIntervalMs * 0.8 + interval * 0.2;
  }
  ui.lastPresentedFrames = presentedFrames;
  const expectedDisplayTime = finiteNumber(metadata.expectedDisplayTime, now);
  const lateness = Math.max(0, performance.now() - expectedDisplayTime);
  if (lateness >= Math.max(1, ui.videoFrameIntervalMs)) ui.rvfcLateFrameCount += 1;
  ui.rvfcLatenessMs = lateness;
  const frameId = ++ui.sequence;
  return Object.freeze({
    sessionId: ui.activeRealtimeSessionId || "preview",
    frameId,
    presentedFrames,
    mediaTime: finiteNumber(metadata.mediaTime, video.currentTime),
    presentationTime,
    expectedDisplayTime,
    captureTime: finiteNumber(metadata.captureTime),
    processingDuration: finiteNumber(metadata.processingDuration, 0),
    width: Math.max(0, Number(video.videoWidth || 0)),
    height: Math.max(0, Number(video.videoHeight || 0)),
    callbackSource: fallback ? "requestAnimationFrame" : "requestVideoFrameCallback",
  });
}

function onPresentedVideoFrame(now, metadata = {}, fallback = false) {
  if (!ui.mediaStream) return;
  const frameMeta = buildVideoFrameMeta(now, metadata, fallback);
  ui.latestVideoFrameMeta = frameMeta;
  cameraDiagnostics.observeFrame(frameMeta.presentationTime);
  sampleCameraDiagnostics(now);
  if (!ui.running) return;
  renderPoseForVideoFrame(frameMeta, now);
  if (ui.poseRuntimeMode === "local") submitLocalPoseFrame(frameMeta);
  else if (ui.poseRuntimeMode === "server") void captureLatestFrame(frameMeta);
}

function ensureCameraDiagnosticCanvas() {
  if (cameraDiagnosticCache.canvas) return;
  cameraDiagnosticCache.canvas = typeof OffscreenCanvas === "function"
    ? new OffscreenCanvas(32, 18)
    : document.createElement("canvas");
  cameraDiagnosticCache.canvas.width = 32;
  cameraDiagnosticCache.canvas.height = 18;
  cameraDiagnosticCache.context = cameraDiagnosticCache.canvas.getContext(
    "2d",
    { alpha: false, willReadFrequently: true },
  );
}

function cameraWarningText(warnings) {
  const labels = {
    fps_below_requested: "实际帧率低于请求值",
    low_light: "画面偏暗，请增加照明",
    frame_interval_unstable: "摄像头帧间隔不稳定",
    duplicate_frames: "摄像头重复画面比例偏高",
  };
  return warnings.map(name => labels[name] || name).join("；");
}

function renderCameraDiagnostics(snapshot) {
  const settings = snapshot.settings || {};
  const actualFps = snapshot.actualPresentedFps || settings.frameRate || 0;
  const specification = settings.width && settings.height
    ? `${settings.width}×${settings.height}@${Number(actualFps).toFixed(1)}`
    : "正在测量摄像头";
  const warnings = cameraWarningText(snapshot.warnings || []);
  setTextIfChanged(
    "#cameraDiagnostic",
    warnings ? `${specification} · ${warnings}` : `${specification} · 光照与帧间隔正常`,
  );
  const node = cachedNode("#cameraDiagnostic");
  if (node) node.dataset.state = warnings ? "warning" : "healthy";
}

function sampleCameraDiagnostics(now) {
  const config = ui.realtimeConfig.camera || {};
  const interval = 1000 / Math.max(1, Number(config.diagnostic_sample_fps || 5));
  if (now - cameraDiagnosticCache.lastSampleAt < interval) return;
  cameraDiagnosticCache.lastSampleAt = now;
  const video = cachedNode("#localVideo");
  if (!video || video.readyState < 2 || !video.videoWidth) return;
  ensureCameraDiagnosticCanvas();
  const started = performance.now();
  const context = cameraDiagnosticCache.context;
  context.drawImage(video, 0, 0, 32, 18);
  const pixels = context.getImageData(0, 0, 32, 18).data;
  let luminanceTotal = 0;
  let differenceTotal = 0;
  for (let pixel = 0, sample = 0; pixel < pixels.length; pixel += 4, sample += 1) {
    const luminance = Math.round(
      pixels[pixel] * 0.2126
      + pixels[pixel + 1] * 0.7152
      + pixels[pixel + 2] * 0.0722,
    );
    luminanceTotal += luminance;
    if (cameraDiagnosticCache.hasPreviousLuma) {
      differenceTotal += Math.abs(luminance - cameraDiagnosticCache.previousLuma[sample]);
    }
    cameraDiagnosticCache.previousLuma[sample] = luminance;
  }
  const samples = cameraDiagnosticCache.previousLuma.length;
  cameraDiagnostics.observeImage(
    luminanceTotal / samples,
    cameraDiagnosticCache.hasPreviousLuma && differenceTotal / samples < 1.5,
  );
  cameraDiagnosticCache.hasPreviousLuma = true;
  const ended = performance.now();
  renderPerformance.recordPhaseSpan("camera_diagnostics", started, ended);
  const snapshot = cameraDiagnostics.snapshot();
  renderCameraDiagnostics(snapshot);
  if (now - cameraDiagnosticCache.lastReportAt >= 2000) {
    cameraDiagnosticCache.lastReportAt = now;
    sendSocket({ type: "camera_diagnostics", diagnostics: snapshot });
  }
}

function startVideoFrameAudit() {
  stopVideoFrameAudit();
  const video = $("#localVideo");
  ui.lastPresentedFrames = 0;
  ui.presentedFrameSkipCount = 0;
  ui.rvfcLateFrameCount = 0;
  ui.rvfcLatenessMs = 0;
  ui.lastFallbackMediaTime = -1;
  ui.videoFrameIntervalMs = 1000 / Math.max(1, Number(ui.realtimeConfig.camera_fps || 60));
  const onVideoFrame = (now, metadata = {}) => {
    if (!ui.mediaStream) return;
    onPresentedVideoFrame(now, metadata, false);
    ui.videoFrameCallbackId = video.requestVideoFrameCallback(onVideoFrame);
  };
  if (typeof video.requestVideoFrameCallback === "function") {
    ui.videoFrameCallbackId = video.requestVideoFrameCallback(onVideoFrame);
  } else {
    const fallbackLoop = now => {
      if (!ui.mediaStream) return;
      ui.drawAnimation = requestAnimationFrame(fallbackLoop);
      if (video.readyState < 2 || video.currentTime === ui.lastFallbackMediaTime) return;
      ui.lastFallbackMediaTime = video.currentTime;
      onPresentedVideoFrame(now, {
        mediaTime: video.currentTime,
        presentationTime: now,
        expectedDisplayTime: now,
        presentedFrames: ui.lastPresentedFrames + 1,
      }, true);
    };
    ui.drawAnimation = requestAnimationFrame(fallbackLoop);
  }
}

function stopVideoFrameAudit() {
  const video = $("#localVideo");
  if (ui.videoFrameCallbackId !== null && typeof video.cancelVideoFrameCallback === "function") {
    video.cancelVideoFrameCallback(ui.videoFrameCallbackId);
  }
  ui.videoFrameCallbackId = null;
  if (ui.drawAnimation !== null) cancelAnimationFrame(ui.drawAnimation);
  ui.drawAnimation = null;
  ui.latestVideoFrameMeta = null;
}

function initializeMainThreadAudit() {
  if (!("PerformanceObserver" in window)) return;
  try {
    ui.longTaskObserver = new PerformanceObserver(list => {
      if (!ui.running) return;
      for (const entry of list.getEntries()) {
        ui.longTaskCount += 1;
        ui.longTaskDurationMs += entry.duration;
        renderPerformance.recordLongTask(entry.startTime, entry.duration);
      }
    });
    ui.longTaskObserver.observe({ type: "longtask", buffered: true });
  } catch (_) { /* Long Task API is optional. */ }
}

function setPoseRuntimeMode(mode, detail = "") {
  ui.poseRuntimeMode = mode;
  const badge = $("#poseModeBadge");
  if (badge) {
    const tier = ui.poseWorkerActiveModel === "lite" ? "Lite" : "Full";
    badge.textContent = mode === "local"
      ? ui.poseWorkerBenchmarking ? `本机模型基准 · ${tier}` : `本机实时姿态 · ${tier}`
      : mode === "server"
        ? "服务器兼容姿态"
        : mode === "unavailable" ? "本机姿态不可用" : "正在准备本机姿态";
    badge.title = detail;
  }
}

function closePoseTransfer(item) {
  try { item?.image?.close?.(); } catch (_) { /* Ignore already transferred frames. */ }
}

function activateServerPoseFallback(reason, notifyUser = true) {
  const fallbackAllowed = ui.realtimeConfig.local_first?.server_pose_fallback !== false;
  if (!fallbackAllowed) {
    closePoseTransfer(ui.poseWorkerPending);
    ui.poseWorkerPending = null;
    ui.poseWorkerRequestedMeta = null;
    ui.poseWorkerBusy = false;
    try { ui.poseWorker?.terminate(); } catch (_) { /* Worker may already be gone. */ }
    ui.poseWorker = null;
    setPoseRuntimeMode("unavailable", reason);
    if (notifyUser) {
      toast(`本机姿态不可用，且当前配置禁止服务器姿态回退：${reason}`, true);
    }
    return false;
  }
  if (ui.poseRuntimeMode === "server" && ui.poseWorker === null) return;
  closePoseTransfer(ui.poseWorkerPending);
  ui.poseWorkerPending = null;
  ui.poseWorkerRequestedMeta = null;
  ui.poseWorkerBusy = false;
  ui.poseWorkerSlowFrameCount = 0;
  ui.poseWorkerBenchmarking = false;
  ui.poseWorkerModelSwitching = false;
  try { ui.poseWorker?.terminate(); } catch (_) { /* Worker may already be gone. */ }
  ui.poseWorker = null;
  setPoseRuntimeMode("server", reason);
  if (notifyUser) toast(`本机姿态不可用，已切换服务器兼容姿态：${reason}`, true);
  return true;
}

function dispatchPoseWorkerFrame(item) {
  if (!item || !ui.running || ui.paused || !ui.poseWorker || ui.poseRuntimeMode !== "local") {
    closePoseTransfer(item);
    return;
  }
  ui.poseWorkerBusy = true;
  ui.poseWorkerTransferMode = item.transferMode;
  ui.poseWorker.postMessage({
    type: "frame",
    image: item.image,
    frameMeta: item.frameMeta,
    transferMode: item.transferMode,
    copyStartMs: item.copyStartMs,
    copyEndMs: item.copyEndMs,
  }, [item.image]);
}

function queuePoseWorkerFrame(item) {
  if (ui.poseWorkerBusy) {
    if (ui.poseWorkerPending) ui.poseWorkerDroppedFrames += 1;
    closePoseTransfer(ui.poseWorkerPending);
    ui.poseWorkerPending = item;
  } else {
    dispatchPoseWorkerFrame(item);
  }
}

async function createPoseTransfer(frameMeta) {
  const video = $("#localVideo");
  const copyStartMs = performance.now();
  if ("VideoFrame" in window) {
    try {
      const image = new VideoFrame(video);
      return { image, frameMeta, transferMode: "video-frame", copyStartMs, copyEndMs: performance.now() };
    } catch (_) { /* Fall through to ImageBitmap. */ }
  }
  if (typeof createImageBitmap === "function") {
    const image = await createImageBitmap(video);
    return { image, frameMeta, transferMode: "image-bitmap", copyStartMs, copyEndMs: performance.now() };
  }
  throw new Error("浏览器不支持 VideoFrame 或 ImageBitmap 转移");
}

async function pumpPoseWorkerFrame() {
  if (ui.poseWorkerFrameCreationBusy || !ui.poseWorkerRequestedMeta || ui.poseRuntimeMode !== "local") return;
  ui.poseWorkerFrameCreationBusy = true;
  const frameMeta = ui.poseWorkerRequestedMeta;
  ui.poseWorkerRequestedMeta = null;
  try {
    const item = await createPoseTransfer(frameMeta);
    renderPerformance.recordPhaseSpan("pose_transfer", item.copyStartMs, item.copyEndMs);
    if (!ui.running || ui.paused || ui.poseRuntimeMode !== "local") {
      closePoseTransfer(item);
    } else if (ui.poseWorkerRequestedMeta && ui.poseWorkerRequestedMeta.frameId > frameMeta.frameId) {
      ui.poseWorkerDroppedFrames += 1;
      closePoseTransfer(item);
    } else {
      queuePoseWorkerFrame(item);
    }
  } catch (error) {
    activateServerPoseFallback(error?.message || String(error));
  } finally {
    ui.poseWorkerFrameCreationBusy = false;
    if (ui.poseWorkerRequestedMeta) void pumpPoseWorkerFrame();
  }
}

function submitLocalPoseFrame(frameMeta) {
  if (ui.poseRuntimeMode !== "local" || !ui.poseWorker || ui.paused) return;
  const interval = 1000 / Math.max(1, Number(ui.realtimeConfig.target_fps || 30));
  const now = performance.now();
  if (now - ui.poseWorkerLastCaptureAt < interval) return;
  ui.poseWorkerLastCaptureAt = now;
  if (ui.poseWorkerRequestedMeta) ui.poseWorkerDroppedFrames += 1;
  ui.poseWorkerRequestedMeta = frameMeta;
  void pumpPoseWorkerFrame();
}

function visibleLocalPoseNames() {
  const profile = selectedLandmarkProfile();
  const face = new Set(poseLandmarkNames.slice(0, 11));
  const upper = new Set(["left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb","left_hip","right_hip"]);
  const lower = new Set(["left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel","left_foot_index","right_foot_index"]);
  const fingerProxy = new Set(["left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb"]);
  return new Set(poseLandmarkNames.filter(name => {
    if (profile === "no-face" && face.has(name)) return false;
    if (profile === "upper-body" && !upper.has(name)) return false;
    if (profile === "lower-body" && !lower.has(name)) return false;
    return $("#fingerToggle").checked || !fingerProxy.has(name);
  }));
}

function serializeLocalLandmarks(landmarks) {
  return (landmarks || []).map(point => ({
    x: finiteNumber(point.x, 0), y: finiteNumber(point.y, 0), z: finiteNumber(point.z, 0),
    visibility: finiteNumber(point.visibility, 1), presence: finiteNumber(point.presence, 1),
  }));
}

function requestLiteModelDowngrade(reason) {
  if (!ui.poseWorker || ui.poseWorkerModelSwitching || ui.poseWorkerActiveModel === "lite") return false;
  const config = ui.realtimeConfig.browser_pose || {};
  if (!config.model_urls?.lite || !config.lite_auto_approved) return false;
  ui.poseWorkerModelSwitching = true;
  ui.poseWorkerBenchmarking = false;
  ui.poseWorkerRequestedMeta = null;
  closePoseTransfer(ui.poseWorkerPending);
  ui.poseWorkerPending = null;
  setPoseRuntimeMode("initializing", reason);
  ui.poseWorker.postMessage({ type: "switch_model", model: "lite", reason });
  return true;
}

function handleLocalPoseResult(message) {
  ui.poseWorkerBusy = false;
  if (!ui.running || ui.paused || ui.poseRuntimeMode !== "local") {
    closePoseTransfer(ui.poseWorkerPending);
    ui.poseWorkerPending = null;
    return;
  }
  ui.poseWorkerFailureCount = 0;
  ui.poseWorkerActiveModel = message.model === "lite" ? "lite" : "full";
  ui.poseWorkerBenchmarking = Boolean(message.benchmarking);
  setPoseRuntimeMode("local", ui.poseWorkerBenchmarking ? "正在比较 Full 与 Lite" : "");
  const config = ui.realtimeConfig.browser_pose || {};
  const inferenceMs = finiteNumber(message.poseInferenceMs, 0);
  const slowThresholdMs = ui.poseWorkerActiveModel === "full"
    ? Number(config.full_overload_inference_ms || 50)
    : Number(config.max_inference_ms || 100);
  ui.poseWorkerSlowFrameCount = !ui.poseWorkerBenchmarking && inferenceMs > slowThresholdMs
    ? ui.poseWorkerSlowFrameCount + 1
    : 0;
  if (ui.poseWorkerSlowFrameCount >= Number(config.slow_frame_limit || 12)) {
    const reason = `${ui.poseWorkerActiveModel === "full" ? "Full" : "Lite"} 推理持续超过 ${slowThresholdMs} ms`;
    if (ui.poseWorkerActiveModel === "full" && requestLiteModelDowngrade(reason)) {
      toast(`设备持续过载，正在从 Full 降为 Lite：${reason}`, true);
    } else {
      activateServerPoseFallback(reason);
    }
    return;
  }
  const frameMeta = message.frameMeta;
  const visible = visibleLocalPoseNames();
  const keypoints = (message.imageLandmarks || []).map((point, index) => ({
    name: poseLandmarkNames[index],
    x: finiteNumber(point.x, 0), y: finiteNumber(point.y, 0), z: finiteNumber(point.z, 0),
    visibility: Math.min(finiteNumber(point.visibility, 1), finiteNumber(point.presence, 1)),
  })).filter(point => point.name && visible.has(point.name));
  const connections = poseConnectionNames;
  const analysis = ui.latestAnalysisResult || {};
  const receivedAt = performance.now();
  ui.latestResult = {
    ...analysis,
    type: "result",
    session_id: ui.activeRealtimeSessionId,
    run_id: ui.activeRealtimeRunId,
    frame_id: frameMeta.frameId,
    sequence: frameMeta.frameId,
    frame_meta: frameMeta,
    action: $("#actionSelect").value,
    request_backend: $("#backendSelect").value,
    pose_detected: keypoints.length > 0,
    keypoints,
    connections,
    assessment: analysis.assessment || {},
    floor_reference: analysis.floor_reference || {},
    display_filter: message.displayFilter || {},
    latency_timing: {
      ...timingForVideoFrame(frameMeta),
      frame_copy_start_ms: message.copyStartMs,
      frame_copy_end_ms: message.copyEndMs,
      local_pose_inference_ms: message.poseInferenceMs,
      local_pose_dropped_frames: ui.poseWorkerDroppedFrames,
      client_result_receive_ms: receivedAt,
    },
    metrics: {
      ...(analysis.metrics || {}),
      backend: `browser-mediapipe-${ui.poseWorkerActiveModel}`,
      inference_ms: message.poseInferenceMs,
      queue_dropped: ui.poseWorkerDroppedFrames,
      display_raw_blend_weight: finiteNumber(message.displayFilter?.meanRawWeight, 0),
      display_blended_point_count: Number(message.displayFilter?.blendedPointCount || 0),
      fps: analysis.metrics?.fps || 0,
    },
  };
  updateDisplayPredictionSource(ui.latestResult);
  ui.lastResultAt = receivedAt;
  $("#loadingOverlay").hidden = true;
  if (domUpdateScheduler.due("local_metrics", receivedAt)) {
    const tierLabel = ui.poseWorkerActiveModel === "lite" ? "Lite" : "Full";
    setTextIfChanged("#backendMetric", `本机 MediaPipe ${tierLabel}${ui.poseWorkerBenchmarking ? " · 基准" : ""}`);
    setTextIfChanged("#latencyMetric", Number(message.poseInferenceMs || 0).toFixed(1));
    setTextIfChanged("#poseMetric", keypoints.length ? "已锁定人体" : "正在寻找人体");
    setClassIfChanged("#poseMetric", keypoints.length ? "good" : "bad");
  }
  if (!ui.poseWorkerBenchmarking) {
    ui.latestResult.latency_timing.socket_send_ms = performance.now();
    const benchmarkReport = ui.poseWorkerBenchmarkSent ? null : ui.poseWorkerBenchmarkReport;
    sendSocket({
      type: "pose_frame",
      session_id: ui.activeRealtimeSessionId,
      run_id: ui.activeRealtimeRunId,
      frame_id: frameMeta.frameId,
      action: $("#actionSelect").value,
      backend: $("#backendSelect").value,
      capture_timestamp_ms: finiteNumber(frameMeta.captureTime, frameMeta.presentationTime),
      presentation_timestamp_ms: frameMeta.presentationTime,
      image_landmarks: serializeLocalLandmarks(message.rawImageLandmarks),
      world_landmarks: serializeLocalLandmarks(message.rawWorldLandmarks),
      pose_inference_ms: finiteNumber(message.poseInferenceMs, 0),
      pose_model: ui.poseWorkerActiveModel,
      pose_model_benchmark: benchmarkReport,
      display_filter: message.displayFilter || {},
      source: "browser_mediapipe",
      frame_meta: frameMeta,
      timing: ui.latestResult.latency_timing,
    });
    if (benchmarkReport) ui.poseWorkerBenchmarkSent = true;
  }
  const pending = ui.poseWorkerPending;
  ui.poseWorkerPending = null;
  if (pending) dispatchPoseWorkerFrame(pending);
}

function initializeBrowserPoseWorker() {
  if (ui.poseWorkerReadyPromise) return ui.poseWorkerReadyPromise;
  if ($("#poseRuntimeSelect")?.value === "server") {
    activateServerPoseFallback("用户选择服务器兼容姿态", false);
    return Promise.resolve(false);
  }
  const config = ui.realtimeConfig.browser_pose || {};
  if (!config.enabled || !("Worker" in window)) {
    activateServerPoseFallback("浏览器不支持本机 Worker 姿态", false);
    return Promise.resolve(false);
  }
  setPoseRuntimeMode("initializing");
  ui.poseWorkerReadyPromise = new Promise(resolve => {
    let settled = false;
    let worker;
    try {
      worker = new Worker(config.worker_url, { type: "module", name: "pose-landmarker" });
    } catch (error) {
      activateServerPoseFallback(error?.message || "无法创建本机姿态 Worker");
      resolve(false);
      return;
    }
    ui.poseWorker = worker;
    const timeout = setTimeout(() => {
      if (settled) return;
      settled = true;
      activateServerPoseFallback("本机模型初始化超时");
      resolve(false);
    }, Number(config.initialization_timeout_ms || 20000));
    worker.onmessage = event => {
      const message = event.data || {};
      if (message.type === "ready") {
        if (!settled) {
          settled = true;
          clearTimeout(timeout);
          ui.poseWorkerActiveModel = message.model === "lite" ? "lite" : "full";
          ui.poseWorkerModelPreference = message.modelPreference || "auto";
          ui.poseWorkerBenchmarking = Boolean(message.benchmarking);
          ui.poseWorkerBenchmarkLongTaskBaseline = ui.longTaskCount;
          setPoseRuntimeMode("local", ui.poseWorkerBenchmarking ? "正在进行 3 秒 Full/Lite 基准" : `MediaPipe Pose Landmarker · ${ui.poseWorkerActiveModel}`);
          resolve(true);
        }
      } else if (message.type === "init_error") {
        if (!settled) {
          settled = true;
          clearTimeout(timeout);
          activateServerPoseFallback(message.message || "本机模型加载失败");
          resolve(false);
        }
      } else if (message.type === "result") {
        handleLocalPoseResult(message);
      } else if (message.type === "benchmark_progress") {
        ui.poseWorkerActiveModel = message.nextModel === "lite" ? "lite" : "full";
        ui.poseWorkerBenchmarking = true;
        setPoseRuntimeMode("local", "Full 基准完成，正在测试 Lite");
      } else if (message.type === "benchmark_complete") {
        ui.poseWorkerActiveModel = message.selectedModel === "lite" ? "lite" : "full";
        ui.poseWorkerBenchmarking = false;
        ui.poseWorkerBenchmarkReport = {
          ...message,
          mainThreadLongTaskCount: Math.max(0, ui.longTaskCount - ui.poseWorkerBenchmarkLongTaskBaseline),
        };
        ui.poseWorkerBenchmarkSent = false;
        setPoseRuntimeMode("local", message.reason || "自动基准完成");
        const fullP95 = Number(message.stats?.full?.inferenceP95Ms || 0).toFixed(1);
        const liteP95 = Number(message.stats?.lite?.inferenceP95Ms || 0).toFixed(1);
        toast(`本机姿态基准完成：Full P95 ${fullP95} ms，Lite P95 ${liteP95} ms；已选择 ${ui.poseWorkerActiveModel === "lite" ? "Lite" : "Full"}`);
      } else if (message.type === "model_changed") {
        ui.poseWorkerActiveModel = message.model === "lite" ? "lite" : "full";
        ui.poseWorkerModelSwitching = false;
        ui.poseWorkerSlowFrameCount = 0;
        setPoseRuntimeMode("local", message.reason || "姿态档位已切换");
        toast(`本机姿态已切换为 ${ui.poseWorkerActiveModel === "lite" ? "Lite" : "Full"}`);
      } else if (message.type === "model_change_error") {
        ui.poseWorkerModelSwitching = false;
        activateServerPoseFallback(message.message || "Lite 模型切换失败");
      } else if (message.type === "frame_skipped") {
        ui.poseWorkerBusy = false;
      } else if (message.type === "frame_error") {
        ui.poseWorkerBusy = false;
        ui.poseWorkerFailureCount += 1;
        if (ui.poseWorkerFailureCount >= 3) activateServerPoseFallback(message.message || "本机推理连续失败");
        else if (ui.poseWorkerPending && ui.running && !ui.paused) {
          const pending = ui.poseWorkerPending;
          ui.poseWorkerPending = null;
          dispatchPoseWorkerFrame(pending);
        } else if (ui.poseWorkerPending) {
          closePoseTransfer(ui.poseWorkerPending);
          ui.poseWorkerPending = null;
        }
      }
    };
    worker.onerror = event => {
      clearTimeout(timeout);
      if (!settled) { settled = true; resolve(false); }
      activateServerPoseFallback(event.message || "本机姿态 Worker 异常");
    };
    worker.postMessage({
      type: "init",
      wasmRoot: config.wasm_root,
      modelUrls: config.model_urls || { full: config.model_url },
      modelPreference: $("#poseModelSelect")?.value || config.model_preference || "auto",
      benchmarkDurationMs: config.benchmark_duration_ms || 3000,
      liteAutoApproved: Boolean(config.lite_auto_approved),
      displaySmoothing: config.display_smoothing || {},
    });
  });
  return ui.poseWorkerReadyPromise;
}

function restartBrowserPoseWorker() {
  closePoseTransfer(ui.poseWorkerPending);
  ui.poseWorkerPending = null;
  ui.poseWorkerRequestedMeta = null;
  try { ui.poseWorker?.terminate(); } catch (_) { /* Worker may already be gone. */ }
  ui.poseWorker = null;
  ui.poseWorkerReadyPromise = null;
  ui.poseWorkerBusy = false;
  ui.poseWorkerFailureCount = 0;
  ui.poseWorkerSlowFrameCount = 0;
  ui.poseWorkerDroppedFrames = 0;
  ui.poseWorkerBenchmarking = false;
  ui.poseWorkerBenchmarkReport = null;
  ui.poseWorkerBenchmarkSent = false;
  ui.poseWorkerModelSwitching = false;
  displayPosePredictor.reset();
  setPoseRuntimeMode("initializing");
  if (ui.mediaStream && $("#poseRuntimeSelect")?.value !== "server") void initializeBrowserPoseWorker();
}

const phaseLabels = {
  idle: "等待开始", unknown: "识别中", ready: "准备就绪", no_pose: "未检测到姿态", low_visibility: "可见度不足",
  stand: "站立", standing: "站立", descent: "下降", ascent: "起身", squat_down: "下蹲", bottom: "最低点", drive: "发力",
  throw_extension: "投球伸展", reset: "复位", recovery: "恢复", carrying: "负重行走", rest: "停步休息",
  pull: "拉动", pull_down: "下拉", return: "回位", catch: "起始", finish: "划船终点", top: "顶部",
  setup: "准备", step: "蹬地迈步", hands_down: "双手撑地", chest_down: "俯卧最低点",
  step_or_jump_in: "收腿", broad_jump_takeoff: "跳远起跳", flight_or_move: "腾空或移动",
  reach: "前伸取绳", recover: "向前移动回位", walking: "行走", airborne: "腾空", landing: "落地", support: "支撑",
};
const actionCameraTips = {
  none: {
    view: "根据训练动作选择",
    framing: "保持全身、双手、双脚和脚下地板完整入镜",
  },
  lunge: {
    view: "侧面或斜侧面",
    framing: "全身、双脚与脚下地板，确保站立和后膝触地阶段不出画",
  },
  wall_ball: {
    view: "正面或斜前方",
    framing: "全身、脚下地板、双脚与双手腕，手臂上举时也不得出画",
  },
  rowing: {
    view: "侧面",
    framing: "全身、双手和双脚，并覆盖划船起始与结束位置",
  },
  skierg: {
    view: "正面或斜前方",
    framing: "全身、双手和双脚，并预留双手到顶部与下拉底部的垂直空间",
  },
  burpee_broad_jump: {
    view: "侧面约 45°",
    framing: "全身、双手、双脚和下一落地区域，前进方向预留足够空间",
  },
  sled_push: {
    view: "侧面或斜侧面",
    framing: "全身、双手、双脚及前方移动区域，推动过程中持续不出画",
  },
  sled_pull: {
    view: "侧面或斜侧面",
    framing: "全身、双手、双脚及拉动和回位区域，移动过程中持续不出画",
  },
  farmers_carry: {
    view: "正面或斜前方",
    framing: "全身、双手、双脚及身体两侧负重区域，行走过程中持续不出画",
  },
};
const backendLabels = {
  "browser-mediapipe": "本机 MediaPipe",
  "browser-mediapipe-full": "本机 MediaPipe Full",
  "browser-mediapipe-lite": "本机 MediaPipe Lite",
  "sample-cache": "预计算示例结果",
  "yolo-rtmw-wholebody": "YOLO + RTMW WholeBody",
  "yolo-guided-mediapipe-fallback": "YOLO + MediaPipe（RTMW 降级）",
  "yolo-guided-mediapipe": "YOLO + MediaPipe",
  "yolo-mediapipe": "YOLO + MediaPipe",
  "yolo-pose": "YOLO Pose",
  mediapipe: "纯 MediaPipe",
};

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(node._timer);
  node._timer = setTimeout(() => node.className = "toast", 3200);
}

function setVoiceStatus(message) {
  const node = $("#voiceStatus");
  if (node) node.textContent = message;
}

function preferredChineseVoice() {
  if (!ui.voiceSupported) return null;
  const voices = window.speechSynthesis.getVoices();
  return voices.find(voice => /^zh[-_](CN|Hans)/i.test(voice.lang))
    || voices.find(voice => /^zh/i.test(voice.lang))
    || null;
}

function cancelVoiceFeedback() {
  if (ui.voiceSupported) window.speechSynthesis.cancel();
}

function resetVoiceSession() {
  cancelVoiceFeedback();
  ui.lastVoiceEventId = "";
  setVoiceStatus(ui.voiceEnabled ? "每次完成后播报" : "语音提示已关闭");
}

function speakVoiceFeedback(event) {
  if (!event || !event.id || event.id === ui.lastVoiceEventId) return;
  ui.lastVoiceEventId = event.id;
  if (!event.speech) {
    setVoiceStatus(event.rep ? `第 ${event.rep} 次未发现持续性问题` : "当前动作稳定");
    return;
  }
  if (!ui.voiceEnabled || !ui.voiceSupported) return;
  cancelVoiceFeedback();
  const utterance = new SpeechSynthesisUtterance(String(event.speech));
  utterance.lang = "zh-CN";
  utterance.rate = 1.05;
  utterance.pitch = 1;
  utterance.volume = 1;
  const voice = preferredChineseVoice();
  if (voice) utterance.voice = voice;
  utterance.onstart = () => setVoiceStatus(event.rep ? `正在播报第 ${event.rep} 次` : "正在播报动作提示");
  utterance.onend = () => setVoiceStatus("每次完成后播报");
  utterance.onerror = speechEvent => {
    if (speechEvent.error !== "canceled" && speechEvent.error !== "interrupted") setVoiceStatus("语音播放失败，请检查系统音量");
  };
  window.speechSynthesis.speak(utterance);
}

function initializeVoiceFeedback() {
  const button = $("#voiceToggle");
  if (!button) return;
  if (!ui.voiceSupported) {
    ui.voiceEnabled = false;
    button.disabled = true;
    button.setAttribute("aria-pressed", "false");
    button.textContent = "语音不可用";
    setVoiceStatus("当前浏览器不支持语音合成");
    return;
  }
  try { ui.voiceEnabled = localStorage.getItem("hyroxVoiceFeedback") !== "off"; } catch (_) { /* storage may be blocked */ }
  const render = () => {
    button.setAttribute("aria-pressed", String(ui.voiceEnabled));
    button.textContent = ui.voiceEnabled ? "🔊 语音开" : "🔇 语音关";
    setVoiceStatus(ui.voiceEnabled ? "每次完成后播报" : "语音提示已关闭");
  };
  button.addEventListener("click", () => {
    ui.voiceEnabled = !ui.voiceEnabled;
    if (!ui.voiceEnabled) cancelVoiceFeedback();
    try { localStorage.setItem("hyroxVoiceFeedback", ui.voiceEnabled ? "on" : "off"); } catch (_) { /* storage may be blocked */ }
    render();
    toast(ui.voiceEnabled ? "已开启逐次动作语音提示" : "已关闭语音提示");
  });
  render();
}

async function api(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  const headers = options.body instanceof FormData ? {} : { "Content-Type": "application/json" };
  if (method !== "GET" && ui.csrfToken) headers["X-CSRF-Token"] = ui.csrfToken;
  const response = await fetch(path, { ...options, headers: { ...headers, ...(options.headers || {}) } });
  let data = {};
  try { data = await response.json(); } catch (_) { /* response may be empty */ }
  if (!response.ok) throw new Error(data.error || "操作失败，请稍后重试");
  return data;
}

async function loadOptions() {
  try {
    const options = await api("/api/options");
    ui.csrfToken = options.csrf_token;
    ui.realtimeConfig = options.realtime || ui.realtimeConfig;
    domUpdateScheduler.configure(ui.realtimeConfig.rendering || {});
    renderPerformance.configure(ui.realtimeConfig.rendering || {});
    cameraDiagnostics.configure(ui.realtimeConfig.camera || {});
    displayPosePredictor.configure(ui.realtimeConfig.browser_pose?.display_prediction || {});
    const modelPreference = ui.realtimeConfig.browser_pose?.model_preference || "auto";
    if (["auto", "lite", "full"].includes(modelPreference)) {
      $("#poseModelSelect").value = modelPreference;
      ui.poseWorkerModelPreference = modelPreference;
    }
    ui.samplesByAction = new Map((options.samples || []).map(item => [item.action, item]));
    ui.standards = options.standards || {};
    ui.officialRules = options.official_rules || {};
    $("#actionSelect").innerHTML = options.actions.map(item => `<option value="${item.value}">${item.label}</option>`).join("");
    $("#actionSelect").value = "lunge";
    $("#viewSelect").innerHTML = options.views.map(item => `<option value="${item.value}">${item.label}</option>`).join("");
    $("#viewSelect").value = "side";
    renderStandards($("#actionSelect").value);
    renderCameraTip($("#actionSelect").value);
  } catch (error) {
    toast(error.message, true);
  }
}

function settingsPayload() {
  return {
    action: $("#actionSelect").value,
    camera_view: $("#viewSelect").value,
    sensitivity: $("#sensitivitySelect").value,
    backend: $("#backendSelect").value,
    landmark_profile: selectedLandmarkProfile(),
    show_fingers: $("#fingerToggle").checked,
    mirror: $("#mirrorToggle").checked,
    paused: ui.paused,
    manual_floor_points: ui.manualFloorPoints,
  };
}

function selectedLandmarkProfile() {
  const profile = $("#profileSelect").value;
  return $("#faceToggle").checked && profile === "full" ? "no-face" : profile;
}

function renderStandards(action) {
  const list = $("#standardsList");
  if (!list) return;
  const standards = ui.standards[action] || [];
  const officialRules = ui.officialRules[action] || [];
  $("#standardsPhase").textContent = $("#actionSelect")?.selectedOptions[0]?.textContent || "动作标准";
  const ruleRows = officialRules.length
    ? `<div class="official-rule-list"><b>HYROX 26/27 官方要求</b>${officialRules.map(item => `<p><span>${item.pose_observable ? "可视觉判断" : "需现场确认"}</span>${escapeHtml(item.text)}</p>`).join("")}</div>`
    : "";
  const standardRows = standards.length
    ? standards.map(item => `<div class="standard-row"><strong>${escapeHtml(item.label)}</strong><b>${escapeHtml(item.range_text)}</b><small>${escapeHtml(item.category_text || "训练参考")} · ${escapeHtml(item.phase_text)}${item.note ? ` · ${escapeHtml(item.note)}` : ""}</small></div>`).join("")
    : `<p>该模式没有可由人体关键点直接判断的角度标准。</p>`;
  list.innerHTML = ruleRows + standardRows;
}

function renderCameraTip(action) {
  const tip = actionCameraTips[action] || actionCameraTips.none;
  setTextIfChanged("#cameraTip", `推荐视角：${tip.view}；入镜范围：${tip.framing}。`);
}

function setConnectionMetricForSource(mode = ui.sourceMode) {
  setTextIfChanged("#connectionMetric", mode === "camera" ? "未连接" : "本机处理");
  setClassIfChanged("#connectionMetric", "neutral");
}

function selectSource(mode) {
  if (ui.running) return;
  if (ui.sourceMode === "camera" && mode !== "camera") closeCamera();
  ui.sourceMode = mode;
  $$("#sourceTabs .segment").forEach(button => button.classList.toggle("active", button.dataset.source === mode));
  $$(".source-pane").forEach(pane => pane.classList.toggle("active", pane.dataset.pane === mode));
  $("#mirrorToggle").checked = mode === "camera";
  $("#startButton span:first-child").textContent = mode === "camera" ? "开始实时分析" : "开始分析";
  $("#stopButton").textContent = mode === "camera" ? "停止摄像头" : "停止";
  if (mode === "sample" && !ui.samplesByAction.has($("#actionSelect").value)) {
    const firstAction = ui.samplesByAction.keys().next().value;
    if (firstAction) $("#actionSelect").value = firstAction;
  }
  renderStandards($("#actionSelect").value);
  renderCameraTip($("#actionSelect").value);
  setConnectionMetricForSource(mode);
}

function setRunning(running) {
  ui.running = running;
  $("#videoRepBadge").hidden = !running;
  $("#startButton").disabled = running;
  $("#stopButton").disabled = !running && !ui.mediaStream;
  $("#pauseButton").disabled = !running;
  $("#recordButton").disabled = !running || !window.MediaRecorder;
  $("#screenshotButton").disabled = !running;
  $$("#sourceTabs button, #videoFile, #backendSelect, #poseRuntimeSelect, #poseModelSelect, #openCameraButton").forEach(node => { node.disabled = running; });
  $("#switchCameraButton").disabled = !ui.mediaStream;
  $("#cameraDevice").disabled = !ui.mediaStream;
}

function setPermissionState(state, message) {
  const node = $("#cameraPermission");
  node.dataset.state = state;
  node.textContent = message;
}

function stopMediaTracks() {
  stopVideoFrameAudit();
  displayPosePredictor.reset();
  cameraDiagnostics.reset();
  cameraDiagnosticCache.hasPreviousLuma = false;
  cameraDiagnosticCache.lastSampleAt = 0;
  cameraDiagnosticCache.lastReportAt = 0;
  setTextIfChanged("#cameraDiagnostic", "开启摄像头后显示实际规格与硬件诊断");
  if (ui.mediaStream) ui.mediaStream.getTracks().forEach(track => track.stop());
  ui.mediaStream = null;
  const video = $("#localVideo");
  video.pause();
  video.srcObject = null;
}

async function openCamera({ deviceId = "", facingMode = ui.facingMode } = {}) {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setPermissionState("error", "摄像头需要 HTTPS 或本机安全地址");
    throw new Error("当前页面不是安全连接，浏览器无法开放摄像头");
  }
  setPermissionState("requesting", "正在等待你确认浏览器摄像头权限…");
  stopMediaTracks();
  const cameraConfig = ui.realtimeConfig.camera || {};
  const requestedCameraFps = cameraConfig.preferred_fps || ui.realtimeConfig.camera_fps || 60;
  const fallbackCameraFps = cameraConfig.fallback_fps || 30;
  const preferredWidth = cameraConfig.preferred_width || ui.realtimeConfig.frame_width || 640;
  const preferredHeight = cameraConfig.preferred_height || ui.realtimeConfig.frame_height || 480;
  const videoConstraint = deviceId
    ? {
        deviceId: { exact: deviceId },
        width: { ideal: preferredWidth },
        height: { ideal: preferredHeight },
        frameRate: { ideal: requestedCameraFps, min: fallbackCameraFps },
      }
    : {
        facingMode: { ideal: facingMode },
        width: { ideal: preferredWidth },
        height: { ideal: preferredHeight },
        frameRate: { ideal: requestedCameraFps, min: fallbackCameraFps },
      };
  try {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: false });
    } catch (error) {
      if (error.name !== "OverconstrainedError") throw error;
      const relaxedConstraint = deviceId
        ? {
            deviceId: { exact: deviceId },
            width: { ideal: preferredWidth },
            height: { ideal: preferredHeight },
            frameRate: { ideal: fallbackCameraFps },
          }
        : {
            facingMode: { ideal: facingMode },
            width: { ideal: preferredWidth },
            height: { ideal: preferredHeight },
            frameRate: { ideal: fallbackCameraFps },
          };
      stream = await navigator.mediaDevices.getUserMedia({
        video: relaxedConstraint,
        audio: false,
      });
      toast(`摄像头不支持首选 ${requestedCameraFps} FPS，已明确回退到 ${fallbackCameraFps} FPS`);
    }
    ui.mediaStream = stream;
    const video = $("#localVideo");
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    startVideoFrameAudit();
    void initializeBrowserPoseWorker();
    video.hidden = false;
    video.classList.toggle("mirrored", $("#mirrorToggle").checked);
    $("#streamImage").hidden = true;
    $("#emptyStage").hidden = true;
    $("#videoBadges").hidden = false;
    const track = stream.getVideoTracks()[0];
    const settings = track?.getSettings() || {};
    cameraDiagnostics.setSettings(settings);
    renderCameraDiagnostics(cameraDiagnostics.snapshot());
    $("#sourceBadge").textContent = track?.label || "本机摄像头";
    const actualFps = Number(settings.frameRate || 0);
    $("#captureMetric").textContent = actualFps > 0 ? `${actualFps.toFixed(0)} FPS` : "设备自适应";
    $("#openCameraButton").textContent = "关闭摄像头";
    $("#switchCameraButton").disabled = false;
    $("#cameraDevice").disabled = false;
    $("#stopButton").disabled = false;
    setPermissionState("allowed", "已允许；未采集麦克风音频");
    await refreshCameraDevices();
    resizeOverlay();
    return stream;
  } catch (error) {
    stopMediaTracks();
    const messages = {
      NotAllowedError: "摄像头权限已被拒绝。请在浏览器地址栏的网站设置中允许摄像头后重试。",
      NotFoundError: "未找到可用摄像头，请连接设备后重试。",
      NotReadableError: "摄像头正被其他应用占用，请关闭占用程序后重试。",
      OverconstrainedError: "所选摄像头不支持需要的画面规格，请切换设备。",
      SecurityError: "浏览器安全策略禁止使用摄像头，请确认使用 HTTPS。",
    };
    const message = messages[error.name] || `无法开启摄像头：${error.message || "未知错误"}`;
    setPermissionState(error.name === "NotAllowedError" ? "denied" : "error", message);
    throw new Error(message);
  }
}

async function refreshCameraDevices() {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === "videoinput");
  const select = $("#cameraDevice");
  const activeId = ui.mediaStream?.getVideoTracks()[0]?.getSettings().deviceId || "";
  select.innerHTML = devices.length
    ? devices.map((device, index) => `<option value="${escapeHtml(device.deviceId)}">${escapeHtml(device.label || `摄像头 ${index + 1}`)}</option>`).join("")
    : `<option value="">未找到摄像头</option>`;
  if (activeId) select.value = activeId;
}

function closeCamera() {
  ui.floorCalibrationActive = false;
  $("#overlayCanvas").classList.remove("floor-calibrating");
  $("#floorCalibrateButton").classList.remove("active");
  stopCaptureLoop();
  stopMediaTracks();
  clearOverlay();
  $("#localVideo").hidden = true;
  $("#openCameraButton").textContent = "开启摄像头";
  $("#switchCameraButton").disabled = true;
  $("#cameraDevice").disabled = true;
  $("#captureMetric").textContent = "—";
  if (ui.sourceMode === "camera") {
    $("#emptyStage").hidden = false;
    $("#videoBadges").hidden = true;
  }
  if (!ui.running) $("#stopButton").disabled = true;
  setPermissionState("idle", "摄像头已关闭，所有视频轨道均已释放");
}

async function toggleCameraPreview() {
  try {
    if (ui.mediaStream) closeCamera();
    else await openCamera();
  } catch (error) {
    toast(error.message, true);
  }
}

async function switchCamera() {
  ui.facingMode = ui.facingMode === "user" ? "environment" : "user";
  try {
    await openCamera({ facingMode: ui.facingMode });
    toast(ui.facingMode === "environment" ? "已切换到后置摄像头" : "已切换到前置摄像头");
  } catch (error) {
    toast(error.message, true);
  }
}

function websocketUrl() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host}/ws/pose?csrf=${encodeURIComponent(ui.csrfToken)}`;
}

function sendSocket(payload) {
  if (ui.socket?.readyState === WebSocket.OPEN) ui.socket.send(JSON.stringify(payload));
}

function connectRealtime() {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(websocketUrl());
    socket.binaryType = "arraybuffer";
    ui.socket = socket;
    const timeout = setTimeout(() => {
      socket.close();
      reject(new Error("实时分析连接超时"));
    }, 8000);
    socket.onopen = () => {
      $("#connectionMetric").textContent = "正在连接";
      $("#connectionMetric").className = "neutral";
    };
    socket.onmessage = event => {
      let message;
      try { message = JSON.parse(event.data); } catch (_) { return; }
      if (message.type === "connected") {
        clearTimeout(timeout);
        ui.reconnectAttempts = 0;
        ui.activeRealtimeSessionId = message.session_id || "";
        ui.sequence = Math.max(ui.sequence, Number(message.state?.last_submitted_frame_id || 0));
        sendSocket({ type: "start", settings: settingsPayload() });
        resolve();
      } else if (message.type === "started") {
        ui.activeRealtimeRunId = message.state?.run_id || "";
        $("#loadingOverlay").hidden = false;
        startPoseDrawLoop();
        startCaptureLoop();
      } else if (message.type === "result") {
        handleRealtimeResult(message);
      } else if (message.type === "state") {
        updateState(message.state);
      } else if (message.type === "frame_dropped") {
        if (Number.isFinite(Number(message.frame_id))) {
          ui.lastDiscardedFrameId = Math.max(ui.lastDiscardedFrameId, Number(message.frame_id));
        }
        finishFrameRequest(message.frame_id);
        ui.captureFps = Math.max(15, ui.captureFps - 2);
        $("#connectionMetric").textContent = "正在自适应";
        $("#connectionMetric").className = "neutral";
      } else if (message.type === "error") {
        if (Number.isFinite(Number(message.frame_id))) {
          ui.lastDiscardedFrameId = Math.max(ui.lastDiscardedFrameId, Number(message.frame_id));
        }
        finishFrameRequest(message.frame_id);
        toast(message.message || "实时分析出错", true);
        if (["server_busy", "csrf_failed", "origin_rejected"].includes(message.code)) stopAnalysis();
      }
    };
    socket.onerror = () => {
      clearTimeout(timeout);
      if (!ui.running) reject(new Error("无法建立实时分析连接"));
    };
    socket.onclose = () => {
      clearTimeout(timeout);
      stopCaptureLoop();
      const shouldReconnect = ui.sourceMode === "camera" && ui.running && !ui.manualStop;
      if (shouldReconnect) {
        $("#connectionMetric").textContent = "连接中断";
        $("#connectionMetric").className = "bad";
        scheduleReconnect();
      } else if (ui.sourceMode !== "camera") {
        setConnectionMetricForSource(ui.sourceMode);
      }
    };
  });
}

function scheduleReconnect() {
  clearTimeout(ui.reconnectTimer);
  if (ui.reconnectAttempts >= 5) {
    toast("实时连接多次恢复失败，请停止后重试", true);
    return;
  }
  const delay = Math.min(8000, 500 * (2 ** ui.reconnectAttempts));
  ui.reconnectAttempts += 1;
  $("#topStatus").textContent = `${Math.round(delay / 100) / 10} 秒后重新连接…`;
  ui.reconnectTimer = setTimeout(() => connectRealtime().catch(() => scheduleReconnect()), delay);
}

function startCaptureLoop() {
  stopCaptureLoop();
  ui.captureFps = ui.realtimeConfig.target_fps || 30;
  ui.longTaskCount = 0;
  ui.longTaskDurationMs = 0;
  ui.lastAuditedPoseFrameId = -1;
  domUpdateScheduler.reset();
  renderPerformance.reset();
}

async function captureLatestFrame(frameMeta) {
  const interval = 1000 / Math.max(1, ui.captureFps);
  if (!ui.running || !ui.mediaStream || ui.paused || ui.socket?.readyState !== WebSocket.OPEN) {
    return;
  }
  if (!frameMeta || performance.now() - ui.lastFrameSentAt < interval) return;
  if (ui.requestInFlight || ui.socket.bufferedAmount > 1024 * 1024) return;
  ui.requestInFlight = true;
  const video = $("#localVideo");
  if (video.readyState < 2 || video.videoWidth <= 0) {
    finishFrameRequest();
    return;
  }
  const canvas = $("#captureCanvas");
  const longEdge = ui.realtimeConfig.inference_long_edge || 640;
  const scale = Math.min(1, longEdge / Math.max(video.videoWidth, video.videoHeight));
  canvas.width = Math.max(2, Math.round(video.videoWidth * scale));
  canvas.height = Math.max(2, Math.round(video.videoHeight * scale));
  const clientCaptureMs = performance.now();
  const sourceFrameMeta = timingForVideoFrame(frameMeta);
  const frameCopyStartMs = performance.now();
  canvas.getContext("2d", { alpha: false }).drawImage(video, 0, 0, canvas.width, canvas.height);
  const frameCopyEndMs = performance.now();
  renderPerformance.recordPhaseSpan("frame_copy", frameCopyStartMs, frameCopyEndMs);
  const encodeStartMs = performance.now();
  const jpegQuality = Number(ui.realtimeConfig.jpeg_quality ?? 0.65);
  const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", jpegQuality));
  const encodeEndMs = performance.now();
  renderPerformance.recordPhaseSpan("encode", encodeStartMs, encodeEndMs);
  if (!ui.running || ui.socket?.readyState !== WebSocket.OPEN) {
    finishFrameRequest();
    return;
  }
  if (!blob || blob.size > (ui.realtimeConfig.max_frame_bytes || 512 * 1024)) {
    finishFrameRequest();
    return;
  }
  const image = new Uint8Array(await blob.arrayBuffer());
  if (!ui.running || ui.socket?.readyState !== WebSocket.OPEN) {
    finishFrameRequest();
    return;
  }
  const frameId = frameMeta.frameId;
  const socketSendMs = performance.now();
  const timing = {
    ...sourceFrameMeta,
    frame_copy_start_ms: frameCopyStartMs,
    frame_copy_end_ms: frameCopyEndMs,
    encode_start_ms: encodeStartMs,
    encode_end_ms: encodeEndMs,
    socket_send_ms: socketSendMs,
  };
  const metadata = new TextEncoder().encode(JSON.stringify({
    session_id: ui.activeRealtimeSessionId,
    run_id: ui.activeRealtimeRunId,
    frame_id: frameId,
    client_capture_ms: clientCaptureMs,
    action: $("#actionSelect").value,
    backend: $("#backendSelect").value,
    frame_meta: frameMeta,
    timing,
  }));
  const packet = new Uint8Array(8 + metadata.length + image.length);
  packet.set([0x50, 0x53, 0x56, 0x32], 0);
  new DataView(packet.buffer).setUint32(4, metadata.length, false);
  packet.set(metadata, 8);
  packet.set(image, 8 + metadata.length);
  const sentAt = performance.now();
  ui.lastFrameSentAt = sentAt;
  ui.inFlightFrameId = frameId;
  ui.pendingFrames.clear();
  ui.pendingFrames.set(frameId, { sentAt, captureMs: clientCaptureMs, timing, frameMeta });
  try {
    ui.socket.send(packet.buffer);
  } catch (_) {
    finishFrameRequest(frameId);
    return;
  }
  clearTimeout(ui.requestTimeout);
  ui.requestTimeout = setTimeout(() => {
    if (ui.inFlightFrameId !== frameId) return;
    ui.pendingFrames.delete(frameId);
    ui.lastDiscardedFrameId = Math.max(ui.lastDiscardedFrameId, frameId);
    ui.requestInFlight = false;
    ui.inFlightFrameId = -1;
    ui.staleResultCount += 1;
    $("#connectionMetric").textContent = "请求超时 · 正在恢复";
    $("#connectionMetric").className = "bad";
  }, Math.max(1000, Number(ui.realtimeConfig.request_timeout_ms || 3000)));
}

function finishFrameRequest(frameId = null) {
  if (frameId !== null && frameId !== undefined && Number(frameId) !== ui.inFlightFrameId) return;
  clearTimeout(ui.requestTimeout);
  ui.requestTimeout = null;
  if (ui.inFlightFrameId >= 0) ui.pendingFrames.delete(ui.inFlightFrameId);
  ui.requestInFlight = false;
  ui.inFlightFrameId = -1;
}

function stopCaptureLoop() {
  clearTimeout(ui.requestTimeout);
  ui.requestTimeout = null;
  ui.requestInFlight = false;
  ui.inFlightFrameId = -1;
  ui.pendingFrames.clear();
  ui.poseWorkerRequestedMeta = null;
  closePoseTransfer(ui.poseWorkerPending);
  ui.poseWorkerPending = null;
}

function handleRealtimeResult(result) {
  const frameId = Number(result.frame_id ?? result.sequence ?? -1);
  const receivedAt = performance.now();
  const pending = ui.pendingFrames.get(frameId);
  finishFrameRequest(frameId);
  const contextIsCurrent = Boolean(
    ui.running
    && result.session_id === ui.activeRealtimeSessionId
    && result.run_id === ui.activeRealtimeRunId
    && result.action === $("#actionSelect").value
    && result.request_backend === $("#backendSelect").value
  );
  if (
    !contextIsCurrent
    || frameId <= ui.lastRenderedPoseFrameId
    || frameId <= ui.lastDiscardedFrameId
  ) {
    ui.staleResultCount += 1;
    return;
  }
  const captureMs = Number(result.client_capture_ms ?? pending?.captureMs);
  const poseAge = Number.isFinite(captureMs) ? Math.max(0, receivedAt - captureMs) : Number(result.pose_age_ms || 0);
  if (poseAge > Number(ui.realtimeConfig.hide_pose_after_ms || 300)) {
    ui.staleResultCount += 1;
    return;
  }
  ui.lastRenderedPoseFrameId = frameId;
  const isLocalAnalysis = ui.poseRuntimeMode === "local" && result.source === "browser_mediapipe";
  if (isLocalAnalysis) {
    ui.latestAnalysisResult = result;
    if (ui.latestResult) {
      ui.latestResult = {
        ...ui.latestResult,
        assessment: result.assessment || {},
        floor_reference: result.floor_reference || {},
      };
    }
  } else {
    ui.latestResult = {
      ...result,
      frame_id: frameId,
      frame_meta: result.frame_meta || pending?.frameMeta || null,
      client_capture_ms: captureMs,
      latency_timing: { ...(result.latency_timing || pending?.timing || {}), client_result_receive_ms: receivedAt },
    };
    ui.lastResultAt = receivedAt;
    updateDisplayPredictionSource(ui.latestResult);
  }
  const roundTrip = pending ? receivedAt - pending.sentAt : result.metrics.server_ms;
  let quality = "流畅";
  let qualityClass = "good";
  if (roundTrip > 500) { quality = "网络较慢"; qualityClass = "bad"; }
  else if (roundTrip > 250) { quality = "一般"; qualityClass = "neutral"; }
  if (roundTrip > 500) ui.captureFps = Math.max(15, ui.captureFps - 2);
  else if (roundTrip < 200 && result.sequence % 30 === 0) ui.captureFps = Math.min(ui.realtimeConfig.target_fps || 30, ui.captureFps + 1);
  if (domUpdateScheduler.due("connection_metrics", receivedAt)) {
    setTextIfChanged(
      "#connectionMetric",
      isLocalAnalysis
        ? `本机姿态 · 规则 ${Math.round(roundTrip)} ms`
        : `${quality} · RTT ${Math.round(roundTrip)} ms · 姿态 ${Math.round(poseAge)} ms`,
    );
    setClassIfChanged("#connectionMetric", qualityClass);
  }
  $("#loadingOverlay").hidden = true;
  updateState({
    running: true,
    status: "running",
    status_text: "实时分析中",
    source_name: ui.mediaStream?.getVideoTracks()[0]?.label || "本机摄像头",
    source_mode: "browser-camera",
    backend: result.metrics.backend,
    action: result.action,
    action_label: result.action_label,
    camera_view: $("#viewSelect").value,
    pose_detected: result.pose_detected,
    fps: result.metrics.fps,
    inference_ms: result.metrics.inference_ms,
    phase: result.phase,
    reps: result.reps,
    candidate_count: result.candidate_count,
    pose_valid_rep_count: result.pose_valid_rep_count,
    no_rep_count: result.no_rep_count,
    unsure_count: result.unsure_count,
    floor_reference: result.floor_reference,
    feedback: result.feedback,
    voice_feedback: result.voice_feedback,
    frame_index: result.sequence,
  });
  $$("#downloadText, #downloadJson, #downloadCsv").forEach(link => link.setAttribute("aria-disabled", "false"));
  $("#generateReportButton").disabled = false;
  $("#reportReadyBanner").hidden = false;
}

function resizeOverlay() {
  const canvas = $("#overlayCanvas");
  const stage = $("#videoStage");
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.round(stage.clientWidth * ratio));
  canvas.height = Math.max(1, Math.round(stage.clientHeight * ratio));
  canvas.hidden = !ui.mediaStream;
}

function clearOverlay() {
  const canvas = $("#overlayCanvas");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  canvas.hidden = true;
}

function displayPredictionIdentity(result) {
  return [
    result?.session_id || "",
    result?.run_id || "",
    result?.action || "",
    result?.metrics?.backend || result?.request_backend || "",
  ].join("|");
}

function updateDisplayPredictionSource(result) {
  const frameMeta = result?.frame_meta || {};
  const rawCaptureTimestamp = frameMeta.captureTime;
  const captureTimestamp = rawCaptureTimestamp === null || rawCaptureTimestamp === undefined
    ? null
    : finiteNumber(rawCaptureTimestamp);
  const sourceTimestamp = captureTimestamp ?? finiteNumber(frameMeta.presentationTime);
  if (!result?.pose_detected || !Array.isArray(result.keypoints) || sourceTimestamp === null) {
    displayPosePredictor.reset(displayPredictionIdentity(result));
    return;
  }
  displayPosePredictor.update(
    result.keypoints,
    sourceTimestamp,
    displayPredictionIdentity(result),
  );
}

function renderPoseForVideoFrame(currentVideoFrame, now = performance.now()) {
  const started = performance.now();
  try {
    return renderPoseForVideoFrameCore(currentVideoFrame, now);
  } finally {
    const ended = performance.now();
    renderPerformance.record("render_loop_ms", ended - started);
    renderPerformance.recordPhaseSpan("render", started, ended);
  }
}

function renderPoseForVideoFrameCore(currentVideoFrame, now = performance.now()) {
  if (!ui.mediaStream || !ui.running) return;
  const result = ui.latestResult;
  const actionSelect = cachedNode("#actionSelect");
  const backendSelect = cachedNode("#backendSelect");
  const contextIsCurrent = Boolean(
    result
    && result.session_id === ui.activeRealtimeSessionId
    && result.run_id === ui.activeRealtimeRunId
    && result.action === actionSelect.value
    && result.request_backend === backendSelect.value
  );
  const displayAge = ui.lastResultAt > 0 ? Math.max(0, now - ui.lastResultAt) : Infinity;
  const maxAge = Number(ui.realtimeConfig.max_pose_age_ms || 150);
  const hideAfter = Math.max(maxAge, Number(ui.realtimeConfig.hide_pose_after_ms || 300));
  if (!contextIsCurrent || displayAge > hideAfter) {
    if (ui.floorCalibrationActive || ui.manualFloorPoints.length) {
      drawSkeleton({ keypoints: [], connections: [], assessment: {}, floor_reference: {} }, 1);
    } else {
      const canvas = cachedNode("#overlayCanvas");
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      canvas.hidden = true;
    }
    return;
  }
  const fadeAfter = Math.max(maxAge, hideAfter * 0.8);
  const opacity = displayAge <= fadeAfter
    ? 1
    : Math.max(0, 1 - (displayAge - fadeAfter) / Math.max(1, hideAfter - fadeAfter));
  drawingCache.predictionContext.action = result.action;
  drawingCache.predictionContext.phase = result.assessment?.phase || result.phase || "";
  drawingCache.predictionContext.supportFootNames = result.assessment?.support_foot_names || [];
  const prediction = displayPosePredictor.predict(
    finiteNumber(currentVideoFrame.expectedDisplayTime, now),
    drawingCache.predictionContext,
  );
  const renderStart = performance.now();
  drawSkeleton(result, opacity, renderStart, prediction.landmarks);
  const renderEnd = performance.now();
  if (result.frame_id > ui.lastAuditedPoseFrameId) {
    if (domUpdateScheduler.due("performance_snapshot", renderStart)) {
      drawingCache.performanceSnapshot = renderPerformance.snapshot();
    }
    const renderMetrics = drawingCache.performanceSnapshot;
    const timing = {
      ...(result.latency_timing || {}),
      pose_render_start_ms: renderStart,
      pose_render_end_ms: renderEnd,
      expected_display_time_ms: finiteNumber(currentVideoFrame.expectedDisplayTime, renderEnd),
      video_frame_presented_at_render_ms: finiteNumber(currentVideoFrame.presentationTime, renderStart),
      source_video_frame_id: finiteNumber(result.frame_meta?.frameId, result.frame_id),
      current_video_frame_id: finiteNumber(currentVideoFrame.frameId, 0),
      current_video_presented_frames: finiteNumber(currentVideoFrame.presentedFrames, 0),
      rvfc_lateness_ms: ui.rvfcLatenessMs,
      rvfc_late_frame_count: ui.rvfcLateFrameCount,
      presented_frame_skip_count: ui.presentedFrameSkipCount,
      main_thread_long_task_count: ui.longTaskCount,
      main_thread_long_task_duration_ms: Math.round(ui.longTaskDurationMs * 1000) / 1000,
      render_loop_p95_ms: renderMetrics.renderLoopP95Ms,
      canvas_draw_p95_ms: renderMetrics.canvasDrawP95Ms,
      dom_update_p95_ms: renderMetrics.domUpdateP95Ms,
      long_task_render_count: Number(renderMetrics.longTaskPhases.render || 0),
      long_task_dom_update_count: Number(renderMetrics.longTaskPhases.dom_update || 0),
      long_task_frame_copy_count: Number(renderMetrics.longTaskPhases.frame_copy || 0),
      long_task_encode_count: Number(renderMetrics.longTaskPhases.encode || 0),
      long_task_pose_transfer_count: Number(renderMetrics.longTaskPhases.pose_transfer || 0),
      long_task_other_count: Number(renderMetrics.longTaskPhases.other || 0),
    };
    result.latency_timing = timing;
    result.latency = deriveWebLatencies(timing);
    ui.lastAuditedPoseFrameId = result.frame_id;
    sendSocket({ type: "latency_audit", frame_id: result.frame_id, timing });
  }
}

// Drawing is driven by the camera frame callback.  These lifecycle functions
// remain as compatibility hooks for the existing start/stop flow.
function startPoseDrawLoop() {}
function stopPoseDrawLoop() {}

function drawSkeleton(result, opacity = 1, now = performance.now(), displayLandmarks = null) {
  const started = performance.now();
  try {
    return drawSkeletonCore(result, opacity, now, displayLandmarks);
  } finally {
    const ended = performance.now();
    renderPerformance.record("canvas_draw_ms", ended - started);
    renderPerformance.recordPhaseSpan("canvas_draw", started, ended);
  }
}

function drawSkeletonCore(result, opacity = 1, now = performance.now(), displayLandmarks = null) {
  const video = cachedNode("#localVideo");
  const canvas = cachedNode("#overlayCanvas");
  if (!ui.mediaStream || !video.videoWidth || !result) return;
  resizeOverlayIfNeeded();
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.globalAlpha = Math.max(0, Math.min(1, opacity));
  const contentRect = cachedVideoContentRect(canvas, video);
  const { drawWidth, drawHeight, offsetX, offsetY } = contentRect;
  const mirrored = cachedNode("#mirrorToggle").checked;
  drawingCache.pointPresent.fill(0);
  for (const point of displayLandmarks || result.keypoints || []) {
    const index = poseNameToIndex.get(point.name);
    if (index === undefined) continue;
    drawingCache.pointPresent[index] = 1;
    drawingCache.pointX[index] = offsetX + (mirrored ? 1 - point.x : point.x) * drawWidth;
    drawingCache.pointY[index] = offsetY + point.y * drawHeight;
    drawingCache.pointVisibility[index] = Number(point.visibility || 0);
  }
  // Prediction only contains the 33 pose landmarks. Keep the current hand
  // detections alongside the predicted body instead of dropping the fingers.
  if (displayLandmarks) {
    for (const point of result.keypoints || []) {
      const index = poseNameToIndex.get(point.name);
      if (index === undefined || index < poseLandmarkNames.length) continue;
      drawingCache.pointPresent[index] = 1;
      drawingCache.pointX[index] = offsetX + (mirrored ? 1 - point.x : point.x) * drawWidth;
      drawingCache.pointY[index] = offsetY + point.y * drawHeight;
      drawingCache.pointVisibility[index] = Number(point.visibility || 0);
    }
  }
  if (drawingCache.connectionSource !== result.connections) {
    drawingCache.connectionSource = result.connections;
    drawingCache.connectionCount = 0;
    const connections = result.connections?.length
      ? result.connections
      : poseConnectionNames;
    for (const [startName, endName] of connections) {
      const start = poseNameToIndex.get(startName);
      const end = poseNameToIndex.get(endName);
      if (
        start === undefined
        || end === undefined
        || drawingCache.connectionCount * 2 + 1 >= drawingCache.connectionPairs.length
      ) {
        continue;
      }
      const offset = drawingCache.connectionCount * 2;
      drawingCache.connectionPairs[offset] = start;
      drawingCache.connectionPairs[offset + 1] = end;
      drawingCache.connectionCount += 1;
    }
  }
  const styleKey = String(canvas.width);
  if (drawingCache.styleKey !== styleKey) {
    drawingCache.styleKey = styleKey;
    drawingCache.style.lineWidth = Math.max(2, canvas.width / 430);
    drawingCache.style.pointRadius = Math.max(3, canvas.width / 260);
    drawingCache.style.font = `${Math.max(11, canvas.width / 70)}px Inter, sans-serif`;
  }
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = drawingCache.style.lineWidth;
  const formStatus = result.assessment?.status || "unknown";
  const formColor = formStatus === "bad"
    ? "rgba(244, 62, 54, .96)"
    : formStatus === "good" ? "rgba(72, 222, 116, .96)" : "rgba(244, 190, 67, .94)";
  ctx.strokeStyle = formColor;
  for (let connectionIndex = 0; connectionIndex < drawingCache.connectionCount; connectionIndex += 1) {
    const offset = connectionIndex * 2;
    const start = drawingCache.connectionPairs[offset];
    const end = drawingCache.connectionPairs[offset + 1];
    if (
      !drawingCache.pointPresent[start]
      || !drawingCache.pointPresent[end]
      || drawingCache.pointVisibility[start] < 0.2
      || drawingCache.pointVisibility[end] < 0.2
    ) {
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(drawingCache.pointX[start], drawingCache.pointY[start]);
    ctx.lineTo(drawingCache.pointX[end], drawingCache.pointY[end]);
    ctx.stroke();
  }
  for (let index = 0; index < renderLandmarkNames.length; index += 1) {
    if (!drawingCache.pointPresent[index] || drawingCache.pointVisibility[index] < 0.2) continue;
    ctx.beginPath();
    ctx.arc(
      drawingCache.pointX[index],
      drawingCache.pointY[index],
      drawingCache.style.pointRadius,
      0,
      Math.PI * 2,
    );
    ctx.fillStyle = drawingCache.pointVisibility[index] >= 0.55 ? formColor : "#f0a023";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(255,255,255,.9)";
    ctx.stroke();
  }
  if (domUpdateScheduler.due("angles", now)) updateAngleDrawingCache(result.assessment?.angles || []);
  ctx.font = drawingCache.style.font;
  ctx.textBaseline = "bottom";
  for (let angleIndex = 0; angleIndex < drawingCache.angleCount; angleIndex += 1) {
    const angle = drawingCache.angleEntries[angleIndex];
    const anchor = angle.anchorIndex;
    if (!drawingCache.pointPresent[anchor] || drawingCache.pointVisibility[anchor] < 0.2) continue;
    const x = drawingCache.pointX[anchor];
    const y = drawingCache.pointY[anchor];
    const width = ctx.measureText(angle.label).width;
    ctx.fillStyle = "rgba(20,20,18,.72)";
    ctx.fillRect(x + 6, y - 22, width + 10, 20);
    ctx.fillStyle = angle.color;
    ctx.fillText(angle.label, x + 11, y - 5);
  }
  drawFloorReferenceOverlay(ctx, contentRect, result.floor_reference);
  ctx.restore();
  canvas.hidden = false;
}

function updateAngleDrawingCache(angles) {
  drawingCache.angleCount = 0;
  for (const angle of angles) {
    const anchorIndex = poseNameToIndex.get(angle.anchor);
    if (anchorIndex === undefined) continue;
    const index = drawingCache.angleCount;
    const entry = drawingCache.angleEntries[index] || {};
    entry.anchorIndex = anchorIndex;
    entry.label = `${angle.label} ${Math.round(angle.value)}° 3D`;
    entry.color = angle.status === "bad"
      ? "#ff5b50"
      : angle.status === "good" ? "#59e481" : "#f5f5ef";
    drawingCache.angleEntries[index] = entry;
    drawingCache.angleCount += 1;
  }
}

function cachedVideoContentRect(canvas, video) {
  const key = `${canvas.width}:${canvas.height}:${video.videoWidth}:${video.videoHeight}`;
  if (drawingCache.rectKey === key) return drawingCache.rect;
  drawingCache.rectKey = key;
  const stageRatio = canvas.width / canvas.height;
  const videoRatio = video.videoWidth / video.videoHeight;
  const rect = drawingCache.rect;
  rect.drawWidth = canvas.width;
  rect.drawHeight = canvas.height;
  rect.offsetX = 0;
  rect.offsetY = 0;
  if (videoRatio > stageRatio) {
    rect.drawHeight = canvas.width / videoRatio;
    rect.offsetY = (canvas.height - rect.drawHeight) / 2;
  } else {
    rect.drawWidth = canvas.height * videoRatio;
    rect.offsetX = (canvas.width - rect.drawWidth) / 2;
  }
  return rect;
}

function videoContentRect(canvas, video) {
  const stageRatio = canvas.width / canvas.height;
  const videoRatio = video.videoWidth / video.videoHeight;
  let drawWidth = canvas.width;
  let drawHeight = canvas.height;
  let offsetX = 0;
  let offsetY = 0;
  if (videoRatio > stageRatio) {
    drawHeight = canvas.width / videoRatio;
    offsetY = (canvas.height - drawHeight) / 2;
  } else {
    drawWidth = canvas.height * videoRatio;
    offsetX = (canvas.width - drawWidth) / 2;
  }
  return { drawWidth, drawHeight, offsetX, offsetY };
}

function drawFloorReferenceOverlay(ctx, rect, floorReference) {
  const mirrored = cachedNode("#mirrorToggle").checked;
  const line = floorReference?.line;
  const manual = ui.manualFloorPoints;
  const useManual = (ui.floorCalibrationActive && manual.length) || (!line && manual.length);
  const pointCount = useManual ? manual.length : line ? 2 : 0;
  if (!pointCount) return;

  ctx.save();
  ctx.strokeStyle = floorReference?.status === "UNSURE" ? "#f0a023" : "#c9ff38";
  ctx.fillStyle = ctx.strokeStyle;
  ctx.lineWidth = Math.max(2, ctx.canvas.width / 360);
  ctx.setLineDash([10, 7]);
  if (pointCount === 2) {
    const firstSourceX = useManual ? manual[0][0] : line.x1;
    const firstSourceY = useManual ? manual[0][1] : line.y1;
    const secondSourceX = useManual ? manual[1][0] : line.x2;
    const secondSourceY = useManual ? manual[1][1] : line.y2;
    const firstX = rect.offsetX + (mirrored ? 1 - firstSourceX : firstSourceX) * rect.drawWidth;
    const firstY = rect.offsetY + firstSourceY * rect.drawHeight;
    const secondX = rect.offsetX + (mirrored ? 1 - secondSourceX : secondSourceX) * rect.drawWidth;
    const secondY = rect.offsetY + secondSourceY * rect.drawHeight;
    ctx.beginPath();
    ctx.moveTo(firstX, firstY);
    ctx.lineTo(secondX, secondY);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  for (let index = 0; index < pointCount; index += 1) {
    const pointX = useManual ? manual[index][0] : index === 0 ? line.x1 : line.x2;
    const pointY = useManual ? manual[index][1] : index === 0 ? line.y1 : line.y2;
    const x = rect.offsetX + (mirrored ? 1 - pointX : pointX) * rect.drawWidth;
    const y = rect.offsetY + pointY * rect.drawHeight;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(5, ctx.canvas.width / 180), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function renderFloorCalibrationStatus(floorReference) {
  const status = cachedNode("#floorCalibrationStatus");
  if (!status || ui.floorCalibrationActive) return;
  let text;
  if (ui.manualFloorPoints.length === 2) {
    text = floorReference?.status === "UNSURE"
      ? "手动线已设置，但当前人体或脚部参考不可靠"
      : "手动地板线已启用";
  } else if (floorReference?.status === "READY") {
    text = `自动地板线已就绪 · 置信度 ${Math.round(Number(floorReference.confidence || 0) * 100)}%`;
  } else {
    text = "正在自动估计；请站直并保持双脚完整可见";
  }
  if (status.textContent !== text) status.textContent = text;
}

function beginFloorCalibration() {
  if (!ui.mediaStream || !$("#localVideo").videoWidth) {
    toast("请先开启本机摄像头预览", true);
    return;
  }
  ui.manualFloorPoints = [];
  ui.floorCalibrationActive = true;
  const canvas = $("#overlayCanvas");
  resizeOverlayIfNeeded();
  canvas.hidden = false;
  canvas.classList.add("floor-calibrating");
  $("#floorCalibrateButton").classList.add("active");
  $("#floorCalibrateButton").textContent = "点击两个点";
  $("#floorCalibrationStatus").textContent = "请依次点击脚下地板的两个相距较远的点";
}

function handleFloorCalibrationClick(event) {
  if (!ui.floorCalibrationActive) return;
  const canvas = $("#overlayCanvas");
  const video = $("#localVideo");
  const bounds = canvas.getBoundingClientRect();
  const scaleX = canvas.width / Math.max(bounds.width, 1);
  const scaleY = canvas.height / Math.max(bounds.height, 1);
  const x = (event.clientX - bounds.left) * scaleX;
  const y = (event.clientY - bounds.top) * scaleY;
  const rect = videoContentRect(canvas, video);
  if (
    x < rect.offsetX
    || x > rect.offsetX + rect.drawWidth
    || y < rect.offsetY
    || y > rect.offsetY + rect.drawHeight
  ) {
    toast("请点击实际视频画面内的地板", true);
    return;
  }
  const visualX = (x - rect.offsetX) / rect.drawWidth;
  const backendX = $("#mirrorToggle").checked ? 1 - visualX : visualX;
  ui.manualFloorPoints.push([
    Math.max(0, Math.min(1, backendX)),
    Math.max(0, Math.min(1, (y - rect.offsetY) / rect.drawHeight)),
  ]);
  if (ui.manualFloorPoints.length < 2) {
    $("#floorCalibrationStatus").textContent = "已记录第一个点，请点击另一侧地板";
    return;
  }
  if (Math.abs(ui.manualFloorPoints[1][0] - ui.manualFloorPoints[0][0]) <= 0.05) {
    ui.manualFloorPoints = [];
    $("#floorCalibrationStatus").textContent = "两点水平距离太近，请重新点击";
    return;
  }
  const floorSlope = Math.abs(
    (ui.manualFloorPoints[1][1] - ui.manualFloorPoints[0][1])
    / (ui.manualFloorPoints[1][0] - ui.manualFloorPoints[0][0])
  );
  if (floorSlope > 1) {
    ui.manualFloorPoints = [];
    $("#floorCalibrationStatus").textContent = "地板线倾斜过大，请重新点击";
    return;
  }
  ui.floorCalibrationActive = false;
  canvas.classList.remove("floor-calibrating");
  $("#floorCalibrateButton").classList.remove("active");
  $("#floorCalibrateButton").textContent = "重新标定";
  $("#floorCalibrationStatus").textContent = "手动地板线已设置";
  updateLiveSetting("manual_floor_points", ui.manualFloorPoints);
}

function resetFloorCalibration() {
  ui.manualFloorPoints = [];
  ui.floorCalibrationActive = false;
  $("#overlayCanvas").classList.remove("floor-calibrating");
  $("#floorCalibrateButton").classList.remove("active");
  $("#floorCalibrateButton").textContent = "两点标定";
  $("#floorCalibrationStatus").textContent = "已恢复自动估计；请站直并保持双脚完整可见";
  updateLiveSetting("manual_floor_points", []);
}

function resizeOverlayIfNeeded() {
  const canvas = $("#overlayCanvas");
  const stage = $("#videoStage");
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.round(stage.clientWidth * ratio));
  const height = Math.max(1, Math.round(stage.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
}

async function uploadSelectedVideo() {
  const file = $("#videoFile").files[0];
  if (!file) throw new Error("请先选择本地视频");
  if (file.size > 250 * 1024 * 1024) throw new Error("视频文件不能超过 250 MB");
  $("#startButton span:first-child").textContent = "正在上传…";
  const form = new FormData();
  form.append("video", file);
  const uploaded = await api("/api/upload", { method: "POST", body: form });
  ui.uploadId = uploaded.id;
  return uploaded.id;
}

async function startAnalysis() {
  resetVoiceSession();
  $("#reportPreview").hidden = true;
  $("#reportReadyBanner").hidden = true;
  $$("#downloadText, #downloadJson, #downloadCsv").forEach(link => link.setAttribute("aria-disabled", "true"));
  try {
    $("#startButton").disabled = true;
    $("#startButton span:first-child").textContent = "正在启动…";
    ui.manualStop = false;
    if (ui.sourceMode === "camera") {
      ui.latestResult = null;
      ui.latestAnalysisResult = null;
      displayPosePredictor.reset();
      ui.lastResultAt = 0;
      ui.lastRenderedPoseFrameId = -1;
      ui.lastDiscardedFrameId = -1;
      ui.staleResultCount = 0;
      ui.activeRealtimeSessionId = "";
      ui.activeRealtimeRunId = "";
      ui.poseWorkerLastCaptureAt = 0;
      ui.poseWorkerFailureCount = 0;
      ui.poseWorkerSlowFrameCount = 0;
      ui.poseWorkerDroppedFrames = 0;
      ui.poseWorkerBenchmarkSent = false;
      ui.poseWorker?.postMessage({ type: "reset" });
      if (!ui.mediaStream) await openCamera();
      $("#loadingOverlay").hidden = false;
      await initializeBrowserPoseWorker();
      setRunning(true);
      $("#loadingOverlay").hidden = false;
      $("#videoBadges").hidden = false;
      $("#progressTrack").hidden = true;
      await connectRealtime();
      return;
    }
    let videoId;
    if (ui.sourceMode === "sample") {
      const sample = ui.samplesByAction.get($("#actionSelect").value);
      if (!sample) throw new Error("当前动作没有可用的示例视频");
      videoId = sample.id;
    } else {
      videoId = await uploadSelectedVideo();
    }
    const payload = { source_mode: ui.sourceMode, video_id: videoId, ...settingsPayload() };
    const state = await api("/api/start", { method: "POST", body: JSON.stringify(payload) });
    setRunning(true);
    $("#emptyStage").hidden = true;
    $("#streamImage").hidden = false;
    $("#streamImage").src = `/api/stream?t=${Date.now()}`;
    $("#loadingOverlay").hidden = false;
    $("#videoBadges").hidden = false;
    $("#progressTrack").hidden = false;
    updateState(state);
    clearInterval(ui.stateTimer);
    ui.stateTimer = setInterval(pollState, 500);
  } catch (error) {
    setRunning(false);
    toast(error.message, true);
  } finally {
    $("#startButton span:first-child").textContent = ui.sourceMode === "camera" ? "开始实时分析" : "开始分析";
    if (!ui.running) $("#startButton").disabled = false;
  }
}

async function stopAnalysis() {
  ui.manualStop = true;
  cancelVoiceFeedback();
  clearTimeout(ui.reconnectTimer);
  stopLocalRecording();
  try {
    if (ui.sourceMode === "camera") {
      sendSocket({ type: "stop" });
      ui.socket?.close(1000, "user_stop");
      ui.socket = null;
      closeCamera();
      resetStage(true);
      $("#connectionMetric").textContent = "已断开";
      $("#connectionMetric").className = "neutral";
      toast("摄像头已停止并释放");
      if (ui.latestResult) setTimeout(() => generateReport({ auto: true }), 500);
    } else {
      const state = await api("/api/stop", { method: "POST", body: "{}" });
      updateState(state);
      resetStage(false);
      toast("分析已停止");
    }
  } catch (error) { toast(error.message, true); }
}

function resetStage(clearImage = true) {
  clearInterval(ui.stateTimer);
  ui.stateTimer = null;
  stopCaptureLoop();
  stopPoseDrawLoop();
  setRunning(false);
  $("#videoRepCount").textContent = "0";
  ui.paused = false;
  $("#pauseButton").classList.remove("active");
  $("#pauseIcon").textContent = "Ⅱ";
  $("#pauseButton small").textContent = "暂停";
  $("#recordButton").classList.remove("active");
  $("#recordButton small").textContent = "录制";
  if (clearImage) {
    $("#streamImage").src = "";
    $("#streamImage").hidden = true;
    if (!ui.mediaStream) $("#emptyStage").hidden = false;
    $("#videoBadges").hidden = true;
  }
  $("#loadingOverlay").hidden = true;
}

async function updateLiveSetting(key, value) {
  if (key === "mirror") {
    $("#localVideo").classList.toggle("mirrored", Boolean(value));
  }
  if (["action", "backend"].includes(key)) {
    ui.latestResult = null;
    ui.latestAnalysisResult = null;
    ui.lastRenderedPoseFrameId = -1;
    ui.poseWorker?.postMessage({ type: "reset" });
    displayPosePredictor.reset();
  }
  if (!ui.running) return;
  try {
    if (ui.sourceMode === "camera") sendSocket({ type: "settings", settings: { [key]: value } });
    else await api("/api/settings", { method: "POST", body: JSON.stringify({ [key]: value }) });
  } catch (error) { toast(error.message, true); }
}

async function togglePause() {
  ui.paused = !ui.paused;
  if (ui.paused) {
    cancelVoiceFeedback();
    ui.poseWorkerRequestedMeta = null;
    closePoseTransfer(ui.poseWorkerPending);
    ui.poseWorkerPending = null;
  }
  try {
    await updateLiveSetting("paused", ui.paused);
    $("#pauseButton").classList.toggle("active", ui.paused);
    $("#pauseIcon").textContent = ui.paused ? "▶" : "Ⅱ";
    $("#pauseButton small").textContent = ui.paused ? "继续" : "暂停";
  } catch (error) { ui.paused = !ui.paused; toast(error.message, true); }
}

function takeScreenshot() {
  const video = ui.sourceMode === "camera" ? $("#localVideo") : $("#streamImage");
  const overlay = $("#overlayCanvas");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || video.naturalWidth || 1280;
  canvas.height = video.videoHeight || video.naturalHeight || 720;
  const ctx = canvas.getContext("2d");
  if (ui.sourceMode === "camera" && $("#mirrorToggle").checked) { ctx.translate(canvas.width, 0); ctx.scale(-1, 1); }
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  if (ui.sourceMode === "camera" && $("#mirrorToggle").checked) ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (ui.sourceMode === "camera") ctx.drawImage(overlay, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `hyrox-screenshot-${new Date().toISOString().replaceAll(":", "-")}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast("截图已下载到当前设备，服务器未保存");
  }, "image/png");
}

function recordingMimeType() {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  return candidates.find(value => MediaRecorder.isTypeSupported(value)) || "";
}

function drawRecordingFrame() {
  if (!ui.recordingCanvas || !ui.recorder || ui.recorder.state === "inactive") return;
  const canvas = ui.recordingCanvas;
  const ctx = canvas.getContext("2d", { alpha: false });
  const source = ui.sourceMode === "camera" ? $("#localVideo") : $("#streamImage");
  const sourceWidth = source.videoWidth || source.naturalWidth || 0;
  const sourceHeight = source.videoHeight || source.naturalHeight || 0;
  ctx.fillStyle = "#171816";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (sourceWidth && sourceHeight) {
    const scale = Math.min(canvas.width / sourceWidth, canvas.height / sourceHeight);
    const width = sourceWidth * scale;
    const height = sourceHeight * scale;
    const x = (canvas.width - width) / 2;
    const y = (canvas.height - height) / 2;
    if (ui.sourceMode === "camera" && $("#mirrorToggle").checked) {
      ctx.save(); ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
      ctx.drawImage(source, canvas.width - x - width, y, width, height);
      ctx.restore();
    } else {
      ctx.drawImage(source, x, y, width, height);
    }
  }
  if (ui.sourceMode === "camera" && !$("#overlayCanvas").hidden) {
    ctx.drawImage($("#overlayCanvas"), 0, 0, canvas.width, canvas.height);
  }
  ui.recordingAnimation = requestAnimationFrame(drawRecordingFrame);
}

function startLocalRecording() {
  if (!ui.running) { toast("请先开始分析", true); return; }
  if (!window.MediaRecorder || !HTMLCanvasElement.prototype.captureStream) {
    toast("当前浏览器不支持本机画面录制", true); return;
  }
  const stage = $("#videoStage");
  const canvas = document.createElement("canvas");
  const scale = Math.min(2, 1280 / Math.max(1, stage.clientWidth));
  canvas.width = Math.max(320, Math.round(stage.clientWidth * scale));
  canvas.height = Math.max(180, Math.round(stage.clientHeight * scale));
  const mimeType = recordingMimeType();
  ui.recordingCanvas = canvas;
  ui.recordingChunks = [];
  const recorder = new MediaRecorder(canvas.captureStream(25), mimeType ? { mimeType, videoBitsPerSecond: 3_000_000 } : undefined);
  ui.recorder = recorder;
  recorder.ondataavailable = event => { if (event.data?.size) ui.recordingChunks.push(event.data); };
  recorder.onstop = () => {
    cancelAnimationFrame(ui.recordingAnimation);
    const blob = new Blob(ui.recordingChunks, { type: recorder.mimeType || "video/webm" });
    if (ui.recordingUrl) URL.revokeObjectURL(ui.recordingUrl);
    ui.recordingUrl = URL.createObjectURL(blob);
    $("#recordingPlayback").src = ui.recordingUrl;
    $("#downloadRecording").href = ui.recordingUrl;
    $("#downloadRecording").download = `hyrox-pose-${new Date().toISOString().replaceAll(":", "-")}.webm`;
    $("#recordingReview").hidden = false;
    $("#recordButton").classList.remove("active");
    $("#recordButton small").textContent = "录制";
    ui.recorder = null;
    ui.recordingCanvas = null;
    toast("带姿态节点的录像已生成，可直接回放或保存");
  };
  recorder.start(500);
  $("#recordButton").classList.add("active");
  $("#recordButton small").textContent = "结束";
  drawRecordingFrame();
  toast("正在本机录制带姿态节点的画面");
}

function stopLocalRecording() {
  if (ui.recorder && ui.recorder.state !== "inactive") ui.recorder.stop();
}

function toggleRecording() {
  if (ui.recorder && ui.recorder.state !== "inactive") stopLocalRecording();
  else startLocalRecording();
}

async function generateReport(options = {}) {
  const auto = options?.auto === true;
  const button = $("#generateReportButton");
  button.disabled = true;
  button.textContent = "正在生成…";
  try {
    const report = await api("/api/report");
    const analysis = report.analysis || {};
    const rate = analysis.compliance_rate == null ? "暂无" : `${analysis.compliance_rate}%`;
    $("#reportPreview").innerHTML = `
      <div class="report-heading"><div><small>本次训练结论</small><h3>${escapeHtml(report.summary?.action_label || "动作")} · ${escapeHtml(analysis.overall_status || "已完成")}</h3></div><strong>${rate}</strong></div>
      <div class="report-metrics"><span><b>${Number(report.summary?.candidate_count ?? report.summary?.reps ?? 0)}</b>完整周期</span><span><b>${Number(report.summary?.pose_valid_rep_count ?? report.summary?.reps ?? 0)}</b>有效动作</span><span><b>${Number(analysis.evaluable_frames || 0)}</b>可评价画面</span><span><b>${Number(analysis.nonstandard_frames || 0)}</b>明显偏离画面</span></div>
      <p class="report-explanation">${escapeHtml(analysis.compliance_explanation || "")}</p>
      <p class="report-download-tip">做得好的地方、优先改进建议和逐次动作表现已写入文字报告，请点击上方“文字报告”下载查看。</p>`;
    $("#reportPreview").hidden = false;
    if (!auto) $("#reportPreview").scrollIntoView({ behavior: "smooth", block: "center" });
    toast(auto ? "分析完成，文字报告已自动生成" : "分析报告已更新");
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.textContent = "重新生成";
    button.disabled = false;
  }
}

async function pollState() {
  try {
    const state = await api("/api/state");
    updateState(state);
    if (!state.running && ["completed", "error", "idle"].includes(state.status)) {
      clearInterval(ui.stateTimer);
      ui.stateTimer = null;
      setRunning(false);
      $("#videoRepBadge").hidden = state.status !== "completed";
      $("#loadingOverlay").hidden = true;
      if (state.status === "completed") {
        $$("#downloadText, #downloadJson, #downloadCsv").forEach(link => link.setAttribute("aria-disabled", "false"));
        $("#generateReportButton").disabled = false;
        $("#reportReadyBanner").hidden = false;
        toast("视频分析完成，上传文件已删除");
        await generateReport({ auto: true });
      }
      if (state.status === "error") toast(state.error || "分析出错", true);
    }
  } catch (_) {
    clearInterval(ui.stateTimer);
    toast("无法读取运行状态", true);
  }
}

function updateState(state) {
  const started = performance.now();
  try {
    const isStarting = state.status === "starting";
    setHiddenIfChanged("#loadingOverlay", !isStarting);
    setTextIfChanged("#topStatus", state.error || state.status_text || "系统就绪");
    setClassIfChanged(
      "#statusDot",
      `status-dot${state.status === "running" ? " live" : state.status === "error" ? " error" : ""}`,
    );
    setTextIfChanged(
      "#stageTitle",
      state.running || state.status === "completed"
        ? `${state.action_label || "动作"} · ${state.source_name}`
        : "准备开始训练分析",
    );
    setTextIfChanged("#sourceBadge", state.source_name || "本机画面");

    const now = performance.now();
    if (domUpdateScheduler.due("state_metrics", now)) {
      setTextIfChanged("#backendMetric", backendLabels[state.backend] || state.backend || "—");
      setTextIfChanged("#fpsMetric", Number(state.fps || 0).toFixed(1));
      setTextIfChanged("#latencyMetric", Number(state.inference_ms || 0).toFixed(1));
      setTextIfChanged(
        "#poseMetric",
        state.pose_detected ? "已锁定人体" : state.running ? "正在寻找人体" : "等待画面",
      );
      setClassIfChanged(
        "#poseMetric",
        state.pose_detected ? "good" : state.running ? "bad" : "neutral",
      );
    }
    if (ui.sourceMode !== "camera" && state.source_mode !== "browser-camera") {
      setConnectionMetricForSource(ui.sourceMode);
    }

    const candidateCount = state.candidate_count ?? state.reps ?? 0;
    if (domUpdateScheduler.due("state_stats", now)) {
      setTextIfChanged("#repCount", candidateCount);
      setTextIfChanged("#poseValidRepCount", state.pose_valid_rep_count ?? state.reps ?? 0);
      setTextIfChanged("#noRepCount", state.no_rep_count ?? 0);
      setTextIfChanged("#unsureCount", state.unsure_count ?? 0);
      renderFloorCalibrationStatus(state.floor_reference);
      setTextIfChanged("#videoRepCount", candidateCount);
      setHiddenIfChanged("#videoRepBadge", !(state.running || state.status === "completed"));
      setWidthIfChanged(
        "#phaseIndicator",
        state.pose_detected ? `${Math.min(100, 18 + ((state.frame_index || 0) % 5) * 19)}%` : "8%",
      );
      setWidthIfChanged("#progressBar", `${state.progress || 0}%`);
    }
    setTextIfChanged("#actionLabel", state.action_label || "动作指导关闭");
    const phase = state.phase || "idle";
    setTextIfChanged("#phaseValue", phaseLabels[phase] || phase.replaceAll("_", " "));
    renderCameraTip(state.action);
    speakVoiceFeedback(state.voice_feedback);
    renderFeedback(state.feedback || [], state.pose_detected, state.running);
    ui.lastStatus = state.status;
  } finally {
    const ended = performance.now();
    renderPerformance.record("dom_update_ms", ended - started);
    renderPerformance.recordPhaseSpan("dom_update", started, ended);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function renderFeedback(items, poseDetected, running) {
  const list = $("#feedbackList");
  let signature = `${Number(running)}:${Number(poseDetected)}`;
  for (const item of items) signature += `|${item.level}:${item.text}`;
  if (signature === ui.lastFeedbackSignature) return;
  ui.lastFeedbackSignature = signature;
  if (items.length) {
    list.innerHTML = items.map(item => `<div class="feedback-item ${escapeHtml(item.level)}"><strong>${item.level === "warn" ? "调整建议" : item.level === "error" ? "需要注意" : "动作提示"}</strong><p>${escapeHtml(item.text)}</p></div>`).join("");
  } else if (running && poseDetected) {
    list.innerHTML = `<div class="feedback-empty"><span>✓</span><p>当前动作稳定，继续保持完整幅度。</p></div>`;
  } else if (running) {
    list.innerHTML = `<div class="feedback-empty"><span>!</span><p>请站到画面中央，并确保头部到脚部完整入镜。</p></div>`;
  } else {
    list.innerHTML = `<div class="feedback-empty"><span>✓</span><p>开始后，这里会显示动作质量与调整建议。</p></div>`;
  }
  setTextIfChanged("#feedbackTime", running ? "实时更新" : "等待分析");
}

async function deleteCurrentSession() {
  if (!window.confirm("确定删除本次会话的上传文件和分析结果吗？此操作不可撤销。")) return;
  try {
    await api("/api/session", { method: "DELETE" });
    stopMediaTracks();
    location.reload();
  } catch (error) { toast(error.message, true); }
}

$$("#sourceTabs .segment").forEach(button => button.addEventListener("click", () => selectSource(button.dataset.source)));
$("#videoFile").addEventListener("change", event => {
  const file = event.target.files[0];
  $("#uploadTitle").textContent = file ? file.name : "选择一个视频";
  $("#uploadDetail").textContent = file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "MP4、MOV、AVI、MKV 或 WebM";
  ui.uploadId = "";
});
$("#openCameraButton").addEventListener("click", toggleCameraPreview);
$("#switchCameraButton").addEventListener("click", switchCamera);
$("#cameraDevice").addEventListener("change", event => openCamera({ deviceId: event.target.value }).catch(error => toast(error.message, true)));
$("#startButton").addEventListener("click", startAnalysis);
$("#stopButton").addEventListener("click", stopAnalysis);
$("#pauseButton").addEventListener("click", togglePause);
$("#recordButton").addEventListener("click", toggleRecording);
$("#screenshotButton").addEventListener("click", takeScreenshot);
$("#floorCalibrateButton").addEventListener("click", beginFloorCalibration);
$("#floorResetButton").addEventListener("click", resetFloorCalibration);
$("#overlayCanvas").addEventListener("click", handleFloorCalibrationClick);
$("#generateReportButton").addEventListener("click", generateReport);
$("#openReportButton").addEventListener("click", generateReport);
$("#deleteSessionButton").addEventListener("click", deleteCurrentSession);
$("#actionSelect").addEventListener("change", event => { resetVoiceSession(); renderStandards(event.target.value); renderCameraTip(event.target.value); updateLiveSetting("action", event.target.value); });
$("#viewSelect").addEventListener("change", event => updateLiveSetting("camera_view", event.target.value));
$("#sensitivitySelect").addEventListener("change", event => updateLiveSetting("sensitivity", event.target.value));
$("#profileSelect").addEventListener("change", () => updateLiveSetting("landmark_profile", selectedLandmarkProfile()));
$("#poseRuntimeSelect").addEventListener("change", event => {
  if (event.target.value === "server") {
    activateServerPoseFallback("用户选择服务器兼容姿态", false);
    toast("已选择服务器兼容姿态");
    return;
  }
  restartBrowserPoseWorker();
});
$("#poseModelSelect").addEventListener("change", event => {
  ui.poseWorkerModelPreference = event.target.value;
  if ($("#poseRuntimeSelect").value === "server") {
    toast("本机姿态档位将在切换到本机运行时生效");
    return;
  }
  restartBrowserPoseWorker();
});
$("#fingerToggle").addEventListener("change", event => {
  if (event.target.checked && ui.poseRuntimeMode === "local") {
    activateServerPoseFallback("手指关键点需要服务器兼容姿态");
  }
  updateLiveSetting("show_fingers", event.target.checked);
});
$("#faceToggle").addEventListener("change", () => updateLiveSetting("landmark_profile", selectedLandmarkProfile()));
$("#mirrorToggle").addEventListener("change", event => updateLiveSetting("mirror", event.target.checked));
window.addEventListener("resize", resizeOverlay);
window.addEventListener("pagehide", () => {
  ui.manualStop = true;
  cancelVoiceFeedback();
  stopCaptureLoop();
  stopPoseDrawLoop();
  stopLocalRecording();
  stopMediaTracks();
  ui.socket?.close();
  if (ui.recordingUrl) URL.revokeObjectURL(ui.recordingUrl);
});
document.addEventListener("visibilitychange", () => { if (document.hidden && ui.mediaStream) stopAnalysis(); });

initializeVoiceFeedback();
initializeMainThreadAudit();
loadOptions();
