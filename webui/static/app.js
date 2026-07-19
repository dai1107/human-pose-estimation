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
  realtimeConfig: { frame_width: 640, frame_height: 480, target_fps: 30, camera_fps: 60 },
  mediaStream: null,
  socket: null,
  captureTimer: null,
  reconnectTimer: null,
  reconnectAttempts: 0,
  manualStop: false,
  facingMode: "user",
  sequence: 0,
  pendingFrames: new Map(),
  latestResult: null,
  lastResultAt: 0,
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
};

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
  "yolo-rtmw-wholebody": "YOLO + RTMW WholeBody",
  "yolo-guided-mediapipe-fallback": "YOLO + MediaPipe（RTMW 降级）",
  "yolo-guided-mediapipe": "YOLO + MediaPipe",
  "yolo-pose": "YOLO Pose",
  mediapipe: "MediaPipe",
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
  const node = $("#cameraTip");
  if (!node) return;
  const tip = actionCameraTips[action] || actionCameraTips.none;
  node.textContent = `推荐视角：${tip.view}；入镜范围：${tip.framing}。`;
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
}

function setRunning(running) {
  ui.running = running;
  $("#videoRepBadge").hidden = !running;
  $("#startButton").disabled = running;
  $("#stopButton").disabled = !running && !ui.mediaStream;
  $("#pauseButton").disabled = !running;
  $("#recordButton").disabled = !running || !window.MediaRecorder;
  $("#screenshotButton").disabled = !running;
  $$("#sourceTabs button, #videoFile, #backendSelect, #openCameraButton").forEach(node => { node.disabled = running; });
  $("#switchCameraButton").disabled = !ui.mediaStream;
  $("#cameraDevice").disabled = !ui.mediaStream;
}

function setPermissionState(state, message) {
  const node = $("#cameraPermission");
  node.dataset.state = state;
  node.textContent = message;
}

function stopMediaTracks() {
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
  const requestedCameraFps = ui.realtimeConfig.camera_fps || 60;
  const videoConstraint = deviceId
    ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: requestedCameraFps, max: requestedCameraFps } }
    : { facingMode: { ideal: facingMode }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: requestedCameraFps, max: requestedCameraFps } };
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: false });
    ui.mediaStream = stream;
    const video = $("#localVideo");
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();
    video.hidden = false;
    video.classList.toggle("mirrored", $("#mirrorToggle").checked);
    $("#streamImage").hidden = true;
    $("#emptyStage").hidden = true;
    $("#videoBadges").hidden = false;
    $("#sourceBadge").textContent = stream.getVideoTracks()[0]?.label || "本机摄像头";
    const actualFps = Number(stream.getVideoTracks()[0]?.getSettings().frameRate || 0);
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
        sendSocket({ type: "start", settings: settingsPayload() });
        resolve();
      } else if (message.type === "started") {
        $("#loadingOverlay").hidden = false;
        startCaptureLoop();
      } else if (message.type === "result") {
        handleRealtimeResult(message);
      } else if (message.type === "state") {
        updateState(message.state);
      } else if (message.type === "frame_dropped") {
        ui.captureFps = Math.max(15, ui.captureFps - 2);
        $("#connectionMetric").textContent = "正在自适应";
        $("#connectionMetric").className = "neutral";
      } else if (message.type === "error") {
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
      $("#connectionMetric").textContent = "连接中断";
      $("#connectionMetric").className = "bad";
      if (ui.running && !ui.manualStop) scheduleReconnect();
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
  const capture = async () => {
    const interval = Math.max(34, Math.ceil(1000 / ui.captureFps));
    if (!ui.running || !ui.mediaStream || ui.paused || ui.socket?.readyState !== WebSocket.OPEN) {
      ui.captureTimer = setTimeout(capture, interval);
      return;
    }
    if (ui.socket.bufferedAmount > 1024 * 1024) {
      ui.captureTimer = setTimeout(capture, interval);
      return;
    }
    const video = $("#localVideo");
    if (video.readyState >= 2 && video.videoWidth > 0) {
      const canvas = $("#captureCanvas");
      const maxWidth = ui.realtimeConfig.frame_width || 640;
      const maxHeight = ui.realtimeConfig.frame_height || 480;
      const scale = Math.min(1, maxWidth / video.videoWidth, maxHeight / video.videoHeight);
      canvas.width = Math.max(2, Math.round(video.videoWidth * scale));
      canvas.height = Math.max(2, Math.round(video.videoHeight * scale));
      canvas.getContext("2d", { alpha: false }).drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.72));
      if (blob && blob.size <= (ui.realtimeConfig.max_frame_bytes || 512 * 1024)) {
        const sequence = ++ui.sequence;
        const image = new Uint8Array(await blob.arrayBuffer());
        const packet = new Uint8Array(4 + image.length);
        new DataView(packet.buffer).setUint32(0, sequence, false);
        packet.set(image, 4);
        ui.pendingFrames.set(sequence, performance.now());
        for (const [key] of ui.pendingFrames) if (key < sequence - 50) ui.pendingFrames.delete(key);
        ui.socket.send(packet.buffer);
      }
    }
    ui.captureTimer = setTimeout(capture, interval);
  };
  capture();
}

function stopCaptureLoop() {
  clearTimeout(ui.captureTimer);
  ui.captureTimer = null;
  ui.pendingFrames.clear();
}

function handleRealtimeResult(result) {
  ui.latestResult = result;
  ui.lastResultAt = performance.now();
  const sentAt = ui.pendingFrames.get(result.sequence);
  const roundTrip = sentAt ? performance.now() - sentAt : result.metrics.server_ms;
  ui.pendingFrames.delete(result.sequence);
  let quality = "流畅";
  let qualityClass = "good";
  if (roundTrip > 500) { quality = "网络较慢"; qualityClass = "bad"; }
  else if (roundTrip > 250) { quality = "一般"; qualityClass = "neutral"; }
  if (roundTrip > 500) ui.captureFps = Math.max(15, ui.captureFps - 2);
  else if (roundTrip < 200 && result.sequence % 30 === 0) ui.captureFps = Math.min(ui.realtimeConfig.target_fps || 30, ui.captureFps + 1);
  $("#connectionMetric").textContent = `${quality} · ${Math.round(roundTrip)} ms`;
  $("#connectionMetric").className = qualityClass;
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
  drawSkeleton(result);
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
  if (ui.latestResult) drawSkeleton(ui.latestResult);
}

function clearOverlay() {
  const canvas = $("#overlayCanvas");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  canvas.hidden = true;
}

function drawSkeleton(result) {
  const video = $("#localVideo");
  const canvas = $("#overlayCanvas");
  if (!ui.mediaStream || !video.videoWidth || !result) return;
  resizeOverlayIfNeeded();
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const { drawWidth, drawHeight, offsetX, offsetY } = videoContentRect(canvas, video);
  const mirrored = $("#mirrorToggle").checked;
  const points = new Map((result.keypoints || []).map(point => [point.name, point]));
  const xy = point => [offsetX + (mirrored ? 1 - point.x : point.x) * drawWidth, offsetY + point.y * drawHeight];
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = Math.max(2, canvas.width / 430);
  const formStatus = result.assessment?.status || "unknown";
  const formColor = formStatus === "bad"
    ? "rgba(244, 62, 54, .96)"
    : formStatus === "good" ? "rgba(72, 222, 116, .96)" : "rgba(244, 190, 67, .94)";
  ctx.strokeStyle = formColor;
  for (const [startName, endName] of result.connections || []) {
    const start = points.get(startName);
    const end = points.get(endName);
    if (!start || !end || start.visibility < 0.2 || end.visibility < 0.2) continue;
    const [x1, y1] = xy(start);
    const [x2, y2] = xy(end);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  for (const point of points.values()) {
    if (point.visibility < 0.2) continue;
    const [x, y] = xy(point);
    ctx.beginPath();
    ctx.arc(x, y, Math.max(3, canvas.width / 260), 0, Math.PI * 2);
    ctx.fillStyle = point.visibility >= 0.55 ? formColor : "#f0a023";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(255,255,255,.9)";
    ctx.stroke();
  }
  ctx.font = `${Math.max(11, canvas.width / 70)}px Inter, sans-serif`;
  ctx.textBaseline = "bottom";
  for (const angle of result.assessment?.angles || []) {
    const point = points.get(angle.anchor);
    if (!point || point.visibility < 0.2) continue;
    const [x, y] = xy(point);
    const label = `${angle.label} ${Math.round(angle.value)}°`;
    const color = angle.status === "bad" ? "#ff5b50" : angle.status === "good" ? "#59e481" : "#f5f5ef";
    const width = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(20,20,18,.72)";
    ctx.fillRect(x + 6, y - 22, width + 10, 20);
    ctx.fillStyle = color;
    ctx.fillText(label, x + 11, y - 5);
  }
  drawFloorReferenceOverlay(ctx, { drawWidth, drawHeight, offsetX, offsetY }, result.floor_reference);
  canvas.hidden = false;
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
  const mirrored = $("#mirrorToggle").checked;
  const backendToCanvas = point => [
    rect.offsetX + (mirrored ? 1 - point[0] : point[0]) * rect.drawWidth,
    rect.offsetY + point[1] * rect.drawHeight,
  ];
  let points = null;
  const line = floorReference?.line;
  if (ui.floorCalibrationActive && ui.manualFloorPoints.length) points = ui.manualFloorPoints;
  else if (line) points = [[line.x1, line.y1], [line.x2, line.y2]];
  else if (ui.manualFloorPoints.length) points = ui.manualFloorPoints;
  if (!points?.length) return;

  const canvasPoints = points.map(backendToCanvas);
  ctx.save();
  ctx.strokeStyle = floorReference?.status === "UNSURE" ? "#f0a023" : "#c9ff38";
  ctx.fillStyle = ctx.strokeStyle;
  ctx.lineWidth = Math.max(2, ctx.canvas.width / 360);
  ctx.setLineDash([10, 7]);
  if (canvasPoints.length === 2) {
    ctx.beginPath();
    ctx.moveTo(...canvasPoints[0]);
    ctx.lineTo(...canvasPoints[1]);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const [x, y] of canvasPoints) {
    ctx.beginPath();
    ctx.arc(x, y, Math.max(5, ctx.canvas.width / 180), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function renderFloorCalibrationStatus(floorReference) {
  const status = $("#floorCalibrationStatus");
  if (!status || ui.floorCalibrationActive) return;
  if (ui.manualFloorPoints.length === 2) {
    status.textContent = floorReference?.status === "UNSURE"
      ? "手动线已设置，但当前人体或脚部参考不可靠"
      : "手动地板线已启用";
  } else if (floorReference?.status === "READY") {
    status.textContent = `自动地板线已就绪 · 置信度 ${Math.round(Number(floorReference.confidence || 0) * 100)}%`;
  } else {
    status.textContent = "正在自动估计；请站直并保持双脚完整可见";
  }
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
  drawSkeleton(ui.latestResult || { keypoints: [], connections: [], assessment: {} });
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
  drawSkeleton(ui.latestResult || { keypoints: [], connections: [], assessment: {} });
  if (ui.manualFloorPoints.length < 2) {
    $("#floorCalibrationStatus").textContent = "已记录第一个点，请点击另一侧地板";
    return;
  }
  if (Math.abs(ui.manualFloorPoints[1][0] - ui.manualFloorPoints[0][0]) <= 0.05) {
    ui.manualFloorPoints = [];
    $("#floorCalibrationStatus").textContent = "两点水平距离太近，请重新点击";
    drawSkeleton(ui.latestResult || { keypoints: [], connections: [], assessment: {} });
    return;
  }
  const floorSlope = Math.abs(
    (ui.manualFloorPoints[1][1] - ui.manualFloorPoints[0][1])
    / (ui.manualFloorPoints[1][0] - ui.manualFloorPoints[0][0])
  );
  if (floorSlope > 1) {
    ui.manualFloorPoints = [];
    $("#floorCalibrationStatus").textContent = "地板线倾斜过大，请重新点击";
    drawSkeleton(ui.latestResult || { keypoints: [], connections: [], assessment: {} });
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
  drawSkeleton(ui.latestResult || { keypoints: [], connections: [], assessment: {} });
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
      if (!ui.mediaStream) await openCamera();
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
    if (ui.latestResult) drawSkeleton(ui.latestResult);
  }
  if (!ui.running) return;
  try {
    if (ui.sourceMode === "camera") sendSocket({ type: "settings", settings: { [key]: value } });
    else await api("/api/settings", { method: "POST", body: JSON.stringify({ [key]: value }) });
  } catch (error) { toast(error.message, true); }
}

async function togglePause() {
  ui.paused = !ui.paused;
  if (ui.paused) cancelVoiceFeedback();
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
  const isStarting = state.status === "starting";
  $("#loadingOverlay").hidden = !isStarting;
  $("#topStatus").textContent = state.error || state.status_text || "系统就绪";
  $("#statusDot").className = `status-dot${state.status === "running" ? " live" : state.status === "error" ? " error" : ""}`;
  $("#stageTitle").textContent = state.running || state.status === "completed" ? `${state.action_label || "动作"} · ${state.source_name}` : "准备开始训练分析";
  $("#sourceBadge").textContent = state.source_name || "本机画面";
  $("#backendMetric").textContent = backendLabels[state.backend] || state.backend || "—";
  $("#fpsMetric").textContent = Number(state.fps || 0).toFixed(1);
  $("#latencyMetric").textContent = Number(state.inference_ms || 0).toFixed(1);
  $("#poseMetric").textContent = state.pose_detected ? "已锁定人体" : state.running ? "正在寻找人体" : "等待画面";
  $("#poseMetric").className = state.pose_detected ? "good" : state.running ? "bad" : "neutral";
  const candidateCount = state.candidate_count ?? state.reps ?? 0;
  $("#repCount").textContent = candidateCount;
  $("#poseValidRepCount").textContent = state.pose_valid_rep_count ?? state.reps ?? 0;
  $("#noRepCount").textContent = state.no_rep_count ?? 0;
  $("#unsureCount").textContent = state.unsure_count ?? 0;
  renderFloorCalibrationStatus(state.floor_reference);
  $("#videoRepCount").textContent = candidateCount;
  $("#videoRepBadge").hidden = !(state.running || state.status === "completed");
  $("#actionLabel").textContent = state.action_label || "动作指导关闭";
  const phase = state.phase || "idle";
  $("#phaseValue").textContent = phaseLabels[phase] || phase.replaceAll("_", " ");
  $("#phaseIndicator").style.width = state.pose_detected ? `${Math.min(100, 18 + ((state.frame_index || 0) % 5) * 19)}%` : "8%";
  $("#progressBar").style.width = `${state.progress || 0}%`;
  renderCameraTip(state.action);
  speakVoiceFeedback(state.voice_feedback);
  renderFeedback(state.feedback || [], state.pose_detected, state.running);
  ui.lastStatus = state.status;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function renderFeedback(items, poseDetected, running) {
  const list = $("#feedbackList");
  if (items.length) {
    list.innerHTML = items.map(item => `<div class="feedback-item ${escapeHtml(item.level)}"><strong>${item.level === "warn" ? "调整建议" : item.level === "error" ? "需要注意" : "动作提示"}</strong><p>${escapeHtml(item.text)}</p></div>`).join("");
  } else if (running && poseDetected) {
    list.innerHTML = `<div class="feedback-empty"><span>✓</span><p>当前动作稳定，继续保持完整幅度。</p></div>`;
  } else if (running) {
    list.innerHTML = `<div class="feedback-empty"><span>!</span><p>请站到画面中央，并确保头部到脚部完整入镜。</p></div>`;
  } else {
    list.innerHTML = `<div class="feedback-empty"><span>✓</span><p>开始后，这里会显示动作质量与调整建议。</p></div>`;
  }
  $("#feedbackTime").textContent = running ? "实时更新" : "等待分析";
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
$("#fingerToggle").addEventListener("change", event => updateLiveSetting("show_fingers", event.target.checked));
$("#faceToggle").addEventListener("change", () => updateLiveSetting("landmark_profile", selectedLandmarkProfile()));
$("#mirrorToggle").addEventListener("change", event => updateLiveSetting("mirror", event.target.checked));
window.addEventListener("resize", resizeOverlay);
window.addEventListener("pagehide", () => { ui.manualStop = true; cancelVoiceFeedback(); stopLocalRecording(); stopMediaTracks(); ui.socket?.close(); if (ui.recordingUrl) URL.revokeObjectURL(ui.recordingUrl); });
document.addEventListener("visibilitychange", () => { if (document.hidden && ui.mediaStream) stopAnalysis(); });

initializeVoiceFeedback();
loadOptions();
