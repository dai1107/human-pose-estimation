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
  realtimeConfig: { frame_width: 640, frame_height: 480, target_fps: 10 },
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
  captureFps: 10,
};

const phaseLabels = {
  idle: "等待开始", ready: "准备就绪", no_pose: "未检测到姿态", low_visibility: "可见度不足",
  stand: "站立", standing: "站立", squat_down: "下蹲", bottom: "最低点", drive: "发力",
  reset: "复位", recovery: "恢复", pull: "拉动", catch: "起始", finish: "完成",
  walking: "行走", airborne: "腾空", landing: "落地", support: "支撑",
};
const viewTips = {
  unknown: "选择实际拍摄视角后，可获得更准确的动作反馈。",
  front: "正面适合观察左右对称、膝部轨迹和站姿。",
  side: "侧面适合观察躯干角度、髋铰链和步幅。",
  front_left: "左前方兼顾身体对称与前后动作幅度。",
  front_right: "右前方兼顾身体对称与前后动作幅度。",
};

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${error ? " error" : ""}`;
  clearTimeout(node._timer);
  node._timer = setTimeout(() => node.className = "toast", 3200);
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
    $("#actionSelect").innerHTML = options.actions.map(item => `<option value="${item.value}">${item.label}</option>`).join("");
    $("#actionSelect").value = "lunge";
    $("#viewSelect").innerHTML = options.views.map(item => `<option value="${item.value}">${item.label}</option>`).join("");
    $("#viewSelect").value = "side";
    $("#sampleVideo").innerHTML = options.samples.length
      ? options.samples.map(item => `<option value="${item.id}">${item.name}</option>`).join("")
      : `<option value="">未找到示例视频</option>`;
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
    landmark_profile: $("#profileSelect").value,
    mirror: $("#mirrorToggle").checked,
    paused: ui.paused,
  };
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
}

function setRunning(running) {
  ui.running = running;
  $("#startButton").disabled = running;
  $("#stopButton").disabled = !running && !ui.mediaStream;
  $("#pauseButton").disabled = !running;
  $("#screenshotButton").disabled = !running;
  $$("#sourceTabs button, #sampleVideo, #videoFile, #backendSelect, #openCameraButton").forEach(node => { node.disabled = running; });
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
  const videoConstraint = deviceId
    ? { deviceId: { exact: deviceId }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 10, max: 15 } }
    : { facingMode: { ideal: facingMode }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 10, max: 15 } };
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
  stopCaptureLoop();
  stopMediaTracks();
  clearOverlay();
  $("#localVideo").hidden = true;
  $("#openCameraButton").textContent = "开启摄像头";
  $("#switchCameraButton").disabled = true;
  $("#cameraDevice").disabled = true;
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
        ui.captureFps = Math.max(5, ui.captureFps - 1);
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
  ui.captureFps = Math.min(ui.realtimeConfig.target_fps || 10, ui.captureFps || 10);
  const capture = async () => {
    const interval = Math.max(67, Math.round(1000 / ui.captureFps));
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
  if (roundTrip > 500) ui.captureFps = Math.max(5, ui.captureFps - 1);
  else if (roundTrip < 200 && result.sequence % 30 === 0) ui.captureFps = Math.min(ui.realtimeConfig.target_fps || 10, ui.captureFps + 1);
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
    feedback: result.feedback,
    frame_index: result.sequence,
  });
  drawSkeleton(result);
  $$("#downloadJson, #downloadCsv").forEach(link => link.setAttribute("aria-disabled", "false"));
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
  const mirrored = $("#mirrorToggle").checked;
  const points = new Map((result.keypoints || []).map(point => [point.name, point]));
  const xy = point => [offsetX + (mirrored ? 1 - point.x : point.x) * drawWidth, offsetY + point.y * drawHeight];
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = Math.max(2, canvas.width / 430);
  ctx.strokeStyle = "rgba(201, 255, 56, .92)";
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
    ctx.fillStyle = point.visibility >= 0.55 ? "#f04a23" : "#f0a023";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(255,255,255,.9)";
    ctx.stroke();
  }
  canvas.hidden = false;
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
    let videoId = ui.sourceMode === "sample" ? $("#sampleVideo").value : await uploadSelectedVideo();
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
  clearTimeout(ui.reconnectTimer);
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
  ui.paused = false;
  $("#pauseButton").classList.remove("active");
  $("#pauseIcon").textContent = "Ⅱ";
  $("#pauseButton small").textContent = "暂停";
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

async function pollState() {
  try {
    const state = await api("/api/state");
    updateState(state);
    if (!state.running && ["completed", "error", "idle"].includes(state.status)) {
      clearInterval(ui.stateTimer);
      ui.stateTimer = null;
      setRunning(false);
      $("#loadingOverlay").hidden = true;
      if (state.status === "completed") {
        $$("#downloadJson, #downloadCsv").forEach(link => link.setAttribute("aria-disabled", "false"));
        toast("视频分析完成，上传文件已删除");
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
  $("#backendMetric").textContent = state.backend === "yolo-pose" ? "YOLO Pose" : state.backend === "mediapipe" ? "MediaPipe" : state.backend || "—";
  $("#fpsMetric").textContent = Number(state.fps || 0).toFixed(1);
  $("#latencyMetric").textContent = Number(state.inference_ms || 0).toFixed(1);
  $("#poseMetric").textContent = state.pose_detected ? "已锁定人体" : state.running ? "正在寻找人体" : "等待画面";
  $("#poseMetric").className = state.pose_detected ? "good" : state.running ? "bad" : "neutral";
  $("#repCount").textContent = state.reps ?? 0;
  $("#actionLabel").textContent = state.action_label || "动作指导关闭";
  const phase = state.phase || "idle";
  $("#phaseValue").textContent = phaseLabels[phase] || phase.replaceAll("_", " ");
  $("#phaseIndicator").style.width = state.pose_detected ? `${Math.min(100, 18 + ((state.frame_index || 0) % 5) * 19)}%` : "8%";
  $("#progressBar").style.width = `${state.progress || 0}%`;
  $("#cameraTip").textContent = viewTips[state.camera_view] || viewTips.unknown;
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
$("#screenshotButton").addEventListener("click", takeScreenshot);
$("#deleteSessionButton").addEventListener("click", deleteCurrentSession);
$("#actionSelect").addEventListener("change", event => updateLiveSetting("action", event.target.value));
$("#viewSelect").addEventListener("change", event => { updateLiveSetting("camera_view", event.target.value); $("#cameraTip").textContent = viewTips[event.target.value]; });
$("#sensitivitySelect").addEventListener("change", event => updateLiveSetting("sensitivity", event.target.value));
$("#profileSelect").addEventListener("change", event => updateLiveSetting("landmark_profile", event.target.value));
$("#mirrorToggle").addEventListener("change", event => updateLiveSetting("mirror", event.target.checked));
window.addEventListener("resize", resizeOverlay);
window.addEventListener("pagehide", () => { ui.manualStop = true; stopMediaTracks(); ui.socket?.close(); });
document.addEventListener("visibilitychange", () => { if (document.hidden && ui.mediaStream) stopAnalysis(); });

loadOptions();
