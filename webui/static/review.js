"use strict";

const QUICK_ROLE = "a";
const QUICK_REVIEWER = "quick_reviewer";
const state = {
  bootstrap: null,
  task: "core",
  records: [],
  currentId: null,
  currentModality: null,
  record: null,
  meta: null,
  review: null,
  proposal: null,
  revision: 0,
  dirty: false,
  filter: "all",
  track: [],
  oniCheckpoints: [],
  disagreementClipIndex: 0,
  clipLoop: false,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const ERROR_LABELS = {
  NO_ERROR: "无错误",
  FOOT_DESYNCHRONIZED: "双脚起跳或落地不同步",
  HANDS_FEET_TOO_FAR: "手脚距离过远",
  NO_CHEST_CONTACT: "胸部未触地",
  EXTRA_STEP: "出现额外补步或碎步",
  NO_KNEE_CONTACT: "后膝未触地",
  SAME_LEG_CONSECUTIVE: "同一条腿连续迈步",
  HIP_NOT_EXTENDED: "髋部未完全伸展",
  NOT_DEEP_ENOUGH: "下蹲深度不足",
  HEEL_RISE: "脚跟抬起",
  ARM_NOT_EXTENDED_VIOLATION: "手臂未充分伸展",
  ARM_NOT_BY_SIDE_VIOLATION: "手臂未保持在身体两侧",
  LEAN_LEFT_RIGHT: "身体左右倾斜",
  TORSO_LEAN: "躯干前后倾斜",
  HANDLE_AROUND_KNEES: "桨把绕膝",
  TOO_MUCH_BACK_LEAN: "结束阶段身体后仰过多",
  EARLY_ARM_PULL: "手臂拉动过早",
  ARMS_NOT_HIGH_ENOUGH: "回到顶部时手未充分上举",
  NO_HIP_HINGE: "下拉时髋部折叠不足",
  TOO_MUCH_SQUAT: "下拉时下蹲过多",
  ASYMMETRIC_PULL: "左右拉动不对称",
  RUSHED_RETURN: "回程过快",
  TORSO_TOO_UPRIGHT: "推雪橇时身体过直",
  TORSO_TOO_LOW: "推雪橇时身体压得过低",
  SHORT_STEPS: "步幅过小",
  NO_LEG_DRIVE: "腿部驱动不足",
  HIP_TOO_HIGH_OR_BACK_ROUND: "髋部过高或背部不稳定",
  SLED_PULL_KNEELING_VIOLATION: "拉雪橇时跪姿",
  SLED_PULL_SEATED_VIOLATION: "拉雪橇时坐姿",
  NOT_STANDING: "拉雪橇时未保持站立",
  OVER_LEAN_BACK: "拉雪橇时后仰过多",
  ARMS_ONLY_PULL: "只用手臂拉动",
  NO_CLEAR_PULL: "没有清晰拉动",
  OTHER: "其他错误（请备注）",
  UNSURE: "无法确认",
};
const PHASE_LABELS = {
  hands_down: "手撑地",
  chest_down: "胸部下降/触地",
  takeoff: "起跳",
  flight: "腾空",
  landing: "落地",
  stabilization: "落地稳定",
  descent: "下降",
  bottom: "最低点",
  contact: "后膝接触",
  ascent: "站起",
  stand: "完全伸展",
  release: "出球",
  recovery: "接球/恢复",
  ready: "准备",
  carrying: "持续行走",
  rest: "停止/休息",
  catch: "划船起始",
  drive: "蹬腿拉动",
  finish: "划船结束",
  top: "顶部",
  pull_down: "下拉",
  return: "回程",
  setup: "准备姿态",
  step: "蹬地步",
  reset: "重置",
  reach: "前伸",
  pull: "拉动",
  recover: "恢复",
};
const EVENT_LABELS = {
  hands_down: "双手撑地",
  chest_contact_candidate: "胸部触地",
  left_takeoff: "左脚起跳",
  right_takeoff: "右脚起跳",
  takeoff_candidate: "起跳",
  left_landing: "左脚落地",
  right_landing: "右脚落地",
  landing_candidate: "落地",
  stabilized: "落地稳定",
  rep_start: "本次开始",
  bottom_reached: "到达最低点",
  rear_knee_contact_candidate: "后膝接触",
  full_extension: "完全伸展",
  ball_release_candidate: "出球",
  recovery: "接球/恢复",
  monitor_start: "监控区间开始",
  carry_start: "行走开始",
  carry_stop: "行走停止",
  monitor_end: "监控区间结束",
  catch_reached: "到达划船起始",
  drive_start: "驱动开始",
  finish_reached: "到达划船结束",
  recovery_end: "恢复结束",
  top_reached: "到达顶部",
  pull_start: "拉动开始",
  return_end: "回程结束",
  step_contact: "蹬地/落脚",
  drive_end: "驱动结束",
  reach_reached: "到达前伸",
  pull_finish: "拉动结束",
};
const SEGMENT_LABELS = {
  target_action: "目标动作",
  setup: "准备",
  idle: "静止/等待",
  transition: "过渡",
  unknown_motion: "未知动作",
  target_out_of_frame: "目标离开画面",
};
const VALIDITY_LABELS = {VALID: "有效", NO_REP: "不构成一次", UNSURE: "无法确认"};
const OBSERVABILITY_LABELS = {
  OBSERVABLE: "可观察",
  PARTIAL: "部分可观察",
  UNOBSERVABLE: "不可观察",
  UNSURE: "无法确认",
  UNKNOWN: "未知",
};

const getPath = (object, path, fallback = undefined) => (
  path.split(".").reduce((value, key) => (value == null ? undefined : value[key]), object) ?? fallback
);
function setPath(object, path, value) {
  const keys = path.split(".");
  let target = object;
  keys.slice(0, -1).forEach((key) => {
    if (!target[key] || typeof target[key] !== "object") target[key] = {};
    target = target[key];
  });
  target[keys.at(-1)] = value;
}
function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}
function labelledOptions(values, labels) {
  return values.map((value) => ({value, label: labels[value] || value}));
}
function currentAction() {
  return getPath(state.review, "quick_review.action", state.record?.action_candidate || "unknown_ood");
}
function phaseOptions() {
  return labelledOptions(state.bootstrap.labels.phases_by_action[currentAction()] || [], PHASE_LABELS);
}
function errorOptions() {
  const values = ["NO_ERROR", ...(state.bootstrap.labels.errors_by_action[currentAction()] || []), "OTHER", "UNSURE"];
  return labelledOptions([...new Set(values)], ERROR_LABELS);
}
function eventOptions() {
  return labelledOptions(state.bootstrap.labels.events_by_action[currentAction()] || [], EVENT_LABELS);
}

const LISTS = {
  segments: {
    path: "quick_review.segments",
    fields: () => [
      ["label", "区间类型", "select", labelledOptions(Object.keys(SEGMENT_LABELS), SEGMENT_LABELS)],
      ["start_frame", "开始帧", "number"],
      ["end_frame", "结束帧", "number"],
      ["notes", "说明", "text", null, "wide"],
    ],
  },
  reps: {
    path: "quick_review.reps",
    fields: () => [
      ["rep_id", "次数编号", "text"],
      ["start_frame", "开始帧", "number"],
      ["end_frame", "结束帧", "number"],
      ["validity", "本次结论", "select", labelledOptions(Object.keys(VALIDITY_LABELS), VALIDITY_LABELS)],
      ["notes", "说明", "text", null, "wide"],
      ["phase_gap_reason", "阶段空白原因", "text", null, "wide"],
    ],
  },
  phaseIntervals: {
    path: "quick_review.phase_error_intervals",
    fields: () => [
      ["rep_id", "所属次数", "text"],
      ["start_frame", "开始帧", "number"],
      ["end_frame", "结束帧", "number"],
      ["phase", "所属阶段", "select", phaseOptions()],
      ["error_code", "出现错误", "select", errorOptions()],
      ["observability", "可见性", "select", labelledOptions(Object.keys(OBSERVABILITY_LABELS), OBSERVABILITY_LABELS)],
      ["notes", "说明", "text", null, "wide"],
    ],
  },
  events: {
    path: "quick_review.events",
    fields: () => [
      ["rep_id", "所属次数", "text"],
      ["event_type", "关键事件", "select", eventOptions()],
      ["frame_index", "帧号", "number"],
      ["observability", "可见性", "select", labelledOptions(Object.keys(OBSERVABILITY_LABELS), OBSERVABILITY_LABELS)],
      ["notes", "说明", "text", null, "wide"],
    ],
  },
};

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove("show"), 2800);
}
async function api(url, options = {}) {
  const headers = {...(options.headers || {})};
  if (options.body) headers["Content-Type"] = "application/json";
  if (state.bootstrap?.csrf_token && !["GET", "HEAD"].includes((options.method || "GET").toUpperCase())) {
    headers["X-CSRF-Token"] = state.bootstrap.csrf_token;
  }
  const response = await fetch(url, {...options, headers});
  const payload = (response.headers.get("content-type") || "").includes("application/json") ? await response.json() : null;
  if (!response.ok) throw new Error(payload?.error || `请求失败 (${response.status})`);
  return payload;
}

function emptyReview(record) {
  return {
    quick_review: {
      status: "draft",
      fine_annotation_status: "draft",
      action: record.action_candidate || "unknown_ood",
      target_status: "correct",
      video_usability: "usable",
      overall_result: "UNSURE",
      observability: "UNKNOWN",
      subject_id: record.subject_id_suggestion || "",
      dataset_role: record.dataset_role_suggestion || "unassigned",
      authorization: "confirmed",
      usable_start_frame: 0,
      usable_end_frame: Number(record.video.decoded_frame_count) - 1,
      segments: [],
      reps: [],
      phase_error_intervals: [],
      events: [],
      disagreement_clips: structuredClone(record.disagreement_clips || []),
      proposal_decision: "unresolved",
      equipment_visibility: "unknown",
      notes: "",
    },
  };
}
function mergeDisagreementClips(sourceClips, savedClips) {
  const savedById = new Map((savedClips || []).map((item) => [item.clip_id, item]));
  return (sourceClips || []).map((clip) => ({
    ...structuredClone(clip),
    ...(savedById.get(clip.clip_id) || {}),
    clip_id: clip.clip_id,
    start_frame: clip.start_frame,
    end_frame: clip.end_frame,
    anchor_frames: structuredClone(clip.anchor_frames || []),
  }));
}
function emptyOniReview(checkpoints, mode = "subject") {
  const frames = checkpoints.map((item) => Number(item.source_frame_index));
  return {
    review_mode: mode,
    status: "draft",
    overall_target_status: "unsure",
    same_subject_throughout: "unsure",
    observability: "UNSURE",
    confirmed_view: "unsure",
    action_usability: "unsure",
    usable_start_frame: frames.length ? Math.min(...frames) : null,
    usable_end_frame: frames.length ? Math.max(...frames) : null,
    full_body_visibility: "unsure",
    floor_visibility: "unsure",
    equipment_visibility: "unsure",
    identity_switch_intervals: [],
    observability_items: [],
    batch_edits: [],
    checkpoints: checkpoints.map((item) => ({
      frame_index: item.source_frame_index,
      target_status: "unreviewed",
      bbox_status: "unreviewed",
      surface_reliable: "unsure",
      notes: "",
    })),
    notes: "",
  };
}

async function initialize() {
  state.bootstrap = await api("/api/review/bootstrap");
  await api("/api/review/session", {
    method: "POST",
    body: JSON.stringify({role: QUICK_ROLE, reviewer_id: QUICK_REVIEWER}),
  });
  renderActionOptions();
  switchTask("remaining");
}
function renderActionOptions() {
  $("#actionSelect").innerHTML = Object.entries(state.bootstrap.labels.actions)
    .map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join("");
}
function taskRecords() {
  if (state.task === "oni") return state.bootstrap.oni_records || [];
  if (state.task === "view") return state.bootstrap.view_prior_records || [];
  if (state.task === "error") return state.bootstrap.error_truth_records || [];
  if (state.task === "core") return (state.bootstrap.records || []).filter((record) => record.core);
  if (state.task === "remaining") return (state.bootstrap.records || []).filter((record) => !record.core);
  if (state.task === "disagreement") return (state.bootstrap.records || []).filter((record) => record.disagreement_clip_count > 0);
  return [];
}
function isOniTask(task = state.task) {
  return ["oni", "view", "error"].includes(task);
}
function recordStatus(record) {
  if (isOniTask()) return record;
  if (state.task === "disagreement") return record.disagreement_review || {saved: false, complete: false};
  const review = record.reviewer_a || {saved: false, fine_complete: false};
  return {...review, complete: Boolean(review.fine_complete)};
}
function switchTask(task) {
  if (state.dirty && !window.confirm("当前有未保存修改，仍要切换任务吗？")) return;
  state.task = task;
  state.currentId = null;
  state.currentModality = null;
  state.record = null;
  state.review = null;
  state.dirty = false;
  state.records = taskRecords();
  $$(".task-switcher button").forEach((button) => button.classList.toggle("active", button.dataset.task === task));
  $("#queueTitle").textContent = task === "core" ? "15 段核心动作"
    : task === "remaining" ? "其余 15 段手机 RGB"
      : task === "disagreement" ? "高分歧短片段" : "64 个模态任务";
  $("#queueEyebrow").textContent = task === "core" ? "已确认核心精标"
    : task === "remaining" ? "人工 4 · 逐次/阶段/事件/错误"
      : task === "disagreement" ? "人工 4 · 0.5–1.0 秒复核"
        : task === "oni" ? "Depth / IR 主体复核" : task === "view" ? "视角 × 可观察性" : "录制意图错误裁决";
  $("#phoneReviewContent").hidden = true;
  $("#oniReviewContent").hidden = true;
  $("#emptyState").hidden = false;
  renderRecordList();
  updateProgress();
  renderDashboard();
}
function renderDashboard() {
  const dashboard = state.bootstrap.dashboard || {};
  const taskRows = dashboard.task_completion || [];
  const taskName = state.task === "core" ? "phone_rgb_fine_review"
    : state.task === "remaining" ? "phone_rgb_remaining_fine_review"
      : state.task === "disagreement" ? "phone_rgb_disagreement_review"
    : state.task === "oni" ? "oni_subject_review"
      : state.task === "view" ? "oni_view_prior_review" : "oni_error_truth_review";
  const current = taskRows.find((item) => item.task === taskName);
  const actionGroups = (dashboard.action_view_modality || [])
    .filter((item) => state.task !== "view" || item.complete < item.total)
    .slice(0, 4);
  $("#dashboardSummary").innerHTML = `
    ${current ? `<div class="dashboard-current"><strong>${escapeHtml(current.label)}</strong><span>${current.complete}/${current.total}</span><i><b style="width:${current.total ? current.complete / current.total * 100 : 0}%"></b></i></div>` : ""}
    <div class="dashboard-tasks">${taskRows.map((item) => `<span><em>${escapeHtml(item.label)}</em><b>${item.complete}/${item.total}</b></span>`).join("")}</div>
    ${state.task === "view" ? `<div class="dashboard-groups">${actionGroups.map((item) => `<span>${escapeHtml(item.action_label)} · ${escapeHtml(item.view_label)} · ${item.modality.toUpperCase()} <b>${item.complete}/${item.total}</b></span>`).join("")}</div>` : ""}`;
}
function renderRecordList() {
  const query = $("#recordSearch").value.trim().toLowerCase();
  const actionFilter = $("#actionFilter").value;
  const records = state.records.filter((record) => {
    const status = recordStatus(record);
    if (state.filter === "pending" && status.complete) return false;
    if (state.filter === "complete" && !status.complete) return false;
    if (state.filter === "low_confidence" && !record.low_confidence) return false;
    if (state.filter === "subject_switch" && !record.subject_switch) return false;
    if (state.filter === "conflict" && !record.conflict) return false;
    if (actionFilter !== "all" && record.action_candidate !== actionFilter) return false;
    return `${record.record_id} ${record.action_label} ${record.modality_label || ""}`.toLowerCase().includes(query);
  });
  $("#recordList").innerHTML = records.map((record, index) => {
    const status = recordStatus(record);
    const itemId = isOniTask() ? record.task_id : record.record_id;
    const disagreementMeta = state.task === "disagreement"
      ? ` · ${status.completed_clip_count || 0}/${status.clip_count || 0} 段`
      : "";
    return `<button class="quick-record ${itemId === currentTaskId() ? "active" : ""}" type="button" data-id="${escapeHtml(record.record_id)}" data-modality="${escapeHtml(record.modality || "")}">
      <span class="order">${String(index + 1).padStart(2, "0")}</span>
      <span><strong>${escapeHtml(record.action_label)}</strong><small>${escapeHtml(record.record_id)}${record.modality_label ? ` · ${escapeHtml(record.modality_label)}` : ""}${escapeHtml(disagreementMeta)}</small></span>
      <i class="record-dot ${status.complete ? "complete" : status.saved ? "saved" : ""}"></i>
    </button>`;
  }).join("");
  $$(".quick-record").forEach((button) => button.addEventListener("click", () => {
    if (isOniTask()) selectOniRecord(button.dataset.id, button.dataset.modality);
    else selectRecord(button.dataset.id);
  }));
}
function currentTaskId() {
  return state.currentModality ? `${state.currentId}__${state.currentModality}` : state.currentId;
}
function updateProgress() {
  const completed = state.records.filter((record) => recordStatus(record).complete).length;
  const total = state.records.length;
  $("#completeCount").textContent = `${completed}/${total}`;
  const taskLabel = state.task === "core" ? "核心动作精标"
    : state.task === "remaining" ? "人工 4 · 其余 RGB 精标"
      : state.task === "disagreement" ? "人工 4 · 高分歧片段"
        : state.task === "oni" ? "ONI 主体复核" : state.task === "view" ? "视角先验复核" : "ONI 错误真值复核";
  $("#topProgressText").textContent = `${taskLabel} · ${completed}/${total} 已完成`;
  $("#topProgressBar").style.width = `${total ? completed / total * 100 : 0}%`;
}

async function selectRecord(recordId) {
  if (state.dirty && !window.confirm("当前有未保存修改，仍要切换记录吗？")) return;
  const payload = await api(`/api/review/records/${encodeURIComponent(recordId)}?role=${QUICK_ROLE}&reviewer_id=${QUICK_REVIEWER}&quick=1`);
  state.currentId = recordId;
  state.currentModality = null;
  state.record = payload.record;
  state.meta = state.records.find((record) => record.record_id === recordId);
  state.proposal = payload.proposal;
  state.revision = payload.saved_review?.revision || 0;
  state.review = structuredClone(payload.saved_review?.review || emptyReview(payload.record));
  if (!state.review.quick_review) state.review = emptyReview(payload.record);
  if (!state.review.quick_review.phase_error_intervals) state.review.quick_review.phase_error_intervals = [];
  if (!state.review.quick_review.fine_annotation_status) {
    state.review.quick_review.fine_annotation_status = payload.record.core && state.review.quick_review.status === "complete"
      ? "complete" : "draft";
    if (state.task === "remaining") state.review.quick_review.status = "draft";
  }
  if (!state.review.quick_review.subject_id || state.review.quick_review.subject_id === "subject_pending") {
    state.review.quick_review.subject_id = payload.record.subject_id_suggestion;
  }
  if (!state.review.quick_review.dataset_role) {
    state.review.quick_review.dataset_role = payload.record.dataset_role_suggestion;
  }
  state.review.quick_review.disagreement_clips = mergeDisagreementClips(
    payload.record.disagreement_clips,
    state.review.quick_review.disagreement_clips,
  );
  state.disagreementClipIndex = Math.max(
    0,
    state.review.quick_review.disagreement_clips.findIndex((clip) => clip.review_status === "pending"),
  );
  state.dirty = false;
  state.track = [];
  renderRecord();
  renderRecordList();
}
function renderRecord() {
  $("#emptyState").hidden = true;
  $("#oniReviewContent").hidden = true;
  $("#phoneReviewContent").hidden = false;
  $("#recordAction").textContent = state.record.action_label;
  $("#recordTitle").textContent = state.record.record_id;
  $("#recordMeta").textContent = `${state.record.video.decoded_frame_count} 帧 · ${Number(state.record.video.fps).toFixed(2)} FPS · ${state.record.camera_view_current || "未知视角"}`;
  $("#reviewVideo").src = state.record.video_url;
  $("#phoneChangeReason").value = "";
  setSaveIndicator(state.revision ? `已保存 · r${state.revision}` : "未保存");
  fillForm();
  renderDisagreementPanel();
  renderProposal();
  renderSheets();
}
function fillForm() {
  $$("[data-path]", $("#quickForm")).forEach((input) => { input.value = getPath(state.review, input.dataset.path, ""); });
  Object.entries(LISTS).forEach(([name, config]) => renderList(name, getPath(state.review, config.path, [])));
}
function inputValue(input) {
  if (input.type === "number") return input.value === "" ? null : Number(input.value);
  return input.value;
}
function collectForm() {
  const review = structuredClone(state.review);
  $$("[data-path]", $("#quickForm")).forEach((input) => setPath(review, input.dataset.path, inputValue(input)));
  Object.entries(LISTS).forEach(([name, config]) => setPath(review, config.path, collectList(name)));
  if (["core", "remaining"].includes(state.task)) {
    review.quick_review.fine_annotation_status = review.quick_review.status;
  }
  return review;
}
function createRowField(definition, value) {
  const [key, label, type, options, className] = definition;
  const wrapper = document.createElement("label");
  if (className) wrapper.className = className;
  wrapper.append(document.createTextNode(label));
  let input;
  if (type === "select") {
    input = document.createElement("select");
    (options || []).forEach((option) => {
      const normalized = typeof option === "string" ? {value: option, label: option} : option;
      const element = document.createElement("option");
      element.value = normalized.value;
      element.textContent = normalized.label;
      input.append(element);
    });
  } else {
    input = document.createElement("input");
    input.type = type === "number" ? "number" : "text";
    if (type === "number") { input.step = "1"; input.min = "0"; }
  }
  input.dataset.key = key;
  input.value = value ?? "";
  input.addEventListener("input", markDirty);
  input.addEventListener("change", markDirty);
  wrapper.append(input);
  return wrapper;
}
function renderList(name, values) {
  const container = $(`[data-list="${name}"]`);
  container.innerHTML = "";
  (values || []).forEach((value) => addRow(name, value, false));
}
function addRow(name, value = {}, dirty = true) {
  if (["phaseIntervals", "events"].includes(name) && !(LISTS[name].fields()[3]?.[3] || []).length) {
    toast("请先在“确认动作”中选择波比跳远、负重箭步蹲或墙球", true);
    return;
  }
  const row = $("#rowTemplate").content.firstElementChild.cloneNode(true);
  LISTS[name].fields().forEach((field) => row.querySelector(".row-fields").append(createRowField(field, value[field[0]])));
  row.querySelector(".remove-button").addEventListener("click", () => { row.remove(); markDirty(); });
  $(`[data-list="${name}"]`).append(row);
  if (dirty) markDirty();
}
function collectList(name) {
  return $$(".simple-row", $(`[data-list="${name}"]`)).map((row) => {
    const value = {};
    $$("[data-key]", row).forEach((input) => { value[input.dataset.key] = inputValue(input); });
    return value;
  });
}
function markDirty() {
  if (!state.currentId) return;
  state.dirty = true;
  if (isOniTask()) setOniSaveIndicator("有未保存修改", "dirty");
  else setSaveIndicator("有未保存修改", "dirty");
}
function setSaveIndicator(text, className = "") {
  $("#saveIndicator").textContent = text;
  $("#saveIndicator").className = `save-indicator ${className}`.trim();
}
function setOniSaveIndicator(text, className = "") {
  $("#oniSaveIndicator").textContent = text;
  $("#oniSaveIndicator").className = `save-indicator ${className}`.trim();
}
function validate(review) {
  // 保存是无条件草稿操作；质量检查仅用于后续提示，不阻止当前进度落盘。
  return [];
}
async function save({next = false} = {}) {
  if (!state.currentId) return;
  const review = collectForm();
  const errors = validate(review);
  if (errors.length) return toast(errors[0], true);
  setSaveIndicator("正在保存…", "dirty");
  try {
    const response = await api(`/api/review/records/${encodeURIComponent(state.currentId)}`, {
      method: "PUT",
      body: JSON.stringify({
        role: QUICK_ROLE, reviewer_id: QUICK_REVIEWER, base_revision: state.revision, review,
        change_reason: $("#phoneChangeReason").value.trim() || "single human core fine annotation initial review",
        evidence_frames: (review.quick_review.events || []).map((item) => item.frame_index).filter((value) => value != null),
        finish_review: review.quick_review.fine_annotation_status === "complete" || review.quick_review.status === "complete",
      }),
    });
    state.review = review;
    state.revision = response.revision;
    state.dirty = false;
    $("#phoneChangeReason").value = "";
    setSaveIndicator(`已保存 · r${state.revision}`, "saved");
    toast(review.quick_review.fine_annotation_status === "complete" ? "人工 4 精标已完成" : "草稿已保存");
    await refreshRecords();
    if (next) selectNextRecord();
  } catch (error) {
    setSaveIndicator("保存失败", "error");
    toast(error.message, true);
  }
}

function currentDisagreementClip() {
  if (state.task !== "disagreement") return null;
  return state.review?.quick_review?.disagreement_clips?.[state.disagreementClipIndex] || null;
}
function disagreementStatusComplete(clip) {
  return clip && clip.review_status && clip.review_status !== "pending";
}
function renderDisagreementPanel() {
  const panel = $("#disagreementPanel");
  const clips = state.review?.quick_review?.disagreement_clips || [];
  const visible = state.task === "disagreement" && clips.length > 0;
  panel.hidden = !visible;
  if (!visible) return;
  state.disagreementClipIndex = Math.max(0, Math.min(clips.length - 1, state.disagreementClipIndex));
  const clip = currentDisagreementClip();
  const completed = clips.filter(disagreementStatusComplete).length;
  $("#clipProgressText").textContent = `${completed}/${clips.length} 段已复核 · 覆盖 ${clips.reduce((sum, item) => sum + Number(item.anchor_frame_count || 0), 0)} 个高分歧帧`;
  $("#clipProgressBar").style.width = `${clips.length ? completed / clips.length * 100 : 0}%`;
  $("#clipCounter").textContent = `片段 ${state.disagreementClipIndex + 1}/${clips.length}`;
  $("#clipWindow").textContent = `${clip.start_frame}–${clip.end_frame} 帧 · ${Number(clip.duration_seconds).toFixed(2)} 秒`;
  $("#clipAnchors").textContent = `锚点：${(clip.anchor_frames || []).join("、")}`;
  $("#clipReviewStatus").value = clip.review_status || "pending";
  $("#clipEvidenceType").value = clip.evidence_type || "unspecified";
  $("#clipObservability").value = clip.observability || "UNKNOWN";
  $("#clipNotes").value = clip.notes || "";
  $("#clipLoopToggle").checked = state.clipLoop;
}
function updateCurrentDisagreementClip() {
  const clip = currentDisagreementClip();
  if (!clip) return;
  clip.review_status = $("#clipReviewStatus").value;
  clip.evidence_type = $("#clipEvidenceType").value;
  clip.observability = $("#clipObservability").value;
  clip.notes = $("#clipNotes").value;
  markDirty();
  renderDisagreementPanel();
}
function selectDisagreementClip(offset) {
  const clips = state.review?.quick_review?.disagreement_clips || [];
  if (!clips.length) return;
  state.disagreementClipIndex = (state.disagreementClipIndex + offset + clips.length) % clips.length;
  renderDisagreementPanel();
  const clip = currentDisagreementClip();
  if (clip) setFrame(clip.anchor_frames?.[0] ?? clip.start_frame);
}
function playCurrentDisagreementClip() {
  const clip = currentDisagreementClip();
  if (!clip) return;
  setFrame(clip.start_frame);
  $("#reviewVideo").play().catch(() => {});
}
function addEventFromCurrentClip() {
  const clip = currentDisagreementClip();
  const options = eventOptions();
  if (!clip || !options.length) return toast("当前动作没有可选关键事件", true);
  const frame = clip.anchor_frames?.[0] ?? clip.start_frame;
  const reps = collectList("reps");
  const rep = reps.find((item) => frame >= item.start_frame && frame <= item.end_frame);
  if (!rep) return toast("请先添加包含该锚点的动作次数/分析周期", true);
  addRow("events", {
    rep_id: rep.rep_id,
    event_type: options[0].value,
    frame_index: frame,
    observability: clip.observability || "UNKNOWN",
    notes: `来自高分歧短片段 ${clip.clip_id}，请人工选择正确事件类型`,
  });
  toast("已把锚点加入关键事件，请选择正确事件类型");
}

function proposalSegments() {
  return (state.proposal?.action_segments?.segments || []).map((item) => ({
    label: item.label || "target_action", start_frame: item.start_frame, end_frame: item.end_frame, notes: "AI 候选，已由人工载入",
  }));
}
function proposalReps() {
  return (state.proposal?.core_annotations?.reps || []).map((rep, index) => ({
    rep_id: rep.rep_id || `rep_${String(index + 1).padStart(3, "0")}`,
    start_frame: rep.start_frame,
    end_frame: rep.end_frame,
    validity: rep.validity || "UNSURE",
    notes: rep.notes || "AI 候选，待人工核对",
    phase_gap_reason: rep.phase_gap_reason || "",
  }));
}
function proposalPhaseIntervals() {
  const output = [];
  (state.proposal?.core_annotations?.reps || []).forEach((rep) => {
    (rep.phases || []).forEach((phase) => {
      const matching = (rep.errors || []).find((error) => error.phase === phase.phase);
      output.push({
        rep_id: rep.rep_id, start_frame: phase.start_frame, end_frame: phase.end_frame, phase: phase.phase,
        error_code: matching?.error_code || "NO_ERROR", observability: "UNKNOWN",
        notes: phase.notes || "AI 候选，待人工核对",
      });
    });
  });
  return output;
}
function proposalEvents() {
  const output = [];
  (state.proposal?.core_annotations?.reps || []).forEach((rep) => {
    (rep.events || []).forEach((event) => output.push({
      rep_id: rep.rep_id, event_type: event.event_type, frame_index: event.frame_index, observability: "UNKNOWN", notes: "AI 候选，待人工核对",
    }));
  });
  return output;
}
function renderProposalRep(rep, index) {
  const errors = (rep.errors || []).map((item) => ERROR_LABELS[item.error_code] || item.error_code).join("、") || "未提出错误";
  return `<article class="ai-rep-card"><header><div><strong>候选第 ${index + 1} 次</strong><span>${rep.start_frame}–${rep.end_frame} 帧</span></div></header><div class="ai-error-list">${escapeHtml(errors)}</div></article>`;
}
function renderProposal() {
  const reps = state.proposal?.core_annotations?.reps || [];
  const segments = state.proposal?.action_segments?.segments || [];
  $("#proposalSummary").innerHTML = [
    ["动作区间", segments.length], ["候选次数", reps.length], ["人工状态", state.revision ? "已有草稿" : "未保存"],
  ].map(([label, value]) => `<div class="proposal-card"><span>${label}</span><strong>${value}</strong></div>`).join("");
  $("#proposalAdvice").innerHTML = reps.length
    ? `<div class="ai-advice-section"><h3>逐次 AI proposal <small>仅供人工参考</small></h3><div class="ai-rep-list">${reps.map(renderProposalRep).join("")}</div></div>`
    : `<div class="ai-advice-empty"><strong>当前没有逐次 AI proposal</strong><span>请直接人工添加逐次边界、阶段和事件。</span></div>`;
  $("#proposalJson").textContent = JSON.stringify(state.proposal, null, 2);
}
function importProposal(kind) {
  if (!state.proposal) return toast("当前没有可载入的 AI 候选", true);
  const quick = state.review.quick_review;
  const segments = proposalSegments();
  const reps = proposalReps();
  const phaseIntervals = proposalPhaseIntervals();
  const events = proposalEvents();
  if ((kind === "segments" || kind === "all") && !segments.length) {
    if (kind === "segments") return toast("当前没有动作区间候选", true);
  }
  if ((kind === "reps" || kind === "all") && !reps.length) {
    if (kind === "reps") return toast("当前算法没有检出候选次数/分析周期", true);
  }
  if ((kind === "events" || kind === "all") && !events.length) {
    if (kind === "events") return toast("当前没有关键帧候选", true);
  }
  if (kind === "all" || kind === "segments") quick.segments = segments;
  if (kind === "all" || kind === "reps") {
    quick.reps = reps;
    quick.phase_error_intervals = phaseIntervals;
  }
  if (kind === "all" || kind === "events") quick.events = events;
  fillForm();
  markDirty();
  const loaded = [
    (kind === "all" || kind === "segments") && segments.length ? `${segments.length} 个动作区间` : "",
    (kind === "all" || kind === "reps") && reps.length ? `${reps.length} 个候选次数/周期` : "",
    (kind === "all" || kind === "events") && events.length ? `${events.length} 个关键帧` : "",
  ].filter(Boolean).join("、");
  toast(`${loaded || "AI 候选"}已载入草稿，请逐项人工核对`);
}
function renderSheets() {
  const entries = Object.entries(state.record.sheet_urls || {});
  $("#sheetButtons").innerHTML = entries.map(([kind, url], index) =>
    `<button class="${index === 0 ? "active" : ""}" type="button" data-url="${escapeHtml(url)}">${kind.toUpperCase()}</button>`).join("");
  if (entries.length) $("#sheetImage").src = entries[0][1];
  else { $("#sheetImage").removeAttribute("src"); $("#sheetButtons").textContent = "没有审阅图"; }
  $$("#sheetButtons button").forEach((button) => button.addEventListener("click", () => {
    $$("#sheetButtons button").forEach((item) => item.classList.toggle("active", item === button));
    $("#sheetImage").src = button.dataset.url;
  }));
}

async function selectOniRecord(recordId, modality) {
  if (state.dirty && !window.confirm("当前有未保存修改，仍要切换记录吗？")) return;
  const reviewMode = state.task === "view" ? "view_prior" : state.task === "error" ? "error_truth" : "subject";
  const payload = await api(`/api/review/oni/${encodeURIComponent(recordId)}/${encodeURIComponent(modality)}?mode=${reviewMode}`);
  state.currentId = recordId;
  state.currentModality = modality;
  state.record = payload.record;
  state.meta = state.records.find((record) => record.task_id === `${recordId}__${modality}`);
  state.oniCheckpoints = payload.checkpoints;
  state.revision = payload.saved_review?.revision || 0;
  const defaults = emptyOniReview(payload.checkpoints, reviewMode);
  const savedReview = payload.saved_review?.review || {};
  state.review = {...defaults, ...structuredClone(savedReview)};
  state.review.review_mode = reviewMode;
  state.review.checkpoints = structuredClone(savedReview.checkpoints || defaults.checkpoints);
  state.review.identity_switch_intervals = structuredClone(savedReview.identity_switch_intervals || []);
  const existingItems = new Map((savedReview.observability_items || []).map((item) => [item.item_code, item]));
  state.review.observability_items = (state.bootstrap.labels.observability_items_by_action[state.record.action_candidate] || [])
    .map((item) => ({
      item_code: item.item_code,
      status: "UNSURE",
      reason: "",
      start_frame: null,
      end_frame: null,
      evidence_frames: [],
      notes: "",
      ...(existingItems.get(item.item_code) || {}),
    }));
  const existingErrorItems = new Map((savedReview.error_truth_items || []).map((item) => [item.error_code, item]));
  state.review.error_truth_items = (state.record.expected_errors_unverified || []).map((errorCode) => ({
    error_code: errorCode,
    truth_status: "unreviewed",
    observability: "UNSURE",
    evidence_frames: [],
    notes: "",
    ...(existingErrorItems.get(errorCode) || {}),
  }));
  state.dirty = false;
  renderOniRecord();
  renderRecordList();
}
function renderOniRecord() {
  $("#emptyState").hidden = true;
  $("#phoneReviewContent").hidden = true;
  $("#oniReviewContent").hidden = false;
  $("#oniRecordAction").textContent = `${state.record.action_label} · ${state.record.modality_label}`;
  $("#oniRecordTitle").textContent = state.record.record_id;
  $("#oniRecordMeta").textContent = `${state.record.checkpoint_count} 个检查点 · 原始视角 ${state.record.camera_view_label || "未知"} · 当前保存 ${state.record.modality_label}`;
  $$(".view-prior-only").forEach((element) => { element.hidden = state.task !== "view"; });
  $$(".error-truth-only").forEach((element) => { element.hidden = state.task !== "error"; });
  $("#oniPrimarySectionTitle").textContent = state.task === "view" ? "视角先验与可用区间" : state.task === "error" ? "错误真值的主体依据" : "模态主体结论";
  $("#oniPrimarySectionHelp").textContent = state.task === "view"
    ? "先确认视角、动作区间与证据可见性，不要求填写 rep 或完整错误标签"
    : state.task === "error"
      ? "先确认当前模态中的目标身份，再裁决录制意图错误是否真实发生"
      : "确认目标运动员、主体连续性与候选框；Depth/IR 结论独立保存";
  $("#oniDepthPreviewImage").src = state.record.preview_urls.depth;
  $("#oniIrPreviewImage").src = state.record.preview_urls.ir;
  $("#oniOriginalView").value = `${state.record.camera_view_label || "未知"}${state.record.camera_view_raw ? `（原标签：${state.record.camera_view_raw}）` : ""}`;
  $("#oniChangeReason").value = "";
  $("#confirmedViewSelect").innerHTML = Object.entries(state.bootstrap.labels.views)
    .map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join("");
  $$(".modality-switch button").forEach((button) => button.classList.toggle("active", button.dataset.modality === state.currentModality));
  const audit = state.record.modality_audit || {};
  $("#candidateAuditSummary").textContent = `当前 ${state.record.modality_label}：自动候选 ${audit.automated_candidate_count ?? 0}/${audit.sampled_checkpoint_count ?? state.record.checkpoint_count}，置信度中位数 ${Number(audit.confidence_p50 || 0).toFixed(3)}`;
  const loss = state.record.candidate_loss_intervals || [];
  $("#candidateLossSummary").textContent = loss.length
    ? `候选丢失区间：${loss.map((item) => `${item.start_frame}–${item.end_frame}`).join("、")}`
    : "候选丢失区间：未检测到";
  setOniSaveIndicator(state.revision ? `已保存 · r${state.revision}` : "未保存");
  $$("[data-oni-path]", $("#oniForm")).forEach((input) => { input.value = getPath(state.review, input.dataset.oniPath, ""); });
  renderIdentitySwitches();
  renderObservabilityMatrix();
  renderErrorTruthMatrix();
  renderOniCheckpoints();
  renderEligibility();
}
function oniOptions(values, selected) {
  return Object.entries(values).map(([value, label]) => `<option value="${value}" ${value === selected ? "selected" : ""}>${escapeHtml(label)}</option>`).join("");
}
function renderOniCheckpoints() {
  const byFrame = new Map((state.review.checkpoints || []).map((item) => [Number(item.frame_index), item]));
  const labels = state.bootstrap.labels.oni;
  $("#oniCheckpointList").innerHTML = state.oniCheckpoints.map((proposal, index) => {
    const frame = Number(proposal.source_frame_index);
    const row = byFrame.get(frame) || {frame_index: frame, target_status: "unreviewed", bbox_status: "unreviewed", surface_reliable: "unsure", notes: ""};
    const candidate = proposal.bbox_px ? `候选框 ${proposal.bbox_px.join(", ")}` : "没有候选框";
    return `<article class="oni-checkpoint" data-frame="${frame}" data-index="${index + 1}">
      <header><strong>${String(index + 1).padStart(2, "0")} · 原始帧 ${frame}</strong><span>${escapeHtml(candidate)} · 置信度 ${Number(proposal.confidence || 0).toFixed(3)}</span></header>
      <div class="oni-checkpoint-fields">
        <label>主体是谁<select data-key="target_status">${oniOptions(labels.target_status, row.target_status)}</select></label>
        <label>候选框<select data-key="bbox_status">${oniOptions(labels.bbox_status, row.bbox_status)}</select></label>
        <label>深度表面可靠<select data-key="surface_reliable">${oniOptions(labels.yes_no_unsure, row.surface_reliable)}</select></label>
        <label class="wide">备注<input data-key="notes" value="${escapeHtml(row.notes || "")}" placeholder="可留空"></label>
      </div>
    </article>`;
  }).join("");
  $$("#oniCheckpointList select, #oniCheckpointList input").forEach((input) => {
    input.addEventListener("input", markDirty);
    input.addEventListener("change", markDirty);
  });
  $$(".oni-checkpoint header").forEach((header) => header.addEventListener("click", () => syncCheckpoint(header.closest(".oni-checkpoint"))));
}
function syncCheckpoint(row) {
  $$(".oni-checkpoint").forEach((item) => item.classList.toggle("synced", item === row));
  const text = `原始帧 ${row.dataset.frame} · 检查点 ${row.dataset.index}`;
  $("#depthCursorLabel").textContent = text;
  $("#irCursorLabel").textContent = text;
}
function renderIdentitySwitches() {
  const rows = state.review.identity_switch_intervals || [];
  $("#identitySwitchList").innerHTML = rows.map((item, index) => `
    <div class="compact-interval" data-index="${index}">
      <input type="number" min="0" data-key="start_frame" value="${item.start_frame ?? ""}" placeholder="开始帧">
      <span>至</span>
      <input type="number" min="0" data-key="end_frame" value="${item.end_frame ?? ""}" placeholder="结束帧">
      <input data-key="notes" value="${escapeHtml(item.notes || "")}" placeholder="切换说明">
      <button type="button" title="删除">×</button>
    </div>`).join("");
  $$("#identitySwitchList input").forEach((input) => input.addEventListener("input", markDirty));
  $$("#identitySwitchList button").forEach((button) => button.addEventListener("click", () => {
    state.review.identity_switch_intervals.splice(Number(button.closest(".compact-interval").dataset.index), 1);
    renderIdentitySwitches();
    markDirty();
  }));
}
function collectIdentitySwitches() {
  return $$(".compact-interval", $("#identitySwitchList")).map((row) => {
    const value = {};
    $$("[data-key]", row).forEach((input) => { value[input.dataset.key] = inputValue(input); });
    return value;
  });
}
function renderObservabilityMatrix() {
  const labels = new Map((state.bootstrap.labels.observability_items_by_action[state.record.action_candidate] || [])
    .map((item) => [item.item_code, item.label]));
  $("#observabilityMatrix").innerHTML = (state.review.observability_items || []).map((item) => `
    <article class="observability-row" data-code="${escapeHtml(item.item_code)}">
      <header><strong>${escapeHtml(labels.get(item.item_code) || item.item_code)}</strong><span>${escapeHtml(item.item_code)}</span></header>
      <div>
        <label>状态<select data-key="status">${oniOptions({
          UNSURE: "无法确认", OBSERVABLE: "可观察", PARTIAL: "部分可观察", UNOBSERVABLE: "不可观察",
        }, item.status)}</select></label>
        <label>不可观察原因<input data-key="reason" value="${escapeHtml(item.reason || "")}" placeholder="不可观察时必填"></label>
        <label>适用开始帧<input type="number" min="0" data-key="start_frame" value="${item.start_frame ?? ""}"></label>
        <label>适用结束帧<input type="number" min="0" data-key="end_frame" value="${item.end_frame ?? ""}"></label>
        <label>证据帧<input data-key="evidence_frames" value="${escapeHtml((item.evidence_frames || []).join(", "))}" placeholder="如 21, 41"></label>
        <label>备注<input data-key="notes" value="${escapeHtml(item.notes || "")}"></label>
      </div>
    </article>`).join("");
  $$("#observabilityMatrix input, #observabilityMatrix select").forEach((input) => {
    input.addEventListener("input", markDirty);
    input.addEventListener("change", markDirty);
  });
}
function collectObservabilityItems() {
  return $$(".observability-row", $("#observabilityMatrix")).map((row) => {
    const value = {item_code: row.dataset.code};
    $$("[data-key]", row).forEach((input) => {
      value[input.dataset.key] = input.dataset.key === "evidence_frames"
        ? input.value.split(/[,，\s]+/).filter(Boolean).map(Number)
        : inputValue(input);
    });
    return value;
  });
}
function renderErrorTruthMatrix() {
  $("#errorTruthMatrix").innerHTML = (state.review.error_truth_items || []).map((item) => `
    <article class="observability-row error-truth-row" data-code="${escapeHtml(item.error_code)}">
      <header><strong>${escapeHtml(ERROR_LABELS[item.error_code] || item.error_code)}</strong><span>${escapeHtml(item.error_code)}</span></header>
      <div>
        <label>错误是否发生<select data-key="truth_status">${oniOptions({
          unreviewed: "尚未复核", confirmed: "确认发生", rejected: "确认未发生", unsure: "无法判断",
        }, item.truth_status)}</select></label>
        <label>可观察性<select data-key="observability">${oniOptions({
          UNSURE: "无法确认", OBSERVABLE: "可观察", PARTIAL: "部分可观察", UNOBSERVABLE: "不可观察",
        }, item.observability)}</select></label>
        <label>证据帧<input data-key="evidence_frames" value="${escapeHtml((item.evidence_frames || []).join(", "))}" placeholder="如 21, 41"></label>
        <label>备注<input data-key="notes" value="${escapeHtml(item.notes || "")}" placeholder="说明接受或拒绝依据"></label>
      </div>
    </article>`).join("");
  $$("#errorTruthMatrix input, #errorTruthMatrix select").forEach((input) => {
    input.addEventListener("input", markDirty);
    input.addEventListener("change", markDirty);
  });
}
function collectErrorTruthItems() {
  return $$(".error-truth-row", $("#errorTruthMatrix")).map((row) => {
    const value = {error_code: row.dataset.code};
    $$("[data-key]", row).forEach((input) => {
      value[input.dataset.key] = input.dataset.key === "evidence_frames"
        ? input.value.split(/[,，\s]+/).filter(Boolean).map(Number)
        : inputValue(input);
    });
    return value;
  });
}
function applyCheckpointBatch(button) {
  const rows = $$(".oni-checkpoint");
  const start = Math.max(1, Number($("#batchStartIndex").value) || 1);
  const end = Math.min(rows.length, Number($("#batchEndIndex").value) || rows.length);
  if (start > end) return toast("批量范围开始序号不能大于结束序号", true);
  rows.slice(start - 1, end).forEach((row) => {
    if (button.dataset.batchTarget) $('[data-key="target_status"]', row).value = button.dataset.batchTarget;
    if (button.dataset.batchBox) $('[data-key="bbox_status"]', row).value = button.dataset.batchBox;
  });
  state.review.batch_edits = [
    ...(state.review.batch_edits || []),
    {
      checkpoint_start_index: start,
      checkpoint_end_index: end,
      target_status: button.dataset.batchTarget || null,
      bbox_status: button.dataset.batchBox || null,
      edited_at: new Date().toISOString(),
    },
  ];
  markDirty();
  toast(`已批量修改检查点 ${start}–${end}，保存后写入审计`);
}
function renderEligibility() {
  const confirmed = state.review.overall_target_status === "correct" && state.review.same_subject_throughout === "yes";
  const viewReady = state.review.confirmed_view && state.review.confirmed_view !== "unsure"
    && ["usable", "partially_usable"].includes(state.review.action_usability)
    && (state.review.observability_items || []).every((item) => ["OBSERVABLE", "PARTIAL", "UNOBSERVABLE", "UNSURE"].includes(item.status));
  const eligibility = {
    "视角标定": state.review.status === "complete" && confirmed && viewReady,
    "RGB 规则标定": false,
    "模型训练": false,
    "发布": false,
  };
  $("#eligibilityGrid").innerHTML = Object.entries(eligibility)
    .map(([label, passed]) => `<div class="${passed ? "eligible" : "blocked"}"><strong>${label}</strong><span>${passed ? "具备资格" : "暂不具备"}</span></div>`).join("");
}
function collectOniReview() {
  const review = structuredClone(state.review);
  $$("[data-oni-path]", $("#oniForm")).forEach((input) => setPath(review, input.dataset.oniPath, inputValue(input)));
  review.identity_switch_intervals = collectIdentitySwitches();
  review.observability_items = collectObservabilityItems();
  review.error_truth_items = collectErrorTruthItems();
  review.checkpoints = $$(".oni-checkpoint").map((row) => {
    const value = {frame_index: Number(row.dataset.frame)};
    $$("[data-key]", row).forEach((input) => { value[input.dataset.key] = inputValue(input); });
    return value;
  });
  return review;
}
function validateOni(review) {
  // ONI 同样允许任意进度保存，包括尚未补齐或彼此矛盾的草稿字段。
  return [];
}
async function saveOni({next = false} = {}) {
  const review = collectOniReview();
  const errors = validateOni(review);
  if (errors.length) return toast(errors[0], true);
  setOniSaveIndicator("正在保存…", "dirty");
  try {
    const response = await api(`/api/review/oni/${encodeURIComponent(state.currentId)}/${state.currentModality}`, {
      method: "PUT",
      body: JSON.stringify({
        reviewer_id: QUICK_REVIEWER,
        base_revision: state.revision,
        review,
        change_reason: $("#oniChangeReason").value.trim() || `single human ONI ${review.review_mode} initial review`,
      }),
    });
    state.review = review;
    state.revision = response.revision;
    state.dirty = false;
    $("#oniChangeReason").value = "";
    setOniSaveIndicator(`已保存 · r${state.revision}`, "saved");
    toast(review.status === "complete" ? "ONI 模态主体复核已完成" : "草稿已保存");
    await refreshRecords();
    if (next) selectNextRecord();
  } catch (error) {
    setOniSaveIndicator("保存失败", "error");
    toast(error.message, true);
  }
}
async function refreshRecords() {
  const taskId = currentTaskId();
  state.bootstrap = await api("/api/review/bootstrap");
  state.records = taskRecords();
  state.meta = state.records.find((record) => (isOniTask() ? record.task_id : record.record_id) === taskId);
  renderRecordList();
  updateProgress();
  renderDashboard();
}
function selectNextRecord() {
  const id = currentTaskId();
  const currentIndex = state.records.findIndex((record) => (isOniTask() ? record.task_id : record.record_id) === id);
  const next = [...state.records.slice(currentIndex + 1), ...state.records.slice(0, currentIndex)].find((record) => !recordStatus(record).complete);
  if (!next) return toast("当前任务的全部记录都已完成");
  if (isOniTask()) selectOniRecord(next.record_id, next.modality);
  else selectRecord(next.record_id);
}

function currentFrame() {
  if (!state.record?.video) return 0;
  const max = Number(state.record.video.decoded_frame_count) - 1;
  return Math.max(0, Math.min(max, Math.round($("#reviewVideo").currentTime * Number(state.record.video.fps))));
}
function setFrame(frame) {
  if (!state.record?.video) return;
  const max = Number(state.record.video.decoded_frame_count) - 1;
  const value = Math.max(0, Math.min(max, Math.round(Number(frame) || 0)));
  $("#reviewVideo").currentTime = value / Number(state.record.video.fps);
  updateFrame();
}
function updateFrame() {
  const frame = currentFrame();
  $("#currentFrame").textContent = String(frame);
  $("#frameJump").value = String(frame);
  drawTrack(frame);
  const clip = currentDisagreementClip();
  const video = $("#reviewVideo");
  if (clip && !video.paused && frame >= Number(clip.end_frame)) {
    if (state.clipLoop) {
      video.currentTime = Number(clip.start_frame) / Number(state.record.video.fps);
    } else {
      video.pause();
    }
  }
}
async function loadTrack() {
  if (state.track.length) return;
  const payload = await api(state.record.track_url);
  state.track = payload.frames || [];
}
function drawTrack(frame) {
  const canvas = $("#trackCanvas");
  const video = $("#reviewVideo");
  const wrap = $(".video-wrap");
  canvas.width = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!$("#trackToggle").checked || !video.videoWidth || !state.track[frame]?.bbox_xyxy) return;
  const videoRect = video.getBoundingClientRect();
  const wrapRect = wrap.getBoundingClientRect();
  const [x1, y1, x2, y2] = state.track[frame].bbox_xyxy;
  const scaleX = videoRect.width / video.videoWidth;
  const scaleY = videoRect.height / video.videoHeight;
  context.strokeStyle = "#d8ff45";
  context.lineWidth = 3;
  context.strokeRect(videoRect.left - wrapRect.left + x1 * scaleX, videoRect.top - wrapRect.top + y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
}

function initializeEvents() {
  $$(".task-switcher button").forEach((button) => button.addEventListener("click", () => switchTask(button.dataset.task)));
  $("#recordSearch").addEventListener("input", renderRecordList);
  $("#actionFilter").addEventListener("change", renderRecordList);
  $$(".quick-filters button").forEach((button) => button.addEventListener("click", () => {
    $$(".quick-filters button").forEach((item) => item.classList.toggle("active", item === button));
    state.filter = button.dataset.filter;
    renderRecordList();
  }));
  $("#quickForm").addEventListener("input", markDirty);
  $("#quickForm").addEventListener("change", markDirty);
  $("#oniForm").addEventListener("input", markDirty);
  $("#oniForm").addEventListener("change", markDirty);
  $$("[data-add]").forEach((button) => button.addEventListener("click", () => addRow(button.dataset.add)));
  $("#actionSelect").addEventListener("change", () => {
    state.review = collectForm();
    setPath(state.review, "quick_review.action", $("#actionSelect").value);
    renderList("phaseIntervals", getPath(state.review, LISTS.phaseIntervals.path, []));
    renderList("events", getPath(state.review, LISTS.events.path, []));
    toast("动作已人工切换；请核对阶段和事件选项");
  });
  $("#saveButton").addEventListener("click", () => save());
  $("#saveNextButton").addEventListener("click", () => save({next: true}));
  $("#oniSaveButton").addEventListener("click", () => saveOni());
  $("#oniSaveNextButton").addEventListener("click", () => saveOni({next: true}));
  $$(".modality-switch button").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.modality !== state.currentModality) selectOniRecord(state.currentId, button.dataset.modality);
  }));
  $("#addIdentitySwitchButton").addEventListener("click", () => {
    state.review.identity_switch_intervals = collectIdentitySwitches();
    state.review.identity_switch_intervals.push({start_frame: null, end_frame: null, notes: ""});
    renderIdentitySwitches();
    markDirty();
  });
  $$("[data-batch-target]").forEach((button) => button.addEventListener("click", () => applyCheckpointBatch(button)));
  $("#useCurrentFrameButton").addEventListener("click", () => {
    const start = $('[data-path="quick_review.usable_start_frame"]');
    const end = $('[data-path="quick_review.usable_end_frame"]');
    (document.activeElement === start ? start : end).value = currentFrame();
    markDirty();
  });
  $("#reviewVideo").addEventListener("timeupdate", updateFrame);
  $("#reviewVideo").addEventListener("seeked", updateFrame);
  $$("[data-step]").forEach((button) => button.addEventListener("click", () => setFrame(currentFrame() + Number(button.dataset.step))));
  $("#jumpButton").addEventListener("click", () => setFrame($("#frameJump").value));
  $("#previousClipButton").addEventListener("click", () => selectDisagreementClip(-1));
  $("#nextClipButton").addEventListener("click", () => selectDisagreementClip(1));
  $("#playClipButton").addEventListener("click", playCurrentDisagreementClip);
  $("#addClipEventButton").addEventListener("click", addEventFromCurrentClip);
  ["clipReviewStatus", "clipEvidenceType", "clipObservability", "clipNotes"].forEach((id) => {
    $(`#${id}`).addEventListener(id === "clipNotes" ? "input" : "change", updateCurrentDisagreementClip);
  });
  $("#clipLoopToggle").addEventListener("change", () => {
    state.clipLoop = $("#clipLoopToggle").checked;
  });
  $("#importAllProposalButton").addEventListener("click", () => importProposal("all"));
  $("#importSegmentsButton").addEventListener("click", () => importProposal("segments"));
  $("#importRepsButton").addEventListener("click", () => importProposal("reps"));
  $("#importEventsButton").addEventListener("click", () => importProposal("events"));
  $("#trackToggle").addEventListener("change", async () => {
    if ($("#trackToggle").checked) {
      try { await loadTrack(); } catch (error) { toast(error.message, true); }
    }
    drawTrack(currentFrame());
  });
  $$(".reference-tabs button").forEach((button) => button.addEventListener("click", () => {
    $$(".reference-tabs button").forEach((item) => item.classList.toggle("active", item === button));
    $$(".reference-pane").forEach((pane) => pane.classList.toggle("active", pane.dataset.pane === button.dataset.tab));
  }));
  $("#exportButton").addEventListener("click", () => { window.location.href = "/api/review/export?scope=a"; });
  $("#exportDataButton").addEventListener("click", () => { $("#exportPopover").hidden = !$("#exportPopover").hidden; });
  $("#copyCodexButton").addEventListener("click", async () => {
    const text = "请读取 datasets/hyrox/reviews/human_v1/reviewer_a/ 下的单人人工复核结果。人工 4 使用 reps、phase_error_intervals、events 与 disagreement_clips；滑雪机和推雪橇采用两个临时 subject 分组，其余动作各一个分组。ONI 暂不属于本轮人工 4。";
    await navigator.clipboard.writeText(text);
    toast("结果说明已复制");
  });
  window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
}

initializeEvents();
initialize().catch((error) => toast(error.message, true));
