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
      action: record.action_candidate || "unknown_ood",
      target_status: "correct",
      video_usability: "usable",
      overall_result: "UNSURE",
      observability: "UNKNOWN",
      subject_id: "",
      authorization: "confirmed",
      usable_start_frame: 0,
      usable_end_frame: Number(record.video.decoded_frame_count) - 1,
      segments: [],
      reps: [],
      phase_error_intervals: [],
      events: [],
      proposal_decision: "unresolved",
      equipment_visibility: "unknown",
      notes: "",
    },
  };
}
function emptyOniReview(checkpoints) {
  return {
    status: "draft",
    overall_target_status: "unsure",
    same_subject_throughout: "unsure",
    observability: "UNKNOWN",
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
  switchTask("core");
}
function renderActionOptions() {
  $("#actionSelect").innerHTML = Object.entries(state.bootstrap.labels.actions)
    .map(([value, label]) => `<option value="${escapeHtml(value)}">${escapeHtml(label)}</option>`).join("");
}
function taskRecords() {
  if (state.task === "oni") return state.bootstrap.oni_records || [];
  return (state.bootstrap.records || []).filter((record) => record.core);
}
function recordStatus(record) {
  return state.task === "oni" ? record : (record.reviewer_a || {saved: false, complete: false});
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
  $("#queueTitle").textContent = task === "core" ? "15 段核心动作" : "64 个模态任务";
  $("#queueEyebrow").textContent = task === "core" ? "逐次/事件精标" : "Depth / IR 主体复核";
  $("#phoneReviewContent").hidden = true;
  $("#oniReviewContent").hidden = true;
  $("#emptyState").hidden = false;
  renderRecordList();
  updateProgress();
}
function renderRecordList() {
  const query = $("#recordSearch").value.trim().toLowerCase();
  const records = state.records.filter((record) => {
    const status = recordStatus(record);
    if (state.filter === "pending" && status.complete) return false;
    if (state.filter === "complete" && !status.complete) return false;
    return `${record.record_id} ${record.action_label} ${record.modality_label || ""}`.toLowerCase().includes(query);
  });
  $("#recordList").innerHTML = records.map((record, index) => {
    const status = recordStatus(record);
    const itemId = state.task === "oni" ? record.task_id : record.record_id;
    return `<button class="quick-record ${itemId === currentTaskId() ? "active" : ""}" type="button" data-id="${escapeHtml(record.record_id)}" data-modality="${escapeHtml(record.modality || "")}">
      <span class="order">${String(index + 1).padStart(2, "0")}</span>
      <span><strong>${escapeHtml(record.action_label)}</strong><small>${escapeHtml(record.record_id)}${record.modality_label ? ` · ${escapeHtml(record.modality_label)}` : ""}</small></span>
      <i class="record-dot ${status.complete ? "complete" : status.saved ? "saved" : ""}"></i>
    </button>`;
  }).join("");
  $$(".quick-record").forEach((button) => button.addEventListener("click", () => {
    if (state.task === "oni") selectOniRecord(button.dataset.id, button.dataset.modality);
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
  $("#topProgressText").textContent = `${state.task === "core" ? "核心动作精标" : "ONI 主体复核"} · ${completed}/${total} 已完成`;
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
  setSaveIndicator(state.revision ? `已保存 · r${state.revision}` : "未保存");
  fillForm();
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
  if (state.task === "oni") setOniSaveIndicator("有未保存修改", "dirty");
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
  const quick = review.quick_review;
  const max = Number(state.record.video.decoded_frame_count) - 1;
  const errors = [];
  const ranges = [
    ["可用范围", quick.usable_start_frame, quick.usable_end_frame],
    ...(quick.segments || []).map((item, index) => [`动作区间 ${index + 1}`, item.start_frame, item.end_frame]),
    ...(quick.reps || []).map((item, index) => [`第 ${index + 1} 次`, item.start_frame, item.end_frame]),
    ...(quick.phase_error_intervals || []).map((item, index) => [`阶段错误区间 ${index + 1}`, item.start_frame, item.end_frame]),
  ];
  ranges.forEach(([name, start, end]) => {
    if (start == null || end == null) return errors.push(`${name}必须填写开始帧和结束帧`);
    if (start < 0 || end > max) errors.push(`${name}超出 0–${max} 帧`);
    if (start > end) errors.push(`${name}开始帧不能大于结束帧`);
  });
  const repIds = new Set((quick.reps || []).map((item) => item.rep_id).filter(Boolean));
  (quick.phase_error_intervals || []).forEach((item, index) => {
    if (!item.phase || !item.error_code) errors.push(`阶段错误区间 ${index + 1} 必须选择阶段和错误`);
    if (item.rep_id && repIds.size && !repIds.has(item.rep_id)) errors.push(`阶段错误区间 ${index + 1} 的所属次数不存在`);
  });
  (quick.events || []).forEach((event, index) => {
    if (event.frame_index == null || event.frame_index < 0 || event.frame_index > max) errors.push(`关键事件 ${index + 1} 帧号无效`);
  });
  if (
    quick.status === "complete"
    && (!(quick.reps || []).length || !(quick.phase_error_intervals || []).length || !(quick.events || []).length)
  ) {
    errors.push("核心动作标记完成前必须填写逐次边界、阶段错误区间和关键事件");
  }
  if (quick.status === "complete" && !(quick.notes || "").trim()) errors.push("标记完成前，请填写复核结论");
  return errors;
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
        change_reason: "single human core fine annotation",
        evidence_frames: (review.quick_review.events || []).map((item) => item.frame_index).filter((value) => value != null),
        finish_review: review.quick_review.status === "complete",
      }),
    });
    state.review = review;
    state.revision = response.revision;
    state.dirty = false;
    setSaveIndicator(`已保存 · r${state.revision}`, "saved");
    toast(review.quick_review.status === "complete" ? "核心动作精标已完成" : "草稿已保存");
    await refreshRecords();
    if (next) selectNextRecord();
  } catch (error) {
    setSaveIndicator("保存失败", "error");
    toast(error.message, true);
  }
}

function proposalSegments() {
  return (state.proposal?.action_segments?.segments || []).map((item) => ({
    label: item.label || "target_action", start_frame: item.start_frame, end_frame: item.end_frame, notes: "AI 候选，已由人工载入",
  }));
}
function proposalReps() {
  return (state.proposal?.core_annotations?.reps || []).map((rep, index) => ({
    rep_id: rep.rep_id || `rep_${String(index + 1).padStart(3, "0")}`,
    start_frame: rep.start_frame, end_frame: rep.end_frame, validity: rep.validity || "UNSURE", notes: "AI 候选，待人工核对",
  }));
}
function proposalPhaseIntervals() {
  const output = [];
  (state.proposal?.core_annotations?.reps || []).forEach((rep) => {
    (rep.phases || []).forEach((phase) => {
      const matching = (rep.errors || []).find((error) => error.phase === phase.phase);
      output.push({
        rep_id: rep.rep_id, start_frame: phase.start_frame, end_frame: phase.end_frame, phase: phase.phase,
        error_code: matching?.error_code || "NO_ERROR", observability: "UNKNOWN", notes: "AI 候选，待人工核对",
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
  if (kind === "all" || kind === "segments") quick.segments = proposalSegments();
  if (kind === "all" || kind === "reps") {
    quick.reps = proposalReps();
    quick.phase_error_intervals = proposalPhaseIntervals();
  }
  if (kind === "all" || kind === "events") quick.events = proposalEvents();
  fillForm();
  markDirty();
  toast("AI 候选已载入草稿，请逐项人工核对");
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
  const payload = await api(`/api/review/oni/${encodeURIComponent(recordId)}/${encodeURIComponent(modality)}`);
  state.currentId = recordId;
  state.currentModality = modality;
  state.record = payload.record;
  state.meta = state.records.find((record) => record.task_id === `${recordId}__${modality}`);
  state.oniCheckpoints = payload.checkpoints;
  state.revision = payload.saved_review?.revision || 0;
  state.review = structuredClone(payload.saved_review?.review || emptyOniReview(payload.checkpoints));
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
  $("#oniRecordMeta").textContent = `${state.record.checkpoint_count} 个检查点 · ${state.record.camera_view || "未知视角"} · 单模态独立复核`;
  $("#oniPreviewImage").src = state.record.preview_url;
  setOniSaveIndicator(state.revision ? `已保存 · r${state.revision}` : "未保存");
  $$("[data-oni-path]", $("#oniForm")).forEach((input) => { input.value = getPath(state.review, input.dataset.oniPath, ""); });
  renderOniCheckpoints();
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
    return `<article class="oni-checkpoint" data-frame="${frame}">
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
}
function collectOniReview() {
  const review = structuredClone(state.review);
  $$("[data-oni-path]", $("#oniForm")).forEach((input) => setPath(review, input.dataset.oniPath, inputValue(input)));
  review.checkpoints = $$(".oni-checkpoint").map((row) => {
    const value = {frame_index: Number(row.dataset.frame)};
    $$("[data-key]", row).forEach((input) => { value[input.dataset.key] = inputValue(input); });
    return value;
  });
  return review;
}
function validateOni(review) {
  const errors = [];
  if (review.status === "complete") {
    review.checkpoints.forEach((row, index) => {
      if (row.target_status === "unreviewed" || row.bbox_status === "unreviewed") errors.push(`检查点 ${index + 1} 尚未复核`);
    });
    if (!(review.notes || "").trim()) errors.push("标记完成前，请填写一条总体备注");
  }
  return errors;
}
async function saveOni({next = false} = {}) {
  const review = collectOniReview();
  const errors = validateOni(review);
  if (errors.length) return toast(errors[0], true);
  setOniSaveIndicator("正在保存…", "dirty");
  try {
    const response = await api(`/api/review/oni/${encodeURIComponent(state.currentId)}/${state.currentModality}`, {
      method: "PUT",
      body: JSON.stringify({reviewer_id: QUICK_REVIEWER, base_revision: state.revision, review, change_reason: "single human ONI modality subject review"}),
    });
    state.review = review;
    state.revision = response.revision;
    state.dirty = false;
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
  state.meta = state.records.find((record) => (state.task === "oni" ? record.task_id : record.record_id) === taskId);
  renderRecordList();
  updateProgress();
}
function selectNextRecord() {
  const id = currentTaskId();
  const currentIndex = state.records.findIndex((record) => (state.task === "oni" ? record.task_id : record.record_id) === id);
  const next = [...state.records.slice(currentIndex + 1), ...state.records.slice(0, currentIndex)].find((record) => !recordStatus(record).complete);
  if (!next) return toast("当前任务的全部记录都已完成");
  if (state.task === "oni") selectOniRecord(next.record_id, next.modality);
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
  $("#copyCodexButton").addEventListener("click", async () => {
    const text = "请读取 datasets/hyrox/reviews/human_v1/reviewer_a/ 下的单人人工复核结果。核心动作使用 reps、phase_error_intervals 和 events；ONI Depth/IR 结果位于 oni_records/，两个模态必须独立处理。自动动作识别暂缓，正式动作类型只取人工选择。";
    await navigator.clipboard.writeText(text);
    toast("结果说明已复制");
  });
  window.addEventListener("beforeunload", (event) => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
}

initializeEvents();
initialize().catch((error) => toast(error.message, true));
