import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import {
  compareUploadStabilizationStrategies,
} from "../webui/static/workers/upload_pose_stabilizer.mjs";

function argument(name, fallback = "") {
  const index = process.argv.indexOf(name);
  return index >= 0 && index + 1 < process.argv.length ? process.argv[index + 1] : fallback;
}

const inputDirectory = argument("--input-dir");
const inputPath = argument("--input");
if (!inputPath && !inputDirectory) {
  throw new Error("usage: node tools/run_display_stability_ablation.mjs (--input timeline.json | --input-dir timelines) [--output report.json]");
}
const outputPath = argument(
  "--output",
  inputDirectory
    ? path.join(path.dirname(inputDirectory), "display_stability_ablation.json")
    : path.join(path.dirname(inputPath), `${path.basename(inputPath, path.extname(inputPath))}_display_ablation.json`),
);
const sources = inputDirectory
  ? fs.readdirSync(inputDirectory)
    .filter(name => name.endsWith(".json"))
    .sort()
    .map(name => path.join(inputDirectory, name))
  : [inputPath];
const cases = sources.map(source => {
  const payload = JSON.parse(fs.readFileSync(source, "utf8"));
  const frames = Array.isArray(payload) ? payload : payload.frames;
  if (!Array.isArray(frames)) throw new Error(`${source}: input must be a frame array or an object with frames`);
  return {
    case_id: payload.case?.case_id || path.basename(source, path.extname(source)),
    source,
    ...compareUploadStabilizationStrategies(frames),
  };
});
const syntheticFrames = Array.from({ length: 90 }, (_, index) => {
  const timestampMs = index * 33;
  const normalX = 0.20 + index * 0.001 + (index % 2 ? 0.0015 : -0.0015);
  const injectedHighConfidenceJump = index === 30 || index === 31;
  const injectedOcclusion = index >= 60 && index <= 66;
  return {
    timestamp_ms: timestampMs,
    phase: index < 45 ? "drive" : "recover",
    reps: index >= 45 ? 1 : 0,
    candidate_count: index >= 45 ? 1 : 0,
    keypoints: [{
      name: "left_wrist",
      x: injectedHighConfidenceJump ? 0.92 : normalX,
      y: 0.40,
      z: 0,
      visibility: injectedOcclusion ? 0.10 : 0.99,
      presence: injectedOcclusion ? 0.10 : 0.99,
    }],
  };
});
const syntheticControl = {
  description: "deterministic smooth trajectory with a two-frame high-confidence jump and a seven-frame low-confidence occlusion",
  ...compareUploadStabilizationStrategies(syntheticFrames),
};
const report = cases.length === 1
  ? { ...cases[0], synthetic_occlusion_control: syntheticControl }
  : {
      schema_version: 1,
      artifact_type: "display_stability_ablation_suite",
      formal_fields_unchanged: cases.every(item => item.formal_fields_unchanged),
      case_count: cases.length,
      cases,
      synthetic_occlusion_control: syntheticControl,
    };
report.generated_at = new Date().toISOString();
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
const markdownPath = outputPath.replace(/\.json$/i, ".md");
const reportCases = report.cases || [report];
const tableRows = reportCases.map(item => {
  const legacy = item.strategies.find(strategy => strategy.strategy === "legacy_bidirectional");
  const gated = item.strategies.find(strategy => strategy.strategy === "bidirectional");
  const reduction = legacy.metrics.normalized_acceleration_p95 > 0
    ? (1 - gated.metrics.normalized_acceleration_p95 / legacy.metrics.normalized_acceleration_p95) * 100
    : 0;
  return `| \`${item.case_id}\` | ${legacy.metrics.normalized_acceleration_p95.toFixed(3)} | ${gated.metrics.normalized_acceleration_p95.toFixed(3)} | ${reduction.toFixed(1)}% | ${legacy.metrics.hidden_point_count} | ${gated.metrics.hidden_point_count} | ${legacy.metrics.reacquisition_or_flicker_count} | ${gated.metrics.reacquisition_or_flicker_count} |`;
});
const syntheticLegacy = syntheticControl.strategies.find(item => item.strategy === "legacy_bidirectional");
const syntheticGated = syntheticControl.strategies.find(item => item.strategy === "bidirectional");
const syntheticReduction = (1 - syntheticGated.metrics.normalized_acceleration_p95
  / syntheticLegacy.metrics.normalized_acceleration_p95) * 100;
const markdown = [
  "# 显示稳定策略消融报告",
  "",
  `- 正式阶段/计数字段保持不变：\`${report.formal_fields_unchanged}\``,
  "- 双向策略的前视窗口：约 `66.7 ms`（2 帧，30 FPS）；仅用于上传回放。",
  "- 动态视频加速度是轨迹稳定性代理，不等同于静止抖动真值；黄金视频尚无逐帧遮挡等级。",
  "",
  "## 现有黄金视频",
  "",
  "| 样例 | 旧版加速度 P95 | 门控+双向 P95 | 变化 | 旧版隐藏点 | 新版隐藏点 | 旧版闪烁/重捕获 | 新版闪烁/重捕获 |",
  "|---|---:|---:|---:|---:|---:|---:|---:|",
  ...tableRows,
  "",
  "## 确定性遮挡控制序列",
  "",
  syntheticControl.description,
  "",
  `- 高置信度跳点与短遮挡下，加速度 P95 从 \`${syntheticLegacy.metrics.normalized_acceleration_p95.toFixed(3)}\` 降至 \`${syntheticGated.metrics.normalized_acceleration_p95.toFixed(3)}\`，下降 \`${syntheticReduction.toFixed(1)}%\`。`,
  `- 正式字段隔离：\`${syntheticControl.formal_fields_unchanged}\`。`,
  "",
  "结论：规则型异常门控能显著抑制高置信度错误跳点；双向平滑对部分真实动态视频继续降低加速度，但不能把未标注的动作加速度直接解释成静止抖动改善率。",
  "",
].join("\n");
fs.writeFileSync(markdownPath, markdown, "utf8");
const baselinePath = path.join(path.dirname(outputPath), "baseline.json");
if (fs.existsSync(baselinePath)) {
  const baseline = JSON.parse(fs.readFileSync(baselinePath, "utf8"));
  baseline.stage_b_display_ablation = {
    formal_fields_unchanged: Boolean(report.formal_fields_unchanged),
    synthetic_acceleration_p95_reduction_percent: Number(syntheticReduction.toFixed(1)),
    upload_bidirectional_lookahead_ms: syntheticGated.metrics.algorithmic_lookahead_ms,
    real_video_label_caveat: "golden videos do not have frame-level occlusion severity labels",
  };
  baseline.files = {
    ...(baseline.files || {}),
    display_ablation_json: path.basename(outputPath),
    display_ablation_markdown: path.basename(markdownPath),
  };
  fs.writeFileSync(baselinePath, `${JSON.stringify(baseline, null, 2)}\n`, "utf8");
}
process.stdout.write(`${outputPath}\n`);
