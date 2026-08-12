# Changelog

## Unreleased

- 完成实时姿态优化第 1–12 轮第一版：统一性能 CSV 与播放速度比、摄像头/有窗口视频
  Latest-Frame、MediaPipe LIVE_STREAM 单槽调度、显示/分析双 One Euro、纯推理自适应
  分辨率、统一 `JointMetric`、3D 可靠性、人体 canonical 坐标、时序地板估计、姿态
  帧号/源时间戳门控，以及按 pose FPS→推理分辨率→可选分析频率顺序降载的
  `RealtimeBudgetController`。视频和渲染时钟不随姿态推理降速。
- 扩展现有人工角度工具为六通道及显式旧新版比较：raw/filtered 2D、raw/filtered
  3D、canonical 3D、selected-rule angle、MAE/Median/P90/P95、曲线延迟、最低点/
  完全伸展事件误差和覆盖审计。当前 150 条人工投影 2D 标注覆盖四个必选动作，缺
  标定 30°/45°视角和成对程序事件帧；代理比较的 MAE/P95 分别退化 0.136°/1.5032°，
  因此不修改正式 2D 阈值、不提升 3D 为正式裁决。
- 同步更新 README、桌面/网页版使用说明、算法与数据现状、上传视频实施方案、产品契约
  说明和实时优化任务清单，明确桌面、浏览器与无窗口批处理的不同运行边界。
- 重构 README，将阶段性数据进步、模型优化过程、消融结果、数据接入轮次和测试基线移出
  项目介绍；README 现在只保留稳定能力、安装、使用、输出语义和限制，迭代历史统一由
  本变更日志及对应研究报告承载。
- 新增轻量时序证据逐视频留一 v2：Ridge 阶段发射加因果 HMM 提高阶段边界精度，
  人工阶段边界校准最多两个缺失阶段及候选合并/结算；候选 recall 提至 31.31%、
  precision 为 94.37%，但明确只作为现有候选链侧车。站立基线增加角度合理性、
  稳定性和双侧一致性质量门，8 段中 6 段安全回退固定阈值。接触影子融合保留二维
  状态并只用分割/3D 修正事件锚点，零新增 FP 下把事件 MAE 从 14.43 降到 12.10 帧。
  置信度校准增加 5 个正例、3 段视频、训练零误报和汇总误报硬门；本轮硬门未通过，
  自动回退基线，不用额外误报换取较低 `UNSURE`。
- 新增统一 RGB 数据角色、泄漏检查和可回溯错误库：30 条记录分为 27 条 development、
  3 条 validation（27 条可训练、2 条可评估、1 条不可用），训练/评估记录及临时 subject
  无角色重叠；核心三动作因缺 validation/test 明确标记为非独立回归。优化报告新增
  dataset role、subject、action × view × subject 和终端事件误差指标；错误库导出
  75 个最长 1 秒的 TP/FP/FN/`UNSURE`/状态不匹配 MP4，并冻结数据、代码与配置哈希。
- 接入 `hyrox_human_review_a_20260728_170359` 的其余手机 RGB 复核：30/30 条形成终态
  人工结论，29 条形成可用精标，`phone_sled_push_005` 按人工“视频不可用”结论排除；
  `phone_skierg_002` 的 73 个候选分析周期按用户确认记为有效。非核心动作缺失事件从
  已人工核对的阶段边界确定性派生，原始导出包和来源哈希保持可审计。高分歧短片段改为
  可延期的主动学习任务，不作为手机 RGB 精标导入门。
- 人工复核站新增独立的 ONI 主体、视角先验和错误真值任务；Depth/IR 支持并排同步定位但
  分别保存，新增动作可观察性矩阵、批量检查点、资格门禁、动作/视角/模态仪表盘、筛选、
  连续审计和三类独立导出。
- 接入 `hyrox_human_review_a_20260727_150326` 的 15 条核心手机 RGB 精标（70 次、
  700 个阶段/错误或事件可观察性字段），并按用户确认将手机标注统一记录为
  `OBSERVABLE`；用户进一步确认 15 条结果和保留的 AI proposal 均已完成人工复审，
  770 条派生备注更新为“已人工核对”，15 条可用于内部规则校准和监督实验；正式 RGB
  优化仍与 ONI 复核保持独立。
- 按用户复核把 `phone_wall_ball_002_rep_003` 从 `NO_REP` 修正为 `VALID`，写入
  revision 3 审计记录，并将精标一致性警告从 1 条降为 0 条。
- 新增隔离的未审核 ONI 辅助实验：读取 32 条记录、64 路 Depth/IR 和 1,536 个采样
  检查点，在不改变 RGB 运行指标和默认配置的前提下，将研究动作覆盖从 3 类扩展到
  8 类、错误码场景从 9 类扩展到 12 类；全部 ONI 结果保持不可训练、不可发布。
- 新增可复现的手机 RGB 基线/优化回放对比和按人工事件帧的单调候选对齐：动作候选
  从 26/70 提升到 57/70，candidate recall 从 35.71% 提升到 74.29%，count MAE
  从 2.93 降到 0.87，精确计数视频从 1/15 提升到 4/15。Lunge 使用经精标验证的
  髋高回位代理；Burpee 提高短阶段响应并在有限视频结束时结算最后一次；Wall Ball
  可用最低点、腕部投掷和下肢伸展证据在下一次下蹲或流结束时安全结算短投掷端点。
- 修复三项人体动作规则链：Burpee 的地板增强结果现会在同帧贯通脚部事件、手部位置和
  胸部触地证据；Wall Ball 的阶段端点与最终规则共享固定的视角侧选择；Lunge 只在
  完全伸展、下一次下降或视频结束边界结算。可观测性门槛改为动作 × 视角 × 规则覆盖，
  保留全局值作为未配置组合的回退。
- 人工精标导入新增闭区间、rep、连续阶段、事件归属和审计链校验；保留
  `phone_lunge_004` 缺少审计 revision 2 的来源审计缺口，不静默补写来源历史。
- 接入 30 条 reviewer A 快速人工复核作为独立覆盖层，确认手机 30 条与 ONI 32 条的全部
  8 类授权用途，并保留第二复核者、逐次精标与 ONI Depth/IR 人工主体门。
- 修复 Rowing/SkiErg 在后方和斜后方视角下因远侧肢体遮挡导致的漏计与 `UNSURE`：按可靠
  侧选择关键关节、使用动作特定可见度和端点迟滞，并增加复核视频因果缓存回放报告。
- 第 10、11 轮报告新增单人复核进度、内部 subject-ID 临时豁免边界和授权确认计数。

All notable changes are recorded here. Versions follow Semantic Versioning;
development builds use the `X.Y.Z.devN` form.

## [Unreleased]

### Added

- Added `RealtimeBudgetController` with rolling inference/pose-age P95 and
  queue saturation. It changes pose admission from 20 to 15 to 12 FPS before
  requesting inference-only resolution steps and optional-analysis throttling;
  capture, video and render clocks remain independent.
- Added six-channel manual angle validation, canonical-3D trace export,
  action/view coverage audits and explicit baseline-report non-regression
  comparisons. The current report truthfully preserves missing calibrated
  30/45-degree views and old/new event-frame evidence.

- Added an internal leave-one-video-out temporal-evidence v2 runner with Ridge
  phase emissions, a causal HMM, manually calibrated phase skipping and
  candidate settlement, quality-gated per-video standing baselines, a
  2D-authoritative segmentation/3D contact-event shadow, and a hard
  false-NO_REP rollback for confidence calibration. None of these experiments
  changes the product default.
- Added an internal 2D + body-relative 3D shadow-evidence experiment covering
  hip-compensated knee/foot motion, leg depth order, 3D knee/hip angles, foot
  speed/dwell/timing and torso shoulder-hip geometry. The leave-one-video-out
  runner keeps the 2D floor/contact chain authoritative, blocks 3D candidate
  creation and VALID promotion, emits leakage/safety audits, and reports the
  required recall, status, UNSURE, event-frame and per-error FP/FN comparison.
  Its second revision treats low-quality angle disagreement as unavailable,
  requires temporal conflict consensus, evaluates angle/body/combined
  ablations per fold, and exactly falls back to 2D when no 3D candidate wins
  on the other videos; all 15 current folds choose that safe fallback.
- Added `pose-oni-research-round11` and a versioned offline-ONI safety
  contract. All 32 Depth and IR recordings now have independent 24-checkpoint
  subject-review proposals, separate JSONL tracks and 64 modality-specific
  contact sheets. The run produced 380/768 Depth and 414/768 IR automatic
  candidates while preserving every miss/low-confidence checkpoint and
  keeping human-confirmed identity, verified errors and training eligibility
  at zero.
- Added scoped ONI evidence, phone-recapture and future synchronized-RGB-D
  value reports. Metric Depth is limited to line-of-sight sensor-surface
  distance; uncalibrated ground, contact, body-part and action-error claims
  remain unobservable or research-only. The contract rejects RGB-Depth
  registration, phone-ONI pairing, phone frame labels, IR-as-RGB, unpaired
  distillation and silhouette-derived contact truth for the current data.
- Added the Round 10 versioned action-gating, scoring/correction, coordinate-
  space and realtime-latency contracts. Desktop and web results now expose
  additive contract versions, manual/automatic provenance, traceable Evidence,
  uncalibrated-score suppression, explicit 2D/relative-3D/camera-ray/metric-
  depth semantics and stale pose/action/correction suppression.
- Added a dependency-light multinomial Logistic Regression action-gating
  baseline with causal body-canonical windows, normalization, balanced
  training, temperature calibration, artifact hashes, group-exclusive cross
  validation, idle/transition/unknown classes, entry/exit hysteresis, minimum
  duration and cooldown. The replay sidecar is explicit and default-off, and
  never switches the manually selected analyzer.
- Added `pose-shadow-round10` and truthful A-F/readiness/failure-pool reports.
  The current 30 records complete the engineering contract loop, but all 30
  remain blocked by authorization, subject identity and independent human
  truth; no model or performance claim is produced while those gates fail.
- Added `pose-cache-round8` and a complete 30-record, 15,515-frame round-eight
  offline workflow. MediaPipe Lite and Full now retain separate target-bound
  raw image/world landmarks, bbox and RLE-mask provenance, model/software
  metadata, inference clocks, native/unified 33-point schemas and mapping loss.
  The completed cache has 15,071 Lite and 14,766 Full detections; 2,439
  high-disagreement frames are queued as human-review priorities, while teacher
  proposals are explicitly prohibited from becoming or silently averaging into
  ground truth.
- Added explicit image-normalized, image-pixel, estimated camera-ray,
  MediaPipe body-relative world and reversible body-canonical coordinate
  layers, including bone/z/left-right/orientation/2D-3D quality audits. The
  current phone footage remains `estimated_intrinsics`; without calibration,
  synchronized depth or 3D truth, no absolute monocular 3D accuracy is claimed
  and no phone-to-ONI frame/pixel pairing is created.
- Added three non-overwriting temporal artifacts for every source frame:
  strictly causal analysis, constrained display-only prediction and centered
  five-frame offline annotation assistance. The selected
  `joint_adaptive_round8_v2` improves the current responsive lag-jitter score
  by about 5.63% without increasing missing rate. A 15 ms display horizon
  reduces the measured offline lag while passing reversal overshoot,
  support-foot drift and bone-length gates, and remains forbidden from driving
  rules or reports.
- Added per-joint lag, endpoint, jitter, missingness, left/right, bone-length
  and prediction-stability audits; provisional event-anchor review sheets; a
  Lite/Full × full-size/640/ROI CPU benchmark matrix; and an artifact-integrity
  validator. GPU is explicitly untested in the current environment, physical
  sensor-to-photon remains `not_measured`, ROI remains disabled, and final
  event timing still requires round-nine independent double review. Integrity
  validation reports 15,515 rows for each expected artifact class and zero
  violations.
- Added `pose-tracking-round7` for the 30-record target-lock workflow:
  YOLO Pose candidate caching, joint IoU/motion/appearance/skeleton/action
  association, explicit visual-review approval, record-local canonical target
  IDs, split-source-track reinitialization, crossing/switch events, and
  other-person ignore masks. The completed run binds 15,166/15,515 frames,
  excludes 349 ambiguous/missing/stale frames, writes 5,640 ignore masks, and
  records the `phone_sled_push_005` source-track transition at frame 550
  without treating the reacquired candidate as a second athlete.
- Added candidate-only tracks for all ten required equipment/scene classes,
  including object IDs, bounded regions/masks, occlusion/out-of-frame fields,
  target association and versioned scene-calibration proposals. The run writes
  43,532 search-region proposals while keeping confirmed visibility at zero
  and actual load, distance, target hit and other non-observable rule fields
  explicitly `UNOBSERVABLE`.
- Added a full-frame versus target-ROI ablation over all 15,515 phone frames,
  including reversible per-frame affine transforms, detection/endpoint/joint
  accuracy, identity IoU and P95 latency. Detection interval 10 and padding
  1.6 produced 97.267% ROI versus 96.758% full-frame detection and negligible
  affine round-trip error, but failed both gates (9.819% normalized joint-error
  P95; 18.761 ms ROI-pipeline versus 20.433 ms full-frame P95), so ROI and all
  default product behavior remain disabled/unchanged.
- Added `pose-phone-rgb-round6` for the independent phone-RGB round-six
  workflow. It excludes 30 AppleDouble metadata files before record-ID
  assignment, preserves the 30 original Chinese filenames, creates read-only
  hash-verified backups, and decodes all 15,515/15,515 declared frames with
  independent timestamps and container/video metadata. New records use only
  `phone_rgb`; `phone_rgb_future` remains a read compatibility alias.
- Added versioned phone data-role, coverage-gap and observability reports plus
  independent phone-data, coordinate-quality and realtime-latency baselines.
  The full-frame run detected a raw pose on 15,012/15,515 frames without
  treating filenames as truth or reporting accuracy. All training/golden roles
  remain disabled, the eight web examples remain unverified candidates, and
  ONI-phone pairing remains zero.
- Added the isolated `oni-export.exe`, `pose-oni-export`, and `pose-oni-sync`
  round-four/five workflows. All 32 ONIs now have lossless uint16 Depth/IR
  frame exports, source frame/timestamp indices, content fingerprints,
  derived depth previews, and per-record metadata. The run exported 18,709
  Depth and 18,713 IR frames (24,228,045,828 payload bytes) with zero audit
  consistency errors; a real repeat export matched frame counts, index hashes,
  and aggregate frame-content hashes.
- Added ONI-internal Color/Depth frame-index and nearest-timestamp pairing,
  error statistics, quality grades, fine-event exclusions, and an independent
  independent-phone timeline schema. The current files have no Color, so all 32
  reports correctly contain zero pairs, are `video_level_only`, and are
  excluded from fine RGB-D event training. IR and independent phone data are never
  substituted for Color.
- Added the isolated C++/OpenNI2 `oni-inspect.exe` and `pose-oni-audit`
  round-three workflow for recorded files only. It audits full playback,
  Color/Depth/IR modes, frame counts, timestamps, indices, interval P50/P95,
  decode/timeline anomalies, estimated drops, and whole/center depth quality
  without adding OpenNI to the product runtime. The 32-record scan completed
  with no decode or timeline errors; every ONI contains Depth + IR but no
  Color, so all 32 are class B and are separately listed as not RGB-D
  qualified. Phone data is not consulted, and target-athlete identity remains
  pending.
- Added `pose-dataset-manifest` for the ONI-only round-two workflow. It parses
  the existing Chinese filenames without renaming them, preserves stable
  record IDs, creates and independently hashes a full read-only backup,
  validates the manifest and no-pairing contract, and writes an explicitly
  independent phone interface. The current run records 32/32 verified ONI
  backups; subject identity, usage authorization, and target-athlete selection
  remain explicitly pending instead of being inferred.
- Added `pose-baseline` and `scripts/run_baseline_regression.ps1` to freeze the
  phase-zero RGB product baseline in one command: dependency/model/config
  hashes, copied configs, action-output schema, eight-video candidate/rule/3D
  evidence, deterministic MediaPipe P50/P95 latency, optional physical-camera
  FPS, full Python/Node/smoke logs, and an annotated Git-tag suggestion. The
  baseline path does not read ONI, require OpenNI, or enable neural inference.
- Added browser camera track/settings diagnostics for actual presented FPS,
  frame-interval P50/P95 and instability, low light, and duplicate frames.
- Added `pose-camera-benchmark` for explicit on-device default/DSHOW/MSMF,
  MJPG/YUY2 benchmarking and an exact configuration backend cache. Physical
  sensor-to-photon values remain null unless externally measured.
- Added an isolated browser `DisplayPosePredictor` for 0–45 ms
  expected-display-time compensation with smoothed velocity, confidence and
  stale-gap gates, body-scale displacement limits, reversal damping, and
  support-foot constraints. Predicted landmarks remain display-only.
- Added fingerprint-validated pose and hand-keypoint caches for all eight fixed
  web samples. Demo playback now performs zero model inference while offline
  golden validation continues to execute the real backends.

- Added MediaPipe world-landmark 3D shadow kinematics, stateful reliability
  gates, per-angle 2D/3D comparison, and grouped availability/failure reports
  without changing HYROX decision inputs.
- Added confidence-only 3D Assist Mode for mapped knee, hip, elbow, and shoulder
  rules. Reliable agreement can raise rule confidence, severe 2D/3D conflict
  downgrades the candidate to `UNSURE`, and unavailable 3D falls back to the
  unchanged 2D decision path.
- Added a strict MediaPipe-only product backend configuration, shared backend
  support tiers, p50/p95 latency metrics, a synchronous baseline tool, and a
  round-one architecture audit report.
- Added fixed-interval golden regression for all eight bundled HYROX videos and
  versioned reports through the `pose-golden` CLI.
- Added short-smoke and formal 30/60 minute endurance validation with FPS, P95
  latency, process-memory growth, read-failure, and output-integrity metrics
  through the `pose-endurance` CLI.

### Changed

- Desktop MediaPipe camera and visible video playback now share a one-slot
  latest-frame LIVE_STREAM path with strict frame/timestamp identity, stale
  pose suppression, independent display/analysis filters, and inference-only
  adaptive resolution. Headless offline video remains synchronous by design.

- Closed the local-first architecture boundary: browser workers send only raw
  landmarks to Python HYROX analysis, constant-velocity predictions are Canvas
  only, server fallback is configuration-controlled, and report protocol
  whitelists reject prediction fields. Neural/temporal models and training
  flows remain unimplemented.
- Desktop camera startup now uses the device cache in `auto` mode and otherwise
  tries the OpenCV default backend before safe platform fallbacks; DSHOW is no
  longer permanently hard-coded.
- Web angle overlays now show quality-gated MediaPipe world-landmark 3D joint
  angles and label them `3D`; unavailable or unreliable 3D measurements are
  omitted instead of being silently replaced by 2D display values. Bundled
  sample caches were upgraded to v2 to include world landmarks.
- File-backed web playback is now paced at the video's encoded frame rate.
  Cached samples still analyze every frame, but no longer play faster than the
  source video when cached pose lookup finishes ahead of realtime.
- Replaced WebSocket result polling with an event-driven sender thread and made
  optional finger tracking off by default to reduce realtime camera latency.

- Promoted the formal product configuration from 3D shadow collection to
  `assist`; selected angles, phase thresholds, contact, floor, takeoff,
  landing, wrist-timing, step, and distance rules remain 2D.
- Reworked realtime One Euro smoothing to use observation capture timestamps,
  selectable stable/balanced/responsive profiles, body-region response scales,
  independent image/world landmark state, and a configurable 250 ms gap reset.
- Product `auto` now resolves only to MediaPipe; the web UI exposes only
  MediaPipe Pose, while YOLO/RTMW remain available solely through explicit
  experimental and offline-comparison paths.
- Split the desktop entry point into runtime, CLI, capture, backend, display,
  recording, session, and HYROX-analysis components. `main.py` is now a thin
  stable launcher.
- Retired the independent `src.realtime_pose` execution loop; its compatibility
  facade now translates legacy arguments and forwards to the maintained desktop
  runtime.

### Performance

- Reduced browser main-thread rendering work with fixed pose-coordinate
  buffers, cached connections/fonts/video transforms, 12 FPS angle labels,
  5 FPS metrics, 3 FPS statistics, and content-sensitive feedback DOM
  updates. Latency audits now include render-loop, Canvas, and DOM P95 plus
  Long Task phase attribution.
- Realtime web results are now pushed as soon as inference completes. On the
  current 30-frame local protocol probe at a 640 px long edge and JPEG quality
  0.65, round-trip latency improved from about 58 ms to 18.7 ms p50 and 35.5 ms
  p95; server processing was 17.2 ms p50, pose inference 14.4 ms p50, and pose
  age 16.0 ms p50. These are machine-specific baselines, not hardware-neutral
  guarantees.
- Without presentation pacing, cached processing of the 133-frame lunge sample
  required about 1.1 seconds with 0.0 ms runtime model inference, compared with
  about 15.7 seconds on the former inference path. Web presentation is now
  intentionally capped at the source frame rate, so this 30 FPS sample plays
  for its original duration of about 4.43 seconds.

### Validation

- The current full suite passes 752 Python tests and 17 Node tests. Full-model
  golden replay passes all 8/8 HYROX videos; Doctor, no-camera smoke,
  compileall, text-format, diff, and package-build checks also pass. Real
  camera backend and physical sensor-to-photon results remain device-site
  acceptance work and are not synthesized by automated tests.

## [0.1.0.dev0]

### Added

- Added strict schema validation for HYROX action, shared contact/foot-event,
  observability, and personal-reference YAML; `pose-doctor` now validates all
  configuration groups and output writability.
- Added rolling logs, classified `CFG/SRC/BCK/OUT/RUN/REC` errors, stable exit
  codes, optional debug tracebacks, safe resource cleanup, and partial-session
  recovery metadata.
- Added versioned JSON/CSV artifacts, legacy-schema compatibility, future
  schema rejection, configurable web output roots, and preview-first
  retention/quota cleanup through `pose-clean`.
- Added reproducible core/YOLO/RTMW/development dependency groups, ten
  installable CLI entry points, wheel/sdist packaging, release guidance,
  no-camera smoke tests, and Windows/Linux CI on Python 3.10 and 3.12.

### Changed

- Split the web model choices into explicit `纯 MediaPipe` and
  `YOLO + MediaPipe` pipelines. Explicit MediaPipe never loads YOLO; explicit
  YOLO + MediaPipe uses identity-checked dual-model fusion for every action.
- Kept automatic Lunge analysis on the YOLO + MediaPipe identity-lock path
  while retaining pure YOLO Pose for internal automatic/desktop compatibility.
- Changed Sled Pull analysis-cycle completion from
  `reach → pull → recover` to `reach → pull → recover → reach`, so the
  forward return belongs to the same cycle.
- Versioned session, web, replay, metrics, doctor, multicamera, reference, and
  comparison outputs with common program and artifact identity fields.
- Updated README, desktop/web/model/configuration guides, maturity status, and
  release documentation to match the current behavior and 400-test baseline.

### Included

- Eight HYROX analyzers, web realtime analysis, MediaPipe, optional YOLO/RTMW
  backends, personal-reference DTW comparison, local/public web launchers,
  session reports, and multi-camera synchronization checks.
