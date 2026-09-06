# HYROX 姿态分析

这是一个面向 HYROX 训练场景的人体姿态分析项目。它通过人体关键点和可复现的时序规则，为 8 项 HYROX 动作提供实时骨架、动作阶段、技术反馈和计数结果。

项目提供两种使用方式：

- **网页版**：在电脑或手机浏览器中使用本机摄像头，支持实时分析、上传视频、逐次动作语音提示，以及文字、JSON、CSV 报告下载。
- **桌面版**：通过 OpenCV 窗口使用摄像头或视频文件，适合本机训练、调试、录像和批量分析。

正式网页、桌面摄像头和默认视频分析统一使用 MediaPipe Pose。YOLO11n Pose、YOLO + MediaPipe 和 YOLO + RTMW 仅用于显式实验或离线比较，不会被产品 `auto` 自动加载。实时关键点使用 One Euro 平滑，个人参考动作比较支持 DTW 对齐。

> 本项目输出的是基于人体关键点的视觉运动学分析，不是医疗诊断，也不等同于 HYROX 正式比赛裁判结论。

## 主要功能

- 8 项 HYROX 动作的姿态与阶段分析；
- 实时骨架、关节角度和动作问题提示；
- 对可验证动作输出 `VALID`、`NO_REP` 和 `UNSURE`；
- 网页版逐次动作语音改进建议；
- 视频上传、桌面视频回放和无窗口批量处理；
- 最新帧低延迟播放、过期姿态抑制和自适应推理负载；
- 统一 raw/filtered 2D/3D、canonical 3D、地板与接触辅助证据；
- 匿名网页会话及文字、JSON、CSV 报告；
- 个人参考动作保存和 DTW 对齐比较；
- 配置校验、运行日志、输出清理和回归验证工具。

## 支持的动作

| 参数 | 动作 | 分析方式 |
|---|---|---|
| `lunge` | 负重箭步蹲 | 人体规则验证、有效/未完成/不确定计数与技术提示 |
| `wall_ball` | Wall Ball | 人体规则验证、有效/未完成/不确定计数与技术提示 |
| `rowing` | Rowing | 划船分析周期与训练区间技术提示 |
| `skierg` | SkiErg | 拉动分析周期与髋铰链提示 |
| `burpee_broad_jump` | Burpee Broad Jump | 人体规则验证、延迟结算计数与落地提示 |
| `sled_push` | Sled Push | 推行状态、步态分析周期与技术提示 |
| `sled_pull` | Sled Pull | 拉动分析周期与跪姿/坐姿违规代理 |
| `farmers_carry` | Farmers Carry | 连续搬运监控、稳定性与手臂位置提示 |

Lunge、Wall Ball 和 Burpee Broad Jump 会形成需要人体规则验证的动作候选；Rowing、SkiErg、Sled Push 和 Sled Pull 记录的是分析周期；Farmers Carry 采用连续监控，不按次数拆分。

## 工作原理

程序不会让姿态模型直接生成“膝盖内扣”或“伸展不足”等结论，也不会自动猜测用户正在进行的动作。用户先选择 HYROX 动作，系统再通过对应的状态机和规则分析姿态关键点。

```text
摄像头/视频最新帧
  → MediaPipe image + world landmarks
  → 显示/分析两套 One Euro
  → 统一关节角、3D 可靠性、地板和脚部证据
  → ReliableSideSelector / 双侧必需规则
  → 动作专属状态机
  → 当前技术反馈与完整周期规则验证
  → 可观测性检查
  → 骨架、文字/语音反馈和计数结果
```

实时显示和正式分析使用相互独立的平滑流。MediaPipe world landmarks 可辅助显示和置信度判断，但动作阶段、接触、计数和规则结论仍以二维证据为主；三维证据缺失或不可靠时会安全回退到二维结果。

页面中的结果分为三层：

| 结果层 | 含义 |
|---|---|
| 当前姿态评价 | 当前阶段的角度、相对位置和高置信度问题，用于骨架颜色与关节标记 |
| 实时动作反馈 | 当前帧或最近一段动作中的技术问题，不一定影响有效计数 |
| 完整周期判定 | 整次动作候选的必需人体规则和证据质量，输出 `VALID`、`NO_REP` 或 `UNSURE` |

证据不足、身体遮挡、正式指标不可计算或关键点置信度不足时，程序会优先返回 `UNSURE`，而不是给出不可靠的有效或无效结论。拍摄视角仅作为推荐信息：非推荐视角会提示 `CAMERA_VIEW_NOT_RECOMMENDED`，但不会单独改变 `VALID / NO_REP / UNSURE`。具体阈值和规则见 [动作配置说明](configs/hyrox/README.md)。

## 六轮姿态系统改进状态

《HYROX 姿态系统三项问题改进方案》的六轮任务现已全部完成：

| 轮次 | 已实现内容 | 正式规则边界 |
|---|---|---|
| 1 | `DisplayPoseController` 的 `TRACKING / DEGRADED / LOST`、短时保持和淡出 | last-good/display prediction 只用于画骨架 |
| 2 | 逐关键点置信度滞回、骨段局部隐藏、滤波缺口与 reacquisition 指标 | 显示补偿不进入 Rule Engine |
| 3 | `OfflineFastPipeline` 时间戳采样、约 15 pose FPS、无播放节流和最小报告 | Fast 仍为 MediaPipe + HYROX 正式规则 |
| 4 | coarse scan + candidate dense refinement | 候选、`VALID / NO_REP / UNSURE` 语义不变 |
| 5 | `camera_view` 改为 metadata/advice，使用 `CAMERA_VIEW_NOT_RECOMMENDED` | 非推荐或 `unknown` 不会单独触发 `UNSURE` |
| 6 | 统一 `ReliableSideSelector`：关键点置信度、指标完整度、迟滞和快速失效切换 | Lunge 保留前/后腿身份；双腕、双脚和双臂规则保持 bilateral |

Rowing、SkiErg、Sled Push 使用可靠单侧关节链；Lunge 只把选择器用于触地后的伸展链；Wall Ball 在允许单链的拍摄配置中使用选择器，前视/unknown 仍保留双侧正式证据；Sled Pull 的正式周期继续使用双臂聚合，仅输出可靠侧诊断 metadata；Burpee Broad Jump 和 Farmers Carry 明确为 `bilateral_required`。逐侧评分与切换原因位于动作状态的 `debug.reliable_side_selection`。

## 实时性能与低延迟策略

桌面摄像头和有窗口的视频回放采用 Latest-Frame 架构：捕获/播放线程只保留一个最新帧，MediaPipe 忙时覆盖旧待处理帧，不形成长推理队列。每个姿态结果都携带 `frame_id` 和源时间戳；超过帧差或 `analysis_max_pose_age_ms: 120` 的结果不会进入正式规则。浏览器骨架显示与正式分析隔离：短时漏检会保持最后一副显示姿态，随后淡出，但保持、淡出和显示预测都不会进入 HYROX 规则、计数或报告。

显示层对每个关键点独立使用置信度滞回：不可见节点需要高于 `0.50` 才进入显示，已显示节点降到 `0.30` 以下才退出。单个腕、踝等节点不可用时只隐藏依赖该节点的骨段，不会清空整副骨架。One Euro 在空姿态或短于 `250ms` 的缺口中保留滤波状态，达到缺口阈值才重新初始化。实时 JSON 报告的 `summary.display_tracking` 会记录 `pose_detection_rate`、`pose_missing_rate`、`consecutive_missing_ms`、`flicker_count` 和 `reacquisition_ms`；这些字段标记为 `display_only`，不参与正式判定。

默认配置位于 `configs/product_pose.yaml`：

```yaml
pose:
  inference_width: 640
  adaptive_resolution: true
realtime_latency:
  target_pose_fps: 15
  max_pose_fps: 20
  queue_size: 1
  warning_pose_age_ms: 80
  analysis_max_pose_age_ms: 120
  display_prediction_ms: 45
  display_hold_ms: 250
  display_fade_ms: 150
display_smoothing:
  max_gap_ms_before_reset: 250
  min_cutoff: 1.4
  beta: 0.08
  max_raw_weight: 0.10
  landmark_enter_confidence: 0.50
  landmark_exit_confidence: 0.30
  landmark_hold_ms: 220
  pose_hold_frames: 5
  jitter_deadband: 0.0025
```

显示骨架会对单关节使用进入/退出置信度迟滞：短于 `landmark_hold_ms` 的低置信度波动保持最近可靠位置，连续整帧丢检最多保持 `pose_hold_frames` 个姿态推理帧。该保持仅用于画面渲染，不进入角度计算或 HYROX 正式判定。

桌面 `RealtimeBudgetController` 根据滚动 P95 推理耗时、P95 姿态年龄和队列饱和度控制负载。降级顺序固定为 pose FPS `20→15→12`、推理宽度 `640→512→416→320`、可选额外分析频率；视频读取、播放和渲染时钟不会随 MediaPipe 变慢。浏览器实时链路使用本机 Worker 和唯一 pending 槽，拥有独立的 Full/Lite 自动基准与服务器兼容回退，不直接复用桌面控制器。

`--save-metrics` 会输出 `source_fps`、`display_fps`、`inference_fps`、各阶段耗时、时间戳、姿态年龄、读/推理/跳过/渲染帧数、队列深度、`playback_speed_ratio` 及推理/姿态年龄 P50/P95。正常速度回放的工程目标是 `playback_speed_ratio` 接近 `1.0`；具体数值仍需在目标设备实测。

## 安装

支持 CPython `3.10–3.12`。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
python -m pip install -e .
```

按需安装其他依赖：

```powershell
# 可选 YOLO 实验后端
python -m pip install -r requirements-yolo.txt

# 开发、测试和构建
python -m pip install -r requirements-dev.txt
```

RTMW 权重、CPU/GPU 运行时和模型检查命令见 [模型安装说明](models/README.md)。

安装后可先检查运行环境：

```powershell
python -m src.doctor
```

安装命令行入口后也可使用 `pose-doctor --json`。

## 网页版快速开始

双击 `启动网页.bat`，或运行：

```powershell
.\.venv\Scripts\python.exe start_web.py
```

浏览器会打开 `http://127.0.0.1:5000`。选择动作和视频来源，允许摄像头权限后即可开始实时分析；也可以上传视频进行离线分析。上传视频使用 MediaPipe 与现有 HYROX 规则完成快速分析，并按原视频时间轴回放正式骨架与动作结果。

“动作反馈”区域的语音提示默认开启。语音由浏览器 Web Speech API 在当前设备本机合成，不申请麦克风权限，也不会向服务器上传音频。

临时公网分享可双击 `启动公网访问.bat`，需要随机访问口令时运行：

```powershell
.\.venv\Scripts\python.exe start_public_web.py --protected
```

Quick Tunnel 地址每次重启都会变化，仅适合临时测试。正式部署说明见 [网页版使用说明](网页版使用说明.md)。

### 网页版隐私与容量

- 摄像头仅在用户主动授权后启用，并始终使用 `audio: false`；
- 摄像头帧只用于实时推理，不写入服务器磁盘；
- 截图直接下载到当前设备，服务器录制默认关闭；
- 上传视频分析完成后删除，单文件上限 250 MB；
- 分析结果可下载为文字报告、JSON 或 CSV，停止后最多保留 10 分钟；
- 实际并发能力取决于部署硬件与网络。

## 桌面版快速开始

使用摄像头：

```powershell
python main.py --hyrox-action lunge --camera-view side
```

分析视频：

```powershell
python main.py `
  --input-video "HYROX视频\负重箭步蹲.mp4" `
  --hyrox-action lunge `
  --camera-view side `
  --no-mirror
```

无窗口批量运行：

```powershell
python main.py `
  --input-video "HYROX视频\划船机.mp4" `
  --hyrox-action rowing `
  --camera-view side `
  --headless `
  --no-mirror
```

常用参数：

- `--hyrox-action`：选择 HYROX 动作；
- `--hyrox-sensitivity low|medium|high`：设置识别灵敏度；
- `--camera-view front|side|front_left|front_right|unknown`：指定拍摄视角；
- `--hyrox-config PATH`：覆盖动作默认配置；
- `--input-video PATH`：使用视频文件而不是摄像头；
- `--headless`：关闭 OpenCV 窗口；
- `--record PATH` / `--record-raw PATH`：保存标注视频或原始视频；
- `--save-metrics PATH`：导出运行指标 CSV；
- `--hyrox-debug`：显示规则与特征调试信息；
- `--experimental-backends`：显式启用实验后端。

运行时按 `A` 打开动作菜单，按 `N` 切换到下一项动作，按 `V` 切换相机视角。非推荐视角会显示仅作拍摄建议的 `CAMERA_VIEW_NOT_RECOMMENDED`，系统仍按实际可观测关键点和指标继续分析；`unknown` 也不会自动导致 `UNSURE`。运行 `python main.py --help` 查看完整参数，其他桌面窗口快捷键和拍摄建议见 [完整使用说明](使用说明.md)。

回放单个 HYROX 视频时必须明确指定 camera view（拍摄视角），例如：

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\划船机.mp4" --hyrox-action rowing --camera-view side --debug
```

检查多路摄像头的时间偏差时可运行：

```powershell
python tools/check_multicamera.py --camera 0:front --camera 1:side
```

## 计数与输出语义

统一计数字段如下：

- `candidate_count`：检测到的完整动作候选数；
- `pose_valid_rep_count`：人体规则验证通过的动作数；
- `no_rep_count`：有充分证据确认未完成必需人体规则的动作数；
- `unsure_count`：证据不足、无法可靠确认的动作数；
- `cycle_count`：距离类动作的分析周期数；
- `rep_count`：兼容字段，等于 `pose_valid_rep_count`。

对于 Lunge、Wall Ball 和 Burpee Broad Jump：

```text
candidate_count
  = pose_valid_rep_count
  + no_rep_count
  + unsure_count
```

Rowing、SkiErg、Sled Push 和 Sled Pull 的 `cycle_count` 只是动作分析周期，不是官方有效次数。Farmers Carry 使用 `count_semantics: continuous_monitor`，应查看搬运状态、持续时间和技术反馈。

距离类动作的违规码同样只是人体视觉代理，并不表示程序看到了器械或正式赛道判罚：

- Rowing 的 `ROWING_EARLY_STAND_PROXY` 仅在用户开始至停止分析的训练区间内检测持续站起代理；视角或关键点证据不足时输出 `UNSURE`，不输出明确违规；
- Sled Pull 的 `SLED_PULL_KNEELING_VIOLATION` 仅在拉动阶段且跪姿接触证据持续、明确时激活；
- Farmers Carry 的 `ARM_NOT_EXTENDED_VIOLATION` 仅在搬运移动期间检测到手臂持续未基本伸展时激活。

开启 `--hyrox-debug` 后会绘制局部地板线、虚拟膝盖表面点 `K` 和虚拟胸部表面点 `C`。这些点只用于解释二维接触代理，不表示真实接触面积或精确物理距离。

网页版会话、上传和报告默认写入当前目录的 `outputs/`，可用 `POSE_OUTPUT_DIR` 覆盖。桌面版可通过 `--save-dir` 和 `--log-dir` 指定输出位置。

## 人工角度对照

上传视频报告中的 `angle_observations` 会同时保留 raw/filtered 2D、raw/filtered 3D、canonical 3D、屏幕显示角度和正式 selected-rule angle。可先检查并导出指定关节曲线：

```powershell
python tools\inspect_joint_angles.py outputs\report.json `
  --joint left_knee `
  --csv outputs\angle_validation\left_knee_curves.csv
```

在真实视频指定帧上依次点击三个点并保存人工角度。膝角的顺序是髋、膝、踝，第二个点始终是角顶点：

```powershell
python tools\manual_angle_annotation.py input.mp4 `
  --report outputs\report.json `
  --joint left_knee `
  --frame 326 `
  --camera-view side `
  --event lowest_point
```

无图形界面时可增加 `--points "812,342;834,581;902,811"`。完成 30～50 个代表性帧后生成误差和延迟报告：

```powershell
python tools\compare_manual_angles.py `
  outputs\angle_validation\manual_angles.json `
  --report outputs\report.json `
  --baseline-report outputs\old_version_report.json `
  --output-dir outputs\angle_validation
```

输出包含人工对照 MAE、中位绝对误差、P90/P95、最低点/完全伸展事件偏移、原始与平滑角度曲线延迟，以及显式旧版/新版非回归比较。现有 150 条人工标注覆盖 Lunge、Wall Ball、Burpee 和 Rowing，但缺少明确标定的 30°/45°斜侧素材和成对旧新版程序事件帧。当前代理比较中 Median/P90 改善，MAE/P95 分别变差 `0.136°/1.5032°`，严格非回归未通过，所以正式 HYROX 阈值仍使用原 2D 规则。完整结果见 [第 12 轮验证报告](outputs/angle_validation/round12/ROUND12_VALIDATION_REPORT.md)。

重新挖掘现有人工数据时，可直接读取 Round 12 逐条结果：

```powershell
python tools\analyze_existing_angle_errors.py `
  --round12 outputs\angle_validation\round12 `
  --output-dir outputs\angle_validation\round1_error_analysis
```

也可显式传入原始人工标注和帧报告：

```powershell
python tools\analyze_existing_angle_errors.py `
  --annotations outputs\angle_validation\manual_angles.json `
  --report outputs\report.json `
  --round12 outputs\angle_validation\round12
```

该工具会重新计算六路逐点绝对误差，并按动作、关节、左右侧、视角、动作相位、10° 角度区间、关键点置信度和 2D/3D 分歧分桶；同时输出动作/关节/角度区间与动作/关节/相位组合。产物包括机器可读 JSON、逐条 CSV 和 Markdown 结论，不修改正式规则。

人工标签是视频像素上的投影 2D 角，因此报告中的 3D 数值属于投影一致性差距，不是空间 3D 真值 MAE；正面视角的 2D 膝角也不能解释为真实三维关节角。

## 个人参考动作与 DTW 比较

桌面版完成会话后，可从指定时间段创建个人参考动作：

```powershell
pose-reference-inspect --session outputs\sessions\SESSION_ID
pose-reference-create `
  --session outputs\sessions\SESSION_ID `
  --start-ms 1000 `
  --end-ms 5000 `
  --name "我的标准动作" `
  --action-type lunge `
  --camera-view side
pose-reference-list
```

将另一个会话片段与参考动作比较：

```powershell
pose-reference-compare `
  --session outputs\sessions\CANDIDATE_ID `
  --reference outputs\references\REFERENCE_ID `
  --start-ms 1000 `
  --end-ms 5000
```

比较结果默认写入 `outputs/comparisons/`，参考动作默认保存在 `outputs/references/`。

## 开发与验证

```powershell
python -m pytest -q
python tools/check_text_format.py
python -m src.smoke_test
python -m compileall -q hyrox src webui tools main.py
node --check webui\static\app.js
python -m build
```

其他验证入口：

```powershell
# 固定示例黄金回归
pose-golden --report outputs\validation\hyrox_golden_report.json

# 遮挡/跨视角阶段 A 诊断基线（逐关节 CSV、阶段轨迹、难例分类、黄金回归）
pose-occlusion-baseline --output-dir outputs\occlusion_view_phase_a

# 阶段 B 上传显示稳定策略消融（可传单个 --input）
node tools\run_display_stability_ablation.mjs `
  --input-dir outputs\occlusion_view_phase_a\timelines `
  --output outputs\occlusion_view_phase_a\display_stability_ablation.json

# 阶段 C/D 正式证据质量、视角能力矩阵和相位解码报告
pose-occlusion-phase-cd --output-dir outputs\occlusion_view_phase_cd

# 耐久测试
pose-endurance --minutes 30 --report outputs\validation\endurance_30m.json

# 摄像头后端基准
pose-camera-benchmark --help

# Angle V2：30 条手机 RGB 全量 shadow 回放
pose-angle-v2-shadow --scope all

# Angle V2：用现有人工标签做有界参数 sweep
pose-angle-v2-sweep

# 游泳第 4、5 轮：持久左右腕身份 + LK 短时轨迹补偿
pose-swim-wrist-track ".\游泳视频.mp4"

# 游泳第 6 轮：现有标记视频上的五模式腕带 + CoTracker 离线比较
pose-swim-round6 --output-dir outputs\swim_wrist_round6

# 输出清理（默认仅预览）
pose-clean --json
```

Angle V2 的质量门、关节组平滑、异常剔除、端点、迟滞和时序证据均仅在
shadow 链路运行，不改变正式 HYROX 2D 规则。回放报告位于
`outputs/angle_validation/angle_v2_round2/`，参数 sweep 位于
`outputs/angle_validation/angle_v2_round3_sweep/`；没有独立验证集时工具会明确禁止
替换正式默认值。

游泳腕部工具把 MediaPipe 的 `left_wrist/right_wrist` 当作每帧候选，不直接当作永久
身份。它维护独立的 anatomical left/right track，以恒速 Kalman、肩—肘—腕骨链、
2×2 Hungarian 等价全局分配和三帧迟滞处理标签交换；短时缺失使用带 forward/backward
一致性检查的 LK 光流，轨迹跳点由身体尺度归一化的 median/MAD 门拒绝。结果写入
`outputs/swim_wrist_tracking_round4_5/`。本轮不使用腕带外观或 CoTracker，也不修改
HYROX `ReliableSideSelector`。

第 6 轮在腕部动态 ROI 中提取 HSV/Lab 直方图与饱和度，并仅在身份和可见度高置信时
以 EMA 更新左右 appearance prototype；可选 CoTracker 后端只在本地包与权重可用时
离线运行，默认不隐式下载。当前两段标记视频共 3,806 帧、26 个人工腕带中心锚点：
Pose+LK 的 identity-switch proxy 为 16（MediaPipe only 为 186），锚点身份正确率
57.69%，平均 coverage 89.71%，body-normalized jitter 0.04069。CoTracker 本机缺包/
权重，因此相关两项明确记为 unavailable；不估算、不用 LK 冒充。实验默认保持
Pose+LK，正式默认未修改；完整报告位于 `outputs/swim_wrist_round6/`。

实验、数据处理、消融、性能优化和各阶段验证的历史结果不在 README 展开，统一记录在 [CHANGELOG.md](CHANGELOG.md) 及对应研究报告中。

## 代码结构

```text
hyrox/                   # HYROX 动作分析器与通用规则
  reliable_side.py       # 与 camera_view 无关的左右侧可靠性评分和迟滞选择
configs/hyrox/           # 动作配置
src/backends/            # 姿态后端
src/biomechanics/        # 通用运动学数据
src/swimming/            # 独立的游泳腕部身份、Kalman、LK、appearance 与 CoTracker 适配
src/realtime/            # 桌面运行时
  budget.py              # 自适应推理预算控制，不改变视频/渲染时钟
src/validation/          # 回归、性能与耐久验证
webui/                   # 网页后端和前端资源
tools/                   # 回放、检查和研究工具
tests/                   # 自动化测试
```

## 限制

- 所有结论均来自人体关键点和二维视觉代理，不是医疗诊断；
- Wall Ball 不检测药球、目标高度或是否命中；
- Sled Push / Pull 不检测器械、真实负载或是否过线；
- Rowing / SkiErg 不读取器械阻力、功率或距离；
- Farmers Carry 不检测壶铃重量、真实距离或完成线；
- 拍摄视角、遮挡、光照、快速运动和多人干扰会影响识别质量；
- 自动化测试不能替代目标设备上的真实摄像头、性能和端到端延迟验收；
- 第 12 轮尚缺标定的 30°/45°角度素材和空间 3D 真值，当前严格角度非回归未通过；
- 临时公网分享不等同于正式生产部署。

## 文档导航

- [完整使用说明](使用说明.md)：网页版、桌面版、拍摄、计数语义和故障排查；
- [网页版使用说明](网页版使用说明.md)：浏览器摄像头、匿名会话、隐私、报告和部署；
- [模型安装说明](models/README.md)：模型文件、可选后端与 CPU/GPU 环境；
- [动作配置说明](configs/hyrox/README.md)：阈值、触地/脚部事件和可观测性；
- [变更记录](CHANGELOG.md)：功能、实验、性能和验证历史；
- [算法模型与数据现状](项目算法模型与数据现状说明.md)：当前技术边界、数据角色和可用性；
- [发布与升级说明](RELEASING.md)：版本规则、构建、发布和 schema 兼容策略。
