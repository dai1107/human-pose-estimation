# HYROX 姿态分析

本项目提供两种使用方式：

- **网页版**：在电脑或手机浏览器中使用本机摄像头，支持多人匿名会话、实时骨架与动作反馈、逐次动作语音改进提示、上传视频分析，以及文字/JSON/CSV 报告下载。
- **桌面版**：通过 OpenCV 窗口使用摄像头或视频文件，适合本机调试、录像、指标导出和模型对比。

当前聚焦 8 项 HYROX 动作的人体姿态识别、人体规则有效计数、距离动作分析周期和实时指导；独立深蹲与篮球投篮分析模式已移除。网页版可手动选择纯 MediaPipe Pose、`YOLO + MediaPipe` 或 `YOLO + RTMW WholeBody`，另有按动作选择标准后端的自动模式；实时关键点默认使用 One Euro 平滑，参考动作比较支持 DTW 对齐。

## 文档导航

- [完整使用说明](使用说明.md)：网页版、桌面版、拍摄、计数语义、调试与故障排查；
- [网页版使用说明](网页版使用说明.md)：浏览器摄像头、匿名会话、隐私、报告和部署；
- [模型安装说明](models/README.md)：MediaPipe、手部模型、RTMW 权重与 CPU/GPU 环境；
- [动作配置说明](configs/hyrox/README.md)：默认阈值、触地/脚部事件、可观测性与自定义配置；
- [变更记录](CHANGELOG.md)：版本变更摘要；
- [发布与升级说明](RELEASING.md)：版本规则、构建、发布和 schema 兼容策略；
- [程序成熟度审计与实施清单](程序成熟度审计与实施清单.md)：已完成工程化项目、当前限制与后续优先级。

## 安装与命令行入口

支持 CPython `3.10–3.12`。依赖文件使用已验证的精确版本，并按用途拆分：

```powershell
# MediaPipe、网页和桌面核心
python -m pip install -r requirements-core.txt

# 可选 YOLO 后端
python -m pip install -r requirements-yolo.txt

# 开发、测试和构建
python -m pip install -r requirements-dev.txt

# 安装本项目及命令行入口；需要 YOLO 时改用 .[yolo]
python -m pip install -e .
```

安装后可从任意工作目录使用：

```powershell
pose-doctor --json
pose-estimation --help
pose-web --no-browser
pose-replay --help
pose-clean --json
pose-reference-create --help
pose-reference-list
pose-reference-inspect --help
pose-reference-compare --help
pose-reference-export --help
```

发布配置位于 `pyproject.toml`。`python -m build` 会在 `dist/` 生成 wheel 和源码包；
模型、默认 YAML 和网页静态资源会进入发布包，不会把运行结果写入 `site-packages`。
网页版会话、上传和报告默认写到当前目录的 `outputs/`，可用 `POSE_OUTPUT_DIR` 覆盖；
桌面版使用 `--save-dir` / `--log-dir`，参考动作命令使用 `--output-dir` / `--root`
指定位置。`pose-doctor` 与 `pose-clean` 会跟随 `POSE_OUTPUT_DIR`。

## 支持的动作

| 参数 | 动作 | 主要输出 |
|---|---|---|
| `lunge` | 负重箭步蹲 | 五项人体规则、有效/未完成/不确定计数与技术提示 |
| `wall_ball` | Wall Ball | 四项人体规则、有效/未完成/不确定计数与深度提示 |
| `rowing` | Rowing | 划船分析周期、技术提示与训练区间站起违规代理 |
| `skierg` | SkiErg | 拉动分析周期与髋铰链提示 |
| `burpee_broad_jump` | Burpee Broad Jump | 八项人体规则、延迟验证计数与落地提示 |
| `sled_push` | Sled Push | 推行状态、分析步态周期与技术提示 |
| `sled_pull` | Sled Pull | 拉动分析周期、技术提示与跪姿/坐姿违规代理 |
| `farmers_carry` | Farmers Carry | 连续搬运监控、稳定性提示与手臂位置违规代理 |

## 动作问题如何判断

程序不会让姿态模型直接生成“膝盖内扣”或“伸展不足”等结论，也不会自动猜测用户正在做哪项动作。用户先选择 HYROX 动作，系统再创建该动作专属分析器；MediaPipe、YOLO Pose 或 RTMW 只负责输出人体关键点，问题定位和计数主要由可复现的时序规则完成。

```text
视频帧
  → 人体关键点识别与 One Euro 平滑
  → 计算关节角度、相对位置、可见度、触地和脚步事件
  → 动作专属状态机确认当前阶段
  → 当前阶段技术反馈 + 完整周期人体规则验证
  → 可观测性门控
  → 骨架颜色、关节角度、文字/语音反馈和计数结果
```

完整处理过程如下：

1. **识别并平滑关键点**：姿态后端输出肩、肘、腕、髋、膝、踝、脚跟和脚尖等关键点及置信度；实时流经过 One Euro 平滑，减少单帧抖动和短时漏检。
2. **计算二维运动学特征**：系统计算左右膝/髋/肘/肩角度、躯干角、肩髋高差、髋膝相对高度、手腕与肩髋位置、身体中心移动、人体尺度归一化距离和各身体区域可见度。需要触地或起落判断的动作还会使用局部地板、膝/胸虚拟表面、左右脚支撑、起跳、落地和碎步事件。
3. **识别动作阶段**：每个动作有独立状态机，例如 Lunge 使用 `stand → descent → bottom → ascent → stand`，Rowing 使用 `catch → drive → finish → recovery`。规则只在适用阶段运行，阶段切换还会经过时序、端点顺序和冷却门控；离线分析和“低”灵敏度通常要求连续多帧，实时“中/高”灵敏度允许一个已处理帧确认短暂关键端点。
4. **生成实时技术反馈**：动作分析器将当前特征与动作配置阈值比较，输出带问题码、严重级别和置信度的中文提示。例如 Lunge 最低点不够深会输出 `NOT_DEEP_ENOUGH`，躯干倾斜过大会输出 `LEAN_TOO_MUCH`。默认每帧最多显示两项；可见度不足时只显示取景提示，不继续给出不可靠的技术结论。
5. **验证完整动作候选**：Lunge、Wall Ball 和 Burpee Broad Jump 在关键端点顺序完成后，分别检查该动作的必需人体规则。每条规则先得到 `PASS`、`FAIL`、`UNSURE` 或 `NOT_APPLICABLE`；任一必需规则明确 `FAIL` 得到 `NO_REP`，没有失败但存在无法判断的必需规则得到 `UNSURE`，全部通过才得到 `VALID`。技术质量提示与计数必需规则相互独立，训练建议不一定取消计数。
6. **执行可观测性门控**：规则聚合后还会检查整次可见度、必需关键点、决定性规则置信度、地板、拍摄视角和失败持续帧数。证据不足时，即使初步结果为 `VALID` 或 `NO_REP`，也会降级为 `UNSURE`。
7. **呈现问题位置**：网页把当前阶段适用的角度标在对应关节附近；明显超出参考范围或存在高置信度问题时显示红色，通过当前标准时显示绿色，接近边界、质量不足或无法评价时显示黄色/中性。某些没有明确姿态标准的过渡阶段也会保持绿色，但不会计入报告的可评价帧。非角度问题（例如补步、触地、左右不同步）主要通过“动作反馈”和报告说明。

页面中的三类结果含义不同：

| 结果层 | 判断对象 | 用途 |
|---|---|---|
| 当前姿态评价 | 当前阶段的角度、归一化位置和高置信度问题 | 控制骨架颜色和关节角度标记；边界值不会直接判红或判绿 |
| 实时动作反馈 | 当前帧或最近一段动作命中的技术问题 | 给出最多两条优先改进提示；不一定影响有效计数 |
| 完整周期判定 | 整次候选的必需人体规则及证据质量 | 生成 `VALID`、`NO_REP`、`UNSURE` 和对应计数字段 |

以默认“中”灵敏度的负重箭步蹲为例：全身可见度达到要求后，最小膝角约 `≥150°` 识别为站立，约 `≤115°` 并结合髋部下降判断最低点；最低点若膝角仍 `>100°` 会提示下蹲幅度不足，躯干绝对角 `>20°` 会提示前倾过多。完整周期结束时还必须确认后膝触地、触地后膝髋约 `≥165°` 并连续保持、触地腿交替且没有额外调整步。上述实时提示阈值会随灵敏度变化，最终配置值以 [HYROX 动作配置说明](configs/hyrox/README.md) 和各动作 YAML 为准。

## 计数与分析周期标准

下表描述当前程序如何解释二维人体关键点。Lunge、Wall Ball 和 Burpee Broad Jump 会形成需要人体规则验证的候选；Rowing、SkiErg、Sled Push 和 Sled Pull 只记录分析周期；Farmers Carry 是连续监控。所有输出均不等同于 HYROX 正式比赛裁判结论。

| 动作 | 当前程序算作一次的关键顺序 | 建议视角 |
|---|---|---|
| 负重箭步蹲 | 站立 → 明确后膝触地 → 触地后双髋双膝完全伸展；与上一条有效触地腿交替且无额外调整步 | 侧面或斜侧面，全身和脚下地板入镜 |
| Wall Ball | 双髋双膝站直 → 髋部明确低于膝部 → 双髋双膝向上完全伸展 → 双腕同步举过双肩；四项人体规则全部通过后计 1 次 | 正面或斜前方，全身、地板、双脚和双手腕持续入镜 |
| Rowing | Catch → Finish；Drive 可作为过渡相位。只增加分析周期，不是官方有效次数 | 侧面 |
| SkiErg | 顶部 → 下拉底部 → 回到顶部。只增加分析周期，不是官方有效次数 | 正面或斜前方 |
| Burpee Broad Jump | 胸部代理触地确认 → 双脚同步起跳 → 双脚同步落地 → 下一次双手开始触地时完成验证；还要求起落错位、手部位置、无补步及前向位移规则全部通过 | 侧面约 45°，全身、双手、双脚和下一落地区域持续入镜 |
| Sled Push | Drive → Step；每个推动步只记为分析周期，不是官方有效次数 | 侧面或斜侧面 |
| Sled Pull | Reach → Pull → Recover → Reach；后拉和随后的向前回正合为一个分析周期，不是官方有效次数 | 侧面或斜侧面 |
| Farmers Carry | 连续监控，不按次数拆分，`cycle_count` 与兼容 `rep_count` 均保持为 0 | 正面或斜前方，全身入镜 |

实时摄像头在“中/高”灵敏度下允许一个已处理帧确认关键端点，以免推理队列丢帧时错过快速顶点；完整端点顺序和计数冷却仍用于防止重复计数。Wall Ball 测试时应先完整站立，再下蹲并把手腕举过肩部；若选择“低”灵敏度，端点需要连续多帧，计数会更保守。具体默认角度和距离阈值见 [HYROX 动作配置说明](configs/hyrox/README.md#默认中灵敏度计数与分析周期端点)。

每个完整动作周期会先进入人体规则验证层，再得到 `VALID`、`NO_REP` 或 `UNSURE`。统一计数字段为 `candidate_count`（动作周期）、`pose_valid_rep_count`（有效动作）、`no_rep_count`（未完成）和 `unsure_count`（无法确认）；兼容字段 `rep_count` 始终等于 `pose_valid_rep_count`。膝内扣、轻微前倾和小幅不对称等技术提示不会单独取消计数。

| 动作类别 | 字段解释 |
|---|---|
| Lunge / Wall Ball / Burpee Broad Jump | `candidate_count` 是人体规则候选数；三种互斥结果满足 `candidate_count = pose_valid_rep_count + no_rep_count + unsure_count`。`pose_valid_rep_count` 仍只是二维人体序列有效，不是官方完赛计数。 |
| Rowing / SkiErg / Sled Push / Sled Pull | `cycle_count` 是划动、拉动或推动步分析周期；通用计数字段为兼容现有界面保留，必须结合 `count_semantics: analysis_cycle` 与 `official_rep_count_supported: false` 解读。 |
| Farmers Carry | `count_semantics: continuous_monitor`；`cycle_count` 和 `rep_count` 保持 0，应查看 `carrying/rest`、技术反馈和持续违规状态。 |

最终计数前还会执行统一可观测性门控。整次动作平均可见度低于 `0.65`、必需关键点置信度低于 `0.60`、决定性规则置信度低于 `0.72`、局部地板失效、已知拍摄视角不适合或失败只出现一个异常帧时，结论统一降级为 `UNSURE`。逐规则结果仍保留原始 `PASS/FAIL`，降级依据可在 `last_rep_observability` 查看；持续多帧且证据充分的明确失败仍记为 `NO_REP`。

负重箭步蹲已接入五项必需人体规则：后膝触地、触地后的膝部伸展、触地后的髋部伸展、有效触地腿交替，以及无额外调整步。系统会综合人体前进方向、脚位置和双膝离地高度确定前后腿；清晰的近地膝证据可纠正侧视图中的左右误配。正面证据要求双侧伸展；推荐的侧面视角使用置信度更高的同侧腿链并保留 3° 二维测量容差。只有 `VALID` 动作会更新上一条触地腿，遮挡或地板不可靠会记为 `UNSURE`。

Burpee Broad Jump 已接入八项必需人体规则：通用胸部触地代理、双脚同步起落、起跳与落地错位代理、`LEGAL_HAND_PLACEMENT_PROXY`、落地后无额外步或碎步，以及身体中心和双脚均发生同向前移。落地只生成待验证候选并显示 `AWAITING_NEXT_HANDS`；系统会继续观察到下一次 hands-down/chest-down，再把该候选记为 `VALID`、`NO_REP` 或 `UNSURE`。脚部错位和手部位置都按本人脚长归一化，不代表精确测得官方厘米距离。

Wall Ball 已接入四项必需人体规则：站直起始、基于局部地板的髋低于膝、投掷端点向上完全伸展，以及 `BILATERAL_THROW_PROXY`。双腕代理要求双手从胸部附近同步上升并均到达肩部以上，峰值时间差采用 `120/220 ms` 视频容差；它只能确认双手投掷样式，不能确认球、目标或命中结果。地板或任一手腕证据不足时记为 `UNSURE`。

距离动作不产生人体姿态意义上的官方有效次数。Rowing、SkiErg、Sled Push 和 Sled Pull 的兼容 `rep_count` 只代表动作分析周期，并同时输出 `cycle_count`、`count_semantics: analysis_cycle` 与 `official_rep_count_supported: false`；Farmers Carry 使用 `count_semantics: continuous_monitor`。人体违规检测只覆盖可见且规则明确的部分：Rowing 在用户开始至停止分析的训练区间内检测持续站起代理 `ROWING_EARLY_STAND_PROXY`；Sled Pull 检测 `SLED_PULL_KNEELING_VIOLATION`、`SLED_PULL_SEATED_VIOLATION`，证据不足的可能坐姿输出 `UNSURE_POSSIBLE_SEATED_PULL`；Farmers Carry 检测 `ARM_NOT_EXTENDED_VIOLATION` 和 `ARM_NOT_BY_SIDE_VIOLATION`。所有明确违规均要求持续多帧，关键点或视角不足时降级为 `UNSURE`。SkiErg 不因脚离地判违规，Sled Push 不因固定关节角度或躯干姿势判违规。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

主要模型文件：

- `models/pose_landmarker_full.task`：MediaPipe Pose；
- `models/hand_landmarker.task`：可选手部关键点；
- `yolo11n-pose.pt`：YOLO11n Pose；
- `models/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx`：RTMW-X/L WholeBody 133 点，约 229 MB，不提交到 Git；
- `yolo11n.pt`：可选人体检测器。

RTMW 属于可选高精度依赖。安装 CPU 或 NVIDIA GPU 运行时（二选一）：

```powershell
# CPU
.venv\Scripts\python.exe -m pip install -r requirements-rtmw-cpu.txt

# NVIDIA GPU
.venv\Scripts\python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu
.venv\Scripts\python.exe -m pip install -r requirements-rtmw-gpu.txt
```

权重下载地址、GPU 环境和 Provider 检查命令见 [模型安装说明](models/README.md)。

## 网页版识别模型

高级设置中有三种可手动选择的姿态方案；“自动选择”只是按动作在标准后端之间选择，不是第四种模型，也不会自动启用计算量较大的 RTMW。

| 页面选项 | 关键点与用途 | 运行说明 |
|---|---|---|
| `纯 MediaPipe` | 33 个姿态点，速度快，适合普通实时分析 | 始终只运行 MediaPipe 姿态后端，不会按动作隐式加载 YOLO；可使用 CPU |
| `YOLO + MediaPipe` | YOLO 锁定目标人物并输出核心身体点，MediaPipe 为同一人补充脚跟和脚尖 | 所有动作都固定运行双模型同人融合，适合多人画面中的目标跟踪 |
| `YOLO + RTMW WholeBody（高精度 133 点）` | YOLO 锁定人物，RTMW 在该人物框中输出身体、脚部、面部和双手关键点 | 推荐 NVIDIA GPU；适合需要脚跟、脚尖和手指细节的 Lunge、Wall Ball |
| `自动选择（推荐）` | 根据动作选择标准后端；拥挤的 Lunge 会使用 YOLO + MediaPipe | 不算独立模型；如需 RTMW，应手动选择 |

RTMW 结果会用最多 13 个 YOLO/RTMW 共有身体点复核身份，只有匹配成功后才接受 133 点结果，避免把背景人物的脚部或手部节点用于当前运动员。RTMW 权重、ONNX Runtime 或初始化不可用时，页面会明确显示 `YOLO + MediaPipe（RTMW 降级）`，而不是中断分析。

网页选择 `YOLO + MediaPipe` 时，会在所有动作中使用融合后端：YOLO 先锁定前景运动员，MediaPipe 最多输出 5 个人体候选，再用鼻、肩、肘、腕、髋、膝和踝的共有节点做身份匹配。髋、膝、踝等核心动作节点保留 YOLO 结果，脚跟和脚尖只从匹配成功的同一人补入；匹配失败时不会改用背景人物。短时漏检只保存脚部相对当前 YOLO 脚踝的偏移，并带置信度衰减和过期限制。

## 网页版快速开始

本机使用可双击 `启动网页.bat`，或在命令行运行：

```powershell
.venv\Scripts\python.exe start_web.py
```

程序会打开 `http://127.0.0.1:5000`。选择“本机摄像头”，点击“开启摄像头”并允许浏览器的视频权限，然后点击“开始实时分析”。

### 逐次动作语音提示

“动作反馈”右上角的“语音开/关”默认开启。系统在一次人体规则候选或分析周期完成时，汇总该段动作中持续出现的问题以及清晰关键阶段的角度偏离，立即播报最多两条改进建议；没有持续性问题时不会播放无意义提示。农夫行走不按次数拆分，会在动作问题持续约 1.2 秒后播报，同一问题至少间隔 8 秒才会再次提示。

语音使用浏览器 Web Speech API 在当前设备本机合成，不申请麦克风权限、不向服务器上传音频，也不会写入录制视频。开关选择会保存在当前浏览器；若浏览器不支持语音合成，页面会显示“语音不可用”。

页面的“高级设置”中，“骨架设置”只选择完整骨架、仅上半身或仅下半身；“显示手指节点”“隐藏面部”和“镜像画面”是与其分开的同级开关。选择 RTMW WholeBody 时，“显示手指节点”直接使用 RTMW 的双手各 21 点；其他姿态模型会按约 10 FPS 运行独立的 MediaPipe Hand Landmarker。页面为每只手绘制五根手指的 20 个非手腕关节点，手部短暂漏检时最多保留约 0.35 秒以减少闪烁。手指节点默认显示，隐藏面部默认关闭；摄像头模式默认开启镜像，示例和上传视频默认关闭镜像。

系统优先在用户站直、双脚稳定且全身完整入镜时，用最近约 0.5 秒的脚跟和脚尖位置自动估计局部地板线；负重箭步蹲和 Wall Ball 若视频没有完整站姿，也可由连续稳定的支撑脚建立地板线，但不会把屈膝姿态误用为站立身高。摄像头明显倾斜时，可在“高级设置 → 手动地板线”中依次点击脚下地板的两个点。该标定只用于归一化人体部位的离地高度，不识别赛道，也不测量真实厘米距离；足部被遮挡、人体出画或相机移动时，地板参考会标记为 `UNSURE`。

通用触地检测器在可靠地板参考之上，综合虚拟膝盖/胸部表面距离、动作阶段、垂直速度、局部最低点、持续时间和关键点置信度，并通过进入/退出双阈值防止临界位置抖动。MediaPipe 会使用可选人体分割辅助胸部判断；YOLO Pose 或没有分割的帧仍可运行，但胸部代理置信度封顶为 `0.74`。该结果是二维视觉代理，不能等同于正式裁判对真实乳头线触地的判定。

脚部事件检测器分别跟踪左右脚的支撑、起跳候选、腾空、落地候选和重新支撑状态，同时使用脚跟与脚尖，避免脚尖仍着地时误报离地。双脚同步起落使用 `100/180 ms` 视频容差；前后错位结果固定命名为 `FOOT_STAGGER_PROXY`，按本人脚长归一化，不能视为官方 5 厘米的精确测量。独立碎步事件必须同时满足有效腾空时间、落地稳定时间和腿长归一化位移。

临时公网分享可双击 `启动公网访问.bat`。默认创建匿名 HTTPS 地址；需要随机访问口令时运行：

```powershell
.venv\Scripts\python.exe start_public_web.py --protected
```

Quick Tunnel 的地址每次重启都会变化，只适合临时测试。正式部署仍需长期域名、HTTPS、支持 WebSocket 的生产服务和进程守护。完整说明见 [网页版使用说明.md](网页版使用说明.md)。

### 网页版隐私与容量

- 摄像头只在用户点击按钮后授权，始终使用 `audio: false`，不请求麦克风；
- 摄像头帧仅用于实时推理，不写入服务器磁盘；
- 截图直接下载到当前设备；服务器录制已关闭；
- 上传视频分析完成后删除，单文件上限 250 MB、单会话临时空间上限 500 MB；
- 分析结果可下载为文字报告、JSON 或 CSV，停止后最多保留 10 分钟；
- 默认容量上限为 100 个匿名网页会话、50 个实时分析会话，实际并发能力取决于部署硬件与网络。

## 桌面版快速开始

摄像头实时识别：

```powershell
python main.py --hyrox-action lunge --camera-view side
```

读取视频：

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

桌面窗口中按 `A` 打开动作菜单、按数字键选择动作、按 `N` 切换动作、按 `V` 切换相机视角。`S` 截图、`R` 录制标注视频、`T` 录制原始视频仅属于桌面版；网页版不会在服务器保存这些画面。

## 常用桌面参数

- `--hyrox-action`：选择 HYROX 动作，默认 `none`；
- `--hyrox-sensitivity low|medium|high`：动作识别灵敏度；
- `--hyrox-config PATH`：覆盖动作默认配置；
- `--camera-view front|side|front_left|front_right|unknown`：拍摄视角；
- `CAMERA_VIEW_LIMITED`：当前拍摄视角不足以可靠判断某项动作标准时的提示码；
- `--backend auto|mediapipe|yolo-pose`：桌面版姿态后端；RTMW WholeBody 当前通过网页版高级设置使用；
- `--input-video PATH`：使用视频而不是摄像头；
- `--hyrox-debug`：显示 HYROX 特征调试面板；
- `--headless`：关闭 OpenCV 窗口；
- `--record PATH` / `--record-raw PATH`：保存标注或原始视频；
- `--save-metrics PATH`：输出运行指标 CSV。

查看完整参数：

```powershell
python main.py --help
```

`--hyrox-debug` 调试面板会绘制局部地板线、虚拟膝盖表面点 `K` 和虚拟胸部表面点
`C`，并显示两者的离地高度、左右脚支撑状态、起跳/落地时间差、脚部错位比例、当前
候选的逐规则 `PASS/FAIL/UNSURE` 列表及最终 `VALID/NO_REP/UNSURE`。这些数值用于
解释二维视觉代理的判定过程；虚拟表面点和归一化高度不是实际人体接触面积或厘米测量。

## 视频回放工具

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\药球.mp4" --hyrox-action wall_ball --camera-view front
```

多摄像头时间同步检查可运行 `python tools/check_multicamera.py --camera 0:front --camera 1:side`。正式使用前先运行 `python -m src.doctor` 做模型和运行环境健康检查。

## 个人参考动作与 DTW 比较

桌面版按 `C` 开始/结束会话后，会在 `outputs/sessions/SESSION_ID/` 生成关键点、运动学、
摘要和元数据。可先检查质量，再从指定时间段创建个人参考动作：

```powershell
pose-reference-inspect --session outputs\sessions\SESSION_ID
pose-reference-create --session outputs\sessions\SESSION_ID --start-ms 1000 --end-ms 5000 --name "我的标准动作" --action-type lunge --camera-view side
pose-reference-list
```

将另一个会话片段与参考动作做归一化和 DTW 对齐：

```powershell
pose-reference-compare --session outputs\sessions\CANDIDATE_ID --reference outputs\references\REFERENCE_ID --start-ms 1000 --end-ms 5000
pose-reference-export --reference outputs\references\REFERENCE_ID
```

比较结果默认写入 `outputs/comparisons/`，参考动作默认位于 `outputs/references/`。
创建和比较命令分别支持 `--output-dir`；列表命令支持 `--root`。参考配置也采用严格
校验，输出沿用统一的 `schema_version` / `program_version` 字段。

## 配置校验、日志与退出码

启动桌面版或视频回放时会严格校验 YAML：未知字段、重复字段、错误类型、越界值、
错误 `action_name` 和不支持的 YAML 结构都会立即停止启动，并显示 `CFG001` 错误。
`python -m src.doctor` 同时验证 8 个动作配置、3 个 HYROX 共享配置和 2 个参考动作配置。
自定义动作配置仍可只写需要覆盖的字段，但字段本身必须合法。

桌面版、网页版和视频回放默认把运行信息写入 `outputs/logs/` 下的滚动日志，单个日志
最大 2 MiB，保留 5 个备份。控制台只显示友好错误；需要 traceback 时加 `--debug`。
可用 `--log-dir` 修改日志目录。

| 退出码 | 含义 | 常见错误编号 |
|---:|---|---|
| `0` | 正常完成 | - |
| `1` | 未预期运行错误 | `RUN001` |
| `2` | 参数或配置错误 | `CFG001`–`CFG004` |
| `3` | 摄像头/视频输入错误 | `SRC001` |
| `4` | 姿态后端初始化错误 | `BCK001` |
| `5` | 日志、视频、CSV 或会话输出错误 | `OUT001`–`OUT003` |
| `130` | 用户中断；程序仍会尝试保存活动会话并关闭资源 | `RUN130` |

会话保存先写入 `write_status: partial` 元数据，再生成 CSV、报告和最终
`write_status: complete`。若磁盘写入中途失败，已生成文件会保留，`metadata.json`
会尽量记录 `recovery_error` 和 `recovered_files`，不会把不完整会话伪装成完整结果。

结构化 JSON 和 CSV 统一包含 `schema_version` 与 `program_version`；JSON 另有
`artifact_type`。没有 schema 的历史文件按版本 `0` 兼容读取，高于当前程序支持版本的
文件会以 `SCH001` 拒绝，避免静默误读。

输出清理默认为只预览。各目录默认保留 2–90 天，也可设置总容量上限：

```powershell
# 只预览，不删除
pose-clean --json
pose-clean --older-than-days 30 --max-total-gb 20

# 确认列表后才实际删除
pose-clean --older-than-days 30 --max-total-gb 20 --apply
```

清理器仅允许处理 `outputs/` 下已登记的生成目录，拒绝文件系统根目录和未知目录名，
并按会话/报告顶层目录整体删除，避免留下半个活动会话。

## 代码结构

```text
hyrox/                   # 8 个 HYROX 动作分析器与通用逻辑
configs/hyrox/           # 动作配置
src/backends/            # MediaPipe / YOLO Pose / YOLO + MediaPipe / YOLO + RTMW
src/biomechanics/        # 通用运动学数据
src/configuration.py     # 受限 YAML 解析与严格配置校验
src/output_schema.py     # JSON/CSV schema 和程序版本字段
src/output_management.py # 安全的输出保留期与容量清理
src/runtime_logging.py   # 错误分类、退出码与滚动日志
src/realtime/            # 桌面实时反馈
webui/                   # 网页后端、实时会话、前端页面与静态资源
start_web.py             # 本机网页入口
start_public_web.py      # 临时公网入口
tools/                   # 回放、检查与隧道工具
tests/                   # 自动化测试
```

## 验证

```powershell
python -m src.doctor
python -m pytest -q
python tools/check_text_format.py
python -m src.smoke_test
python -m compileall -q hyrox src webui tools main.py
node --check webui\static\app.js
python -m build
```

Windows/Linux CI 会在 Python 3.10 和 3.12 上执行依赖安装、导入、编译、文本格式、
全量单测、无摄像头冒烟和发布包构建。自动化测试还覆盖通用接触检测、脚部事件、
`VALID/NO_REP/UNSURE` 可观测性门控、三项人体有效计数动作、距离动作违规边界、
输出 schema/保留策略和相同特征流重复运行确定性。当前全量基线为 `400 passed`。

## 限制

- 当前结论来自人体关键点，是视觉运动学代理，不是医疗诊断；
- Wall Ball 不检测药球、目标命中或目标高度；
- Sled Push / Pull 不检测器械或真实负载；
- Rowing / SkiErg 不读取器械阻力、功率或距离；
- 距离动作的 `cycle_count` 不是官方有效次数，系统不确认雪橇过线、Rowing/SkiErg 完成里程或训练区间外的规则状态；
- Farmers Carry 不检测壶铃重量、真实携带距离或完成线；
- 拍摄视角、遮挡、光照和多人干扰会影响识别质量；
- 网页版虽已完成会话隔离和容量限制，但尚未完成实体设备兼容验收、50 路压力测试和正式公网部署。
