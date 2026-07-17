# HYROX 姿态分析

本项目提供两种使用方式：

- **网页版**：在电脑或手机浏览器中使用本机摄像头，支持多人匿名会话、实时骨架与动作反馈、逐次动作语音改进提示、上传视频分析，以及文字/JSON/CSV 报告下载。
- **桌面版**：通过 OpenCV 窗口使用摄像头或视频文件，适合本机调试、录像、指标导出和模型对比。

当前聚焦 8 项 HYROX 动作的人体姿态识别、人体规则有效计数、距离动作分析周期和实时指导；独立深蹲与篮球投篮分析模式已移除。姿态后端支持 MediaPipe Pose 与 YOLO11n Pose，实时关键点默认使用 One Euro 平滑，参考动作比较支持 DTW 对齐。

## 文档导航

- [完整使用说明](使用说明.md)：网页版、桌面版、拍摄、计数语义、调试与故障排查；
- [网页版使用说明](网页版使用说明.md)：浏览器摄像头、匿名会话、隐私、报告和部署；
- [动作配置说明](configs/hyrox/README.md)：默认阈值、触地/脚部事件、可观测性与自定义配置；
- [人体计数与违规任务清单](HYROX%20人体动作计数与违规检测任务清单.md)：第 1–11 轮实现和验收记录。

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
| Sled Pull | Reach → Pull → Recover；每个完整拉回只记为分析周期，不是官方有效次数 | 侧面或斜侧面 |
| Farmers Carry | 连续监控，不按次数拆分，`cycle_count` 与兼容 `rep_count` 均保持为 0 | 正面或斜前方，全身入镜 |

实时摄像头在“中/高”灵敏度下允许一个已处理帧确认关键端点，以免推理队列丢帧时错过快速顶点；完整端点顺序和计数冷却仍用于防止重复计数。Wall Ball 测试时应先完整站立，再下蹲并把手腕举过肩部；若选择“低”灵敏度，端点需要连续多帧，计数会更保守。具体默认角度和距离阈值见 [HYROX 动作配置说明](configs/hyrox/README.md#默认中灵敏度计数与分析周期端点)。

每个完整动作周期会先进入人体规则验证层，再得到 `VALID`、`NO_REP` 或 `UNSURE`。统一计数字段为 `candidate_count`（动作周期）、`pose_valid_rep_count`（有效动作）、`no_rep_count`（未完成）和 `unsure_count`（无法确认）；兼容字段 `rep_count` 始终等于 `pose_valid_rep_count`。膝内扣、轻微前倾和小幅不对称等技术提示不会单独取消计数。

| 动作类别 | 字段解释 |
|---|---|
| Lunge / Wall Ball / Burpee Broad Jump | `candidate_count` 是人体规则候选数；三种互斥结果满足 `candidate_count = pose_valid_rep_count + no_rep_count + unsure_count`。`pose_valid_rep_count` 仍只是二维人体序列有效，不是官方完赛计数。 |
| Rowing / SkiErg / Sled Push / Sled Pull | `cycle_count` 是划动、拉动或推动步分析周期；通用计数字段为兼容现有界面保留，必须结合 `count_semantics: analysis_cycle` 与 `official_rep_count_supported: false` 解读。 |
| Farmers Carry | `count_semantics: continuous_monitor`；`cycle_count` 和 `rep_count` 保持 0，应查看 `carrying/rest`、技术反馈和持续违规状态。 |

最终计数前还会执行统一可观测性门控。整次动作平均可见度低于 `0.65`、必需关键点置信度低于 `0.60`、决定性规则置信度低于 `0.72`、局部地板失效、已知拍摄视角不适合或失败只出现一个异常帧时，结论统一降级为 `UNSURE`。逐规则结果仍保留原始 `PASS/FAIL`，降级依据可在 `last_rep_observability` 查看；持续多帧且证据充分的明确失败仍记为 `NO_REP`。

负重箭步蹲已接入五项必需人体规则：后膝触地、触地后的双膝伸展、触地后的双髋伸展、有效触地腿交替，以及无额外调整步。前后腿优先按人体前进方向和脚位置确定；方向不足时才以更接近地面的膝作为降置信度回退。只有 `VALID` 动作会更新上一条触地腿，遮挡或地板不可靠会记为 `UNSURE`。

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
- `yolo11n.pt`：可选人体检测器。

## 网页版快速开始

本机使用可双击 `启动网页.bat`，或在命令行运行：

```powershell
.venv\Scripts\python.exe start_web.py
```

程序会打开 `http://127.0.0.1:5000`。选择“本机摄像头”，点击“开启摄像头”并允许浏览器的视频权限，然后点击“开始实时分析”。

### 逐次动作语音提示

“动作反馈”右上角的“语音开/关”默认开启。系统在一次人体规则候选或分析周期完成时，汇总该段动作中持续出现的问题以及清晰关键阶段的角度偏离，立即播报最多两条改进建议；没有持续性问题时不会播放无意义提示。农夫行走不按次数拆分，会在动作问题持续约 1.2 秒后播报，同一问题至少间隔 8 秒才会再次提示。

语音使用浏览器 Web Speech API 在当前设备本机合成，不申请麦克风权限、不向服务器上传音频，也不会写入录制视频。开关选择会保存在当前浏览器；若浏览器不支持语音合成，页面会显示“语音不可用”。

页面的“高级设置”中，“骨架设置”只选择完整骨架、仅上半身或仅下半身；“显示手指节点”“隐藏面部”和“镜像画面”是与其分开的同级开关。手指节点默认显示，隐藏面部默认关闭；摄像头模式默认开启镜像，示例和上传视频默认关闭镜像。

系统会在用户站直、双脚稳定且全身完整入镜时，用最近约 0.5 秒的脚跟和脚尖位置自动估计局部地板线。摄像头明显倾斜时，可在“高级设置 → 手动地板线”中依次点击脚下地板的两个点。该标定只用于归一化人体部位的离地高度，不识别赛道，也不测量真实厘米距离；足部被遮挡、人体出画或相机移动时，地板参考会标记为 `UNSURE`。

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
- `--backend auto|mediapipe|yolo-pose`：姿态后端；
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

## 代码结构

```text
hyrox/                   # 8 个 HYROX 动作分析器与通用逻辑
configs/hyrox/           # 动作配置
src/backends/            # MediaPipe / YOLO Pose
src/biomechanics/        # 通用运动学数据
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
python -m compileall -q hyrox src webui tools main.py
node --check webui\static\app.js
```

自动化测试覆盖通用接触检测、脚部事件、`VALID/NO_REP/UNSURE` 可观测性门控、三项人体有效计数动作、距离动作违规边界、调试显示和相同特征流重复运行确定性。

## 限制

- 当前结论来自人体关键点，是视觉运动学代理，不是医疗诊断；
- Wall Ball 不检测药球、目标命中或目标高度；
- Sled Push / Pull 不检测器械或真实负载；
- Rowing / SkiErg 不读取器械阻力、功率或距离；
- 距离动作的 `cycle_count` 不是官方有效次数，系统不确认雪橇过线、Rowing/SkiErg 完成里程或训练区间外的规则状态；
- Farmers Carry 不检测壶铃重量、真实携带距离或完成线；
- 拍摄视角、遮挡、光照和多人干扰会影响识别质量；
- 网页版虽已完成会话隔离和容量限制，但尚未完成实体设备兼容验收、50 路压力测试和正式公网部署。
