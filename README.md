# Python 实时人体姿态检测与运动学数据采集

这是一个 Windows 本地 Python 项目，主目标是构建实时人体关键点检测与动作指导底座。当前默认主模型是 MediaPipe Pose，可通过 OpenCV 摄像头画面实时显示单人人体骨架，并可采集结构化运动学代理数据。

当前主线能力：

- 第一阶段：实时摄像头人体骨架检测、33 个关键点、骨架线、FPS、截图、录像。
- 实时主链路：`main.py` 支持 MediaPipe、YOLO Pose、One Euro / EMA 平滑、基础反馈、实时性能指标、`record-raw`、`input-video` 和 metrics CSV。
- 当前模型策略：正式保留 `mediapipe` 和 `yolo-pose` 两个姿态 backend。`--backend auto` 会按动作类型选择推荐 backend；未知动作默认 MediaPipe。YOLO Pose 默认自动使用可用 GPU。
- 实时稳定性：`main.py` 默认启用短暂 pose 丢失保持和手部遮挡跳点保护，降低画面闪断和身体点错跳。
- 第二阶段：关节角、速度、角速度、节段方向、稳定性、峰值时序和会话报告导出。
- 第三阶段：个人参考动作库、动作片段裁剪、时间归一化、DTW 对齐和相对差异报告。
- 第四阶段：徒手深蹲周期识别、重复计数、专项运动学指标、视角限制说明和个人参考深蹲比较。
- 第五阶段：篮球投篮动作分段、出手代理时刻、动力链峰值时序、多次投篮一致性和个人参考投篮比较。

## 功能总览

| 功能 | 主要入口 | 使用的方法 | 主要输出 | 当前定位 |
|---|---|---|---|---|
| 实时人体姿态 | `python main.py` | MediaPipe Pose 33 点或 YOLO Pose 17 点、One Euro/EMA 平滑 | 实时骨架、FPS、置信度、延迟 | 单摄像头正式主流程 |
| 手指关键点 | `--show-hands` 或按 `H` | MediaPipe Hand Landmarker 21 点，显示/保存除腕点外 20 点 | 手指 overlay、session landmarks | 可选功能 |
| HYROX 动作指导 | `--hyrox-action` 或运行中按 `A` | 归一化姿态特征、动作状态机、连续帧防抖、计数冷却、视角反馈过滤 | 阶段、次数、最多两条提示、debug CSV | 8 个动作可用 |
| 正面/侧面标准 | `--camera-view` 或按 `V` | 正面保留对称/横向指标，侧面保留深度/倾角/前后位移指标 | `camera_view`、`view_profile`、视角限制提示 | 单摄像头可用 |
| 动作热切换 | 按 `A` 后选 `0–8`，或按 `N` | 不重启摄像头，只安全替换动作分析器并重置状态 | 新动作阶段和计数 | 可用 |
| 视频录制与复跑 | `--record`、`--record-raw`、`--input-video` | OpenCV 采集/编码，共用实时处理链 | MP4、metrics CSV | 可用 |
| 运动学会话 | 按 `C` | 关键点、关节角、速度、角速度、节段和时序代理 | CSV、JSON、PNG | 可用 |
| 深蹲分析 | `--analysis-mode squat` | 站立校准、状态机、视角相关指标、重复计数 | 实时面板和离线报告 | 可用，非医学标准 |
| 篮球投篮 | `--analysis-mode basketball` 和离线工具 | 动作分段、出手代理、峰值时序、一致性 | CSV、JSON、PNG、Markdown | 可用，不判断命中 |
| 个人参考比较 | `src.tools.create_reference/compare_session` | 身体相对归一化、线性重采样、约束 DTW | 对齐轨迹、误差和报告 | 可用，仅相对比较 |
| 启动诊断 | `python -m src.doctor` | 依赖、模型、配置、目录和可选摄像头检查 | 文本或 JSON、READY/NOT READY | 可用 |
| 多摄像头检查 | `tools/check_multicamera.py` | 多设备同时打开、时间戳和帧偏差统计 | 同步率和 skew | 基础层可用，动作融合未启用 |

## 核心模型与技术方法

### 姿态与检测模型

- **MediaPipe Pose Landmarker Full**：日常主入口使用 Tasks API 的 `VIDEO` 模式逐帧推理，输出 33 个身体关键点；适合需要脚跟、脚尖和较完整身体结构的分析。
- **YOLO11n Pose**：通过 Ultralytics 输出 COCO 17 点，可使用 CPU 或 CUDA；复杂背景和部分快速动作下人体框更稳定，但没有脚跟、脚尖和手指细节。
- **MediaPipe Hand Landmarker**：独立可选手部模型，每只手 21 点；项目避免与 Pose 腕点重复，仅叠加和保存 20 个手指点。
- **YOLO11n person detector**：实验性低频人体框检测器，只在显式启用 person detector/fusion 时加载；不是第三套姿态模型。

### 稳定、特征与动作方法

- **One Euro Filter**：默认平滑方法；慢动作抑制抖动，快速动作提高跟随性。也支持 EMA 或完全关闭平滑。
- **短暂姿态保持与遮挡保护**：检测短时丢失或手部遮挡导致身体点异常跳变时，短暂复用上一稳定结果，并计入 metrics。
- **运动学代理**：使用三点夹角计算肩、肘、髋、膝角；位置、距离和身体位移尽量按画面或人体尺度归一化。
- **HYROX 状态机**：各动作独立阶段规则，使用 `stable_frames` 防抖、动作 cooldown 防重复计数、低可见度独占提示和正面/侧面反馈白名单。
- **深蹲状态机**：先建立站立基线，再按下降、最低位、上升和完成状态计数；正面与侧面输出不同指标集合。
- **投篮分析**：依据膝髋变化、手腕速度、肘伸展和手腕高度代理划分阶段；`release_proxy` 不是篮球真实离手检测。
- **个人参考比较**：身体相对坐标、镜像规范化、线性重采样和 NumPy 约束 DTW；只表示与指定个人参考的差异，不代表专业优劣。
- **数据质量与性能**：记录关键点可见度/缺失率、成功率、FPS、平均与 P95 推理时间、jitter、丢人次数和 ROI 指标。

### 两个实时入口的区别

| 入口 | 定位 | MediaPipe 运行方式 |
|---|---|---|
| `main.py` | 推荐日常入口；模型切换、HYROX、视角、录制和专项面板 | `VIDEO`，同步逐帧调用，易复跑和对比 |
| `python -m src.realtime_pose` | 内部高级入口；暴露检测宽度、提交 FPS 和结果过期控制 | `LIVE_STREAM`，callback 异步接收结果 |

除非需要高级异步参数，优先使用 `main.py`。

## 如何判断程序正常

1. `python -m src.doctor --strict` 最后输出 `READY`：Python、依赖、模型、8 份 HYROX 配置和输出目录正常。
2. `python main.py --camera 0 --camera-view front` 能打开窗口，控制台打印 `Resolved backend`，骨架能跟随人体，状态栏 FPS 持续更新。
3. 人体完整入镜时 `pose_detected`/成功率稳定，HYROX 不持续出现 `LOW_VISIBILITY`；视角栏显示实际的 `front` 或 `side`。
4. 用仓库样例视频回放能得到文档记录的帧数和大致计数，说明模型、特征、状态机和输出链路同时工作。
5. `python -m pytest -q` 应全部通过且没有 failed/error；测试数量会随功能增长。代码测试通过不等于真实动作准确率已经获得临床或竞赛认证。

更详细的逐步验收、运行异常表现和处理方式见 `使用说明.md` 的“基础检查”和“如何判断是否正常运行”。

## 目录结构

```text
.
├── README.md
├── main.py
├── requirements.txt
├── hyrox/
│   ├── actions/
│   │   ├── lunge.py
│   │   ├── wall_ball.py
│   │   ├── farmers_carry.py
│   │   ├── rowing.py
│   │   ├── skierg.py
│   │   ├── burpee_broad_jump.py
│   │   ├── sled_push.py
│   │   └── sled_pull.py
│   ├── registry.py
│   └── features.py
├── models/
│   ├── pose_landmarker_full.task
│   └── hand_landmarker.task
├── outputs/
│   ├── sessions/
│   ├── screenshots/
│   ├── recordings/
│   ├── references/
│   ├── comparisons/
│   ├── squat_sessions/
│   ├── squat_reports/
│   └── basketball/
│       ├── shots/
│       ├── references/
│       └── reports/
├── configs/
│   ├── hyrox/
│   │   ├── README.md
│   │   └── <8 个动作的独立 YAML>
│   ├── reference_features.yaml
│   ├── reference_quality.yaml
│   ├── squat_basic_v1.yaml
│   ├── squat_camera_views.yaml
│   ├── basketball_shot_v1.yaml
│   └── basketball_views.yaml
├── src/
│   ├── pose/
│   ├── backends/
│   ├── detectors/
│   ├── fusion/
│   ├── realtime/
│   ├── realtime_pose.py
│   ├── import_test.py
│   ├── biomechanics/
│   ├── fitness/
│   ├── sports/
│   ├── reference/
│   ├── tools/
│   ├── ui/
│   └── utils/
└── tests/
```

## 安装

需要 Python 3.10 或更高版本。当前开发版本可用 `python main.py --version` 查看。

在 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 已包含 `ultralytics`，用于 `--backend yolo-pose` 和实验性的 YOLO bbox 链路。默认 `mediapipe` 流程不会主动加载 YOLO；如果某个环境没有安装 `ultralytics`，只有显式使用 YOLO Pose 或 `--person-detector yolo` 时才会提示。

如果 PowerShell 禁止激活虚拟环境，可以先在当前 PowerShell 窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 下载模型

程序默认读取：

```text
models/pose_landmarker_full.task
```

下载官方 MediaPipe Pose Landmarker full 模型：

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" `
  -OutFile "models\pose_landmarker_full.task"
```

模型说明可参考 MediaPipe 官方 Pose Landmarker 文档：

- https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python

如需显示五根手指的手指点，还需要官方 MediaPipe Hand Landmarker 模型：

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" `
  -OutFile "models\hand_landmarker.task"
```

运行中按 `H` 后，程序会显示五根手指的完整手指点，不显示手腕点。当前显示和保存的是除 `wrist` 外的 20 个手指点：拇指 `CMC/MCP/IP/TIP`，食指/中指/无名指/小指 `MCP/PIP/DIP/TIP`。也可以启动时加 `--show-hands`，让手指点默认显示。

## 运行

首次安装、移动模型或更换电脑后，先运行启动前诊断：

```powershell
python -m src.doctor
python -m src.doctor --camera 0
python -m src.doctor --json
```

第一条检查 Python、核心依赖、模型、8 份 HYROX 配置和输出目录；第二条额外读取一帧摄像头；
第三条生成适合自动化处理的 JSON。只有显式使用 `--strict` 时，手部和 YOLO 等可选组件缺失才会导致失败。

基础导入测试：

```powershell
python -m src.import_test
```

单元测试：

```powershell
python -m pytest -q
```

pytest 已固定只收集 `tests/`，不会误扫描 `outputs/` 中的历史报告。当前成熟度差距与分阶段实施状态见
`程序成熟度审计与实施清单.md`。

启动当前主实时入口：

```powershell
python main.py
```

`main.py` 是唯一面向日常使用的项目入口。旧的 `app.py` 仅转发到 `src.realtime_pose`，没有独立功能，现已删除；需要更细的异步检测参数时仍可直接使用内部高级模块 `python -m src.realtime_pose`。

默认平滑方式是 One Euro Filter，也可以显式指定：

```powershell
python main.py --backend mediapipe --smoothing one-euro
```

`main.py` 默认使用 `--backend auto`。未知动作走 MediaPipe；如果传入已验证的 HYROX 动作名或视频文件名，会按动作级策略选择 MediaPipe 或 YOLO Pose。

实时窗口打开后，可在普通主链路中按 `B` 在 `mediapipe` 与 `yolo-pose` 之间切换，摄像头不会重新打开。该热切换仅支持 `--fusion none --person-detector none`；切换时会重置关键点平滑器，YOLO Pose 仍按 `--yolo-device auto` 自动使用 GPU。热切换适合实时观察效果，公平 metrics 对比仍建议用固定 backend 分别复跑同一段视频。

`main.py` 默认启用两项实时稳定性保护：短暂丢失 pose 时保留上一帧有效骨架最多 `5` 帧；手腕/手指靠近肩、肘、髋、膝等身体点并触发异常跳变时，短暂保留上一帧稳定位置。画面左上角会显示 `tracking: HOLD` 或 `occlusion_guard`，metrics 会记录 `stabilized_hold_count` 和 `occlusion_guard_count`。可用 `--pose-hold-frames 0` 关闭短暂保留，用 `--no-occlusion-guard` 关闭遮挡保护。

### 正面与侧面评价标准

启动时应明确选择当前机位：

```powershell
python main.py --camera 0 --camera-view front --hyrox-action wall_ball
python main.py --camera 0 --camera-view side --hyrox-action lunge
```

运行中按 `V` 可在 `front` 与 `side` 间切换。切换会重置 HYROX 和深蹲状态，防止一次计数混用两套标准；
会话采集中禁止切换，需先按 `C` 停止并保存会话。`front_left`、`front_right` 在评价策略中归入正面档案，
但原始机位名称仍会写入 CSV 和会话元数据。

不同视角不会输出同一套结论：

| 视角 | 主要评价内容 |
|---|---|
| 正面 | 左右肩髋平衡、膝内扣、双手同步、双脚落地差等横向/对称指标 |
| 侧面 | 动作深度、躯干倾角、髋部折叠、伸展幅度、前后位移与阶段节奏 |

划船、波比跳远、推/拉雪橇等优先侧面动作在正面模式下只输出能够可靠判断的有限指标，并显示
`CAMERA_VIEW_LIMITED`，不会用正面图像冒充完整侧面评价。未选择视角时会显示
`CAMERA_VIEW_REQUIRED`。离线回放使用相同策略：

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\划船机.mp4" --hyrox-action rowing --camera-view side --headless
```

### 多摄像头准备

正式动作分析当前仍以单摄像头为默认。项目已提供可同时打开两个或多个摄像头、验证持续读帧和时间偏差的基础工具：

```powershell
python tools/check_multicamera.py --camera 0:front:mirror --camera 1:side:no-mirror --frames 60
python -m src.doctor --camera 0 --camera 1
```

该工具目前只做设备与同步验收，不做双视角动作融合。正式融合的线程、降级、指标归属和验收路线见
`多摄像头设计与实施路线.md`。

### 摄像头运行中切换指导动作

无需关闭摄像头即可切换 HYROX 指导动作：

- 按 `A` 打开动作选择菜单，再按 `0–8` 选择。
- 按 `N` 不打开菜单，直接切换到下一个动作。
- 菜单打开时按 `A` 或 `ESC` 取消；按 `Q` 仍然退出程序。

| 代号 | 动作 |
|---:|---|
| 0 | 关闭动作指导 |
| 1 | 负重箭步蹲 `lunge` |
| 2 | 投掷药球 `wall_ball` |
| 3 | 农夫行走 `farmers_carry` |
| 4 | 划船机 `rowing` |
| 5 | 滑雪机 `skierg` |
| 6 | 波比跳远 `burpee_broad_jump` |
| 7 | 推雪橇 `sled_push` |
| 8 | 拉雪橇 `sled_pull` |

切换时会创建目标动作自己的分析器并清空上一动作的阶段和次数，当前 `camera_view` 与灵敏度保持不变。
如果启动命令通过 `--hyrox-config` 指定了自定义配置，该文件只用于启动时对应的动作；切换到其他动作时
自动使用各自动作默认配置，避免把弓步阈值误用到划船等动作。

录制原始摄像头画面，不带骨架和文字，用于后续公平测试：

```powershell
python main.py --backend mediapipe --smoothing one-euro --record-raw videos/test_raw.mp4
```

录制带骨架和文字的画面：

```powershell
python main.py --backend mediapipe --smoothing one-euro --record outputs/recordings/test_annotated.mp4
```

读取同一段原始视频并保存指标：

```powershell
python main.py --input-video videos/test_raw.mp4 --backend mediapipe --smoothing one-euro --save-metrics results/mediapipe.csv
```

`main.py` 是实时优先的通用底座入口；`record-raw` 和 `input-video` 主要用于测试和公平评测，不是主流程。

### 模型策略

当前正式姿态模型只保留两个：

| backend | 输出点位 | 默认设备 | 适合场景 | 主要限制 |
|---|---:|---|---|---|
| `mediapipe` | 33 点 | CPU | 默认主模型；大多数动作分析；深蹲、投篮、波比跳远、箭步蹲等需要脚跟、脚尖、手部附加点或更多身体细节的场景 | 复杂背景、多人干扰或快速遮挡时可能短暂丢 pose |
| `yolo-pose` | COCO 17 点 | `--yolo-device auto`，有 CUDA 时使用 GPU | 部分快速动作、复杂背景、人体框更稳定的场景；HYROX 中划船机、拉雪橇、滑雪机当前推荐它 | 点位少于 MediaPipe，没有脚跟、脚尖、手指等细节；不适合直接替代 33 点运动学分析 |

`--backend auto` 是当前推荐的实时入口。选择规则为：

1. 如果传入 `--action-type`，优先按动作类型查策略。
2. 如果传入 `--input-video`，会按视频文件名推断 HYROX 动作。
3. 不能识别动作时默认使用 `mediapipe`。

HYROX 批量对比后的默认策略：

| 动作 | 推荐 backend | 说明 |
|---|---|---|
| 农夫行走 | `mediapipe` | 速度和稳定性综合更合适 |
| 划船机 | `yolo-pose` | 对该视频动作的稳定性更好 |
| 投掷药球 | `mediapipe` | 需要更多身体细节，33 点更合适 |
| 拉雪橇 | `yolo-pose` | 对该视频动作的稳定性更好 |
| 推雪橇 | `mediapipe` | 综合指标更好 |
| 波比跳远 | `mediapipe` | 大幅动作下 33 点输出更适合后续分析 |
| 滑雪机 | `yolo-pose` | 对该视频动作的稳定性更好 |
| 负重箭步蹲 | `mediapipe` | 下肢细节和 33 点输出更合适 |

常用启动方式：

```powershell
python main.py --backend auto --smoothing one-euro
python main.py --backend auto --action-type rowing --smoothing one-euro
python main.py --backend auto --input-video "HYROX视频\划船机.mp4" --smoothing one-euro --save-metrics results/rowing_auto.csv
```

强制指定模型：

```powershell
python main.py --backend mediapipe --smoothing one-euro
python main.py --backend yolo-pose --yolo-device auto --smoothing one-euro
python main.py --backend yolo-pose --yolo-device cpu --smoothing one-euro
```

实时窗口中可按 `B` 在 `mediapipe` 和 `yolo-pose` 之间切换，摄像头保持打开。热切换只用于观察实时效果；如果要做公平对比，仍应固定 backend，用同一段 raw video 分别复跑并保存 metrics。

`yolo-roi-mediapipe` 不是第三个正式姿态模型。它只是实验性的辅助链路：YOLO 只负责低频检测人体 bbox，姿态估计仍由 MediaPipe 输出 33 点。当前 HYROX 对比没有把它列为任何动作的默认推荐；它默认不启用，不参与 `--backend auto`，也不支持运行中按 `B` 热切换。只有在多人、复杂背景或目标锁定需求明显时，才建议单独测试：

```powershell
python main.py --backend mediapipe --person-detector yolo --fusion yolo-roi-mediapipe --smoothing one-euro
```

如果没有安装 `ultralytics`，只有显式使用 `--backend yolo-pose` 或 `--person-detector yolo` 时才会提示安装；默认 `mediapipe` 流程不受影响。

### 统一模型输出 `NormalizedPose`

MediaPipe 33 点和 YOLO Pose COCO 17 点现在都会经过 `src/pose/adapters.py` 转换为同一份 17 点公共格式。现有 `PoseResult` 继续服务于绘制、平滑和热切换兼容层；新模块应只依赖 `src.pose.NormalizedPose`，不要读取 MediaPipe/Ultralytics 原始对象。

统一对象包含：

- `source`：严格为 `mediapipe` 或 `yolopose`。
- `frame_id`、`timestamp_ms`、`latency_ms`：用于区分异步模型结果及观测延迟。
- `image_width`、`image_height`：产生结果的图像尺寸。
- `keypoints`：按统一名称索引的 17 点字典，`x/y` 均为像素坐标；YOLO 没有 z 时为 `None`。
- `bbox`：像素坐标 `(x1, y1, x2, y2)`。
- `overall_confidence`：有效公共关键点置信度平均值。

调试转换结果：

```powershell
python main.py --backend mediapipe --normalized-pose-debug
python main.py --backend yolo-pose --yolo-device auto --normalized-pose-debug
```

调试信息只在第 1 帧及每 30 帧输出一次。两条命令应分别运行；该功能不会为了对齐输出而在同一帧同时启动 CPU MediaPipe 与 GPU YOLO Pose。

新运动项目接入时，不直接凭肉眼观感改默认策略。建议先录制 3 到 5 段代表视频，然后分别用 `mediapipe`、`yolo-pose` 复跑，必要时再单独测 `yolo-roi-mediapipe`。优先看成功率、关键部位缺失率、关键点/角度抖动、P95 推理耗时和端到端延迟，再把推荐 backend 写入 `src/utils/backend_policy.py`。

### `main.py` 参数速查

`main.py` 是当前推荐的实时检测、模型切换、视频复跑和公平对比入口。常用参数如下：

- `--backend`：`auto`、`mediapipe` 或 `yolo-pose`，默认 `auto`。
- `--action-type`：动作类型，用于 `--backend auto`，例如 `rowing`、`ski_erg`、`burpee_broad_jump`；默认 `auto`。
- `--input-video`：读取视频文件复跑，不打开摄像头。
- `--camera`、`--width`、`--height`、`--camera-fps`、`--camera-fourcc`：摄像头选择与采集参数；默认 `0`、`640x480`、`60 FPS`、`MJPG`。
- `--model`：MediaPipe `.task` 模型路径，默认 `models/pose_landmarker_full.task`。
- `--yolo-pose-model`：YOLO Pose 权重，默认 `yolo11n-pose.pt`。
- `--yolo-device`：YOLO Pose 推理设备，默认 `auto`；有 CUDA 时通常解析为 GPU `0`，可手动设为 `cpu`。
- `--landmark-profile`：启动显示 profile，支持 `full`、`no-face`、`upper-body`、`lower-body`、`shot`；`main.py` 默认 `full`。
- `--show-hands`、`--hand-model`、`--hand-detect-width`、`--max-hand-detect-fps`、`--max-hands`：手指点显示与独立手部检测参数。
- `--save-dir`：截图、录屏、session 的输出根目录，默认 `outputs`。
- `--hyrox-debug`：显示 HYROX 调试指标覆盖层。
- `--hyrox-action`：启用 HYROX 实时动作分析，当前支持 `none`、`lunge`、`wall_ball`、`farmers_carry`、`rowing`、`skierg`、`burpee_broad_jump`、`sled_push`、`sled_pull`。
- `--hyrox-sensitivity`：HYROX 动作灵敏度，支持 `low`、`medium`、`high`，默认 `medium`。
- `--hyrox-config`：HYROX 动作配置文件路径；留空时按当前动作自动选择 8 份独立 YAML 之一。显式指定的路径不存在时会报错；默认 YAML 缺失或字段不完整时才使用动作内置安全默认值。
- `--metrics-overlay`：启动时显示通用运动学信息面板。
- `--session-autostart`：启动后自动开始一次 session 采集。
- `--analysis-mode`、`--shot-type`、`--shooting-side`：深蹲 / 篮球专项实时模式参数；`--camera-view` 同时影响 HYROX、深蹲和篮球的视角相关结论。
- `--smoothing`：`one-euro`、`ema` 或 `none`，默认 `one-euro`。
- `--pose-hold-frames`：短暂丢 pose 时保留上一帧有效骨架的帧数，默认 `5`，设为 `0` 可关闭。
- `--occlusion-guard` / `--no-occlusion-guard`：开启或关闭手部遮挡跳点保护，默认开启。
- `--record`：保存带骨架和文字的标注视频。
- `--record-raw`：启动后立即保存原始输入画面；运行中也可按 `T` 开始或停止原始视频录制，用于后续公平复跑。
- `--save-metrics`：把本次运行的性能和稳定性指标追加到 CSV。
- `--headless`：不打开 OpenCV 窗口，适合批量视频评估。
- `--normalized-pose-debug`：第 1 帧及每 30 帧打印统一输出的来源、帧号、时间戳、17 点数量、置信度、延迟和像素 bbox。

运行时快捷键：

```text
A：打开动作选择菜单；菜单中 0–8 选择动作
N：快速循环到下一个 HYROX 动作
V：在 front / side 评价视角间切换
B：在 mediapipe / yolo-pose 之间切换当前姿态 backend
S：保存截图
R：开始或停止视频录制
T：开始或停止原始视频录制
M：切换镜像
1：完整 33 点骨架模式
2：关键关节高亮模式
3：显示或隐藏运动学信息面板
F：显示或隐藏面部点
6：切换到 no-face 模式
7：切换到 upper-body 模式
8：切换到 lower-body 模式
H：显示或隐藏手指点面板
C：开始或停止运动学数据采集会话
K：开始或重新进行深蹲站立校准
P：开始或暂停深蹲专项分析
4：显示或隐藏深蹲专项信息面板
5：显示或隐藏篮球时序面板
J：开始或停止投篮片段候选采集标记
L：手动记录当前帧为出手代理时刻
Q / ESC：退出
```

原有高级实时检测入口仍保留：

```powershell
python -m src.realtime_pose
```

常用参数：

```powershell
python -m src.realtime_pose --camera 0 --width 1280 --height 720
python -m src.realtime_pose --no-mirror
python -m src.realtime_pose --record
python -m src.realtime_pose --metrics-overlay
python -m src.realtime_pose --session-autostart
python -m src.realtime_pose --camera-fps 60 --max-detect-fps 30
python -m src.realtime_pose --detect-width 960 --max-detect-fps 30
python -m src.realtime_pose --detect-width 480 --smoothing 0.75
python -m src.realtime_pose --save-dir outputs
python -m src.realtime_pose --smoothing 0.65
python -m src.realtime_pose --model models\pose_landmarker_full.task
python -m src.realtime_pose --landmark-profile full
python main.py --analysis-mode squat --camera-view side
python -m src.realtime_pose --analysis-mode squat --camera-view front --metrics-overlay
python main.py --analysis-mode basketball --shot-type set_shot --shooting-side right --camera-view side
```

检查已保存会话：

```powershell
python -m src.tools.inspect_session --session outputs\sessions\<session_id>
```

从会话片段创建个人参考动作：

```powershell
python -m src.tools.create_reference `
  --session outputs\sessions\<session_id> `
  --start-ms 2400 `
  --end-ms 4100 `
  --name "我的深蹲参考动作 A" `
  --action-type squat `
  --camera-view side `
  --movement-side bilateral
```

列出参考动作并比较新片段：

```powershell
python -m src.tools.list_references

python -m src.tools.compare_session `
  --session outputs\sessions\<candidate_session_id> `
  --reference outputs\references\<reference_id> `
  --start-ms 1800 `
  --end-ms 3900 `
  --output-dir outputs\comparisons
```

离线分析深蹲会话：

```powershell
python -m src.tools.calibrate_squat `
  --session outputs\sessions\<session_id> `
  --camera-view side

python -m src.tools.analyze_squat_session `
  --session outputs\sessions\<session_id> `
  --camera-view side `
  --output-dir outputs\squat_reports

python -m src.tools.analyze_squat_session `
  --session outputs\sessions\<session_id> `
  --camera-view side `
  --reference outputs\references\<squat_reference_id>
```

离线分析篮球投篮：

```powershell
python -m src.tools.inspect_shot `
  --session outputs\sessions\<session_id> `
  --shooting-side right

python -m src.tools.analyze_shot_session `
  --session outputs\sessions\<session_id> `
  --start-ms 1200 `
  --end-ms 3100 `
  --shot-type set_shot `
  --shooting-side right `
  --camera-view side

python -m src.tools.create_shot_reference `
  --session outputs\sessions\<session_id> `
  --start-ms 1200 `
  --end-ms 3100 `
  --name "我的定点投篮参考" `
  --shot-type set_shot `
  --shooting-side right `
  --camera-view side

python -m src.tools.compare_shots `
  --shots outputs\basketball\reports\<shot_a> outputs\basketball\reports\<shot_b> `
  --shooting-side right `
  --shot-type set_shot
```

从多个已确认参考片段创建聚合模板：

```powershell
python -m src.tools.create_reference `
  --from-clips outputs\references\<clip_a> outputs\references\<clip_b> outputs\references\<clip_c> `
  --name "我的稳定深蹲模板" `
  --action-type squat
```

下面这组参数主要对应 `python -m src.realtime_pose` 高级入口；`main.py` 请以上面的“参数速查”为准。

- `--camera`：摄像头编号，默认 `0`。
- `--mirror` / `--no-mirror`：开启或关闭镜像显示，默认开启。
- `--width` / `--height`：请求摄像头分辨率，默认 `640x480`，优先保证实时 FPS。
- `--camera-fps`：请求摄像头采集 FPS，默认 `60`；`0` 表示使用摄像头后端默认值。
- `--camera-fourcc`：请求摄像头 FourCC 编码，默认 `MJPG`；空字符串表示不修改摄像头编码。
- `--record`：启动后立即录制视频。
- `--smoothing`：关键点指数平滑系数，范围 `0` 到 `1`，`0` 表示关闭，默认 `0.65`。
- `--model`：`.task` 模型文件路径，默认 `models/pose_landmarker_full.task`。
- `--landmark-profile`：启动时显示和保存的姿态点集合；`src.realtime_pose` 默认 `no-face`，`main.py` 默认 `full`，可选 `full`、`no-face`、`upper-body`、`lower-body`、`shot`。
- `--show-hands`：启动时直接显示五根手指点；不加该参数时也可按 `H` 开启或关闭。
- `--hand-model`：手部 `.task` 模型文件路径，默认 `models/hand_landmarker.task`。
- `--hand-detect-width`：手部检测输入宽度，默认 `416`；较小可降低延迟，`0` 表示使用完整画面。
- `--max-hand-detect-fps`：手部检测提交上限，默认 `18`。
- `--max-hands`：最多检测手的数量，默认 `2`。
- `--save-dir`：输出根目录，默认 `outputs`。
- `--metrics-overlay`：启动时显示运动学信息面板，默认关闭。
- `--session-autostart`：启动后自动开始运动学数据采集会话，默认关闭。
- `--analysis-mode`：实时专项分析模式，`pose`、`squat` 或 `basketball`，默认 `pose`。
- `--camera-view`：专项分析视角，`side`、`front`、`front_left`、`front_right` 或 `unknown`。
- `--shot-type`：篮球投篮类型，`set_shot` 或 `jump_shot`，默认 `set_shot`。
- `--shooting-side`：篮球投篮侧，`right` 或 `left`，默认 `right`。
- `--detect-width`：检测输入宽度，默认 `480`；较小可降低延迟，`0` 表示使用完整画面。
- `--max-detect-fps`：MediaPipe 检测提交上限，默认 `30`，用于避免异步检测队列堆积。
- `--max-pending-ms`：单次异步检测等待超时，默认 `180` 毫秒。
- `--max-result-lag-ms`：过旧姿态结果隐藏阈值，默认 `280` 毫秒。
- `--plot-on-save` / `--no-plot-on-save`：会话保存时是否生成 PNG 曲线图，默认开启。

## 快速动作稳定性建议

程序已默认启用以下优化：

- 摄像头默认请求 `640x480`、`MJPG` / `60 FPS` 采集模式，缓冲请求设为 `1`，减少读取旧画面并提高动作节点刷新率。
- 检测输入默认缩放到 `480` 宽，降低 CPU 推理延迟，同时仍使用原来的 `pose_landmarker_full.task` 模型。
- `src.realtime_pose` 默认使用 `no-face` profile，`main.py` 默认使用 `full` profile；如果要减少面部点绘制和保存，可显式设置 `--landmark-profile no-face`，或运行中按 `F` / `6`。
- MediaPipe 异步检测最多保留一个待处理任务，避免检测队列积压。
- 过旧检测结果不会继续叠加到当前画面。
- 关键点平滑会根据运动速度自适应：慢速时抑制抖动，快速时提高跟随速度。
- 低置信度关键点发生大幅跳变时会被抑制。
- 手或手腕靠近肩、肘、髋、膝等身体点并触发异常大跳时，会短暂保留上一帧稳定位置，降低遮挡导致的错点跳变。

如果快速动作仍然延迟明显，可以优先尝试：

```powershell
python -m src.realtime_pose --width 640 --height 480 --camera-fps 60 --detect-width 480 --max-detect-fps 30 --smoothing 0.75
```

如果画面很流畅但关节点偶尔不准，可以提高检测输入宽度：

```powershell
python -m src.realtime_pose --detect-width 1280 --smoothing 0.65
```

## 快捷键

- `A`：打开 HYROX 动作选择菜单；菜单中按 `0–8` 选择动作。
- `N`：快速切换到下一个 HYROX 动作。
- `V`：在正面和侧面评价档案间切换；切换会重置动作状态。
- `B`：在 `main.py` 中切换当前姿态 backend。
- `Q` 或 `ESC`：退出。
- `S`：保存截图到 `outputs/screenshots/`。
- `R`：开始或停止视频录制，文件保存到 `outputs/recordings/`。
- `T`：开始或停止原始视频录制，文件保存到 `outputs/recordings/`，默认文件名带 `_raw.mp4`。
- `M`：切换镜像显示。
- `1`：完整 33 点骨架模式。
- `2`：投篮关键关节高亮模式，突出肩、肘、腕、髋、膝、踝。
- `3`：显示或隐藏运动学信息面板。
- `F`：显示或隐藏面部点。
- `6`：切换到 `no-face` 模式。
- `7`：切换到 `upper-body` 模式。
- `8`：切换到 `lower-body` 模式。
- `H`：显示或隐藏手指点。
- `C`：开始或停止一次运动学数据采集会话。
- `K`：在深蹲模式下开始或重新进行站立校准。
- `P`：在深蹲模式下开始或暂停专项分析。
- `4`：在深蹲模式下显示或隐藏专项信息面板。
- `5`：在篮球模式下显示或隐藏篮球时序面板。
- `J`：在篮球模式下开始或停止投篮片段候选采集标记。
- `L`：在篮球模式下手动记录当前帧为出手代理时刻。

## 第二阶段功能

- 实时姿态骨架检测。
- 关键点会话采集。
- 关节角与角速度计算。
- 关键点与节段速度计算。
- 身体相对坐标归一化。
- 基础稳定性与左右对称性代理指标。
- 峰值与时序分析基础。
- CSV、JSON、PNG 报告导出。

## 第三阶段功能

第三阶段实现的是“个人参考动作对比”，不是专业动作标准判定。

- 从已保存的 `outputs/sessions/<session_id>/` 中检查数据质量和可用字段。
- 按时间戳或帧号裁剪动作片段，不修改原始 session 文件。
- 创建用户主动命名和确认的个人参考动作库，保存到 `outputs/references/`。
- 使用 `configs/reference_features.yaml` 定义参与比较的关节角、速度、角速度和稳定性代理特征。
- 使用 `configs/reference_quality.yaml` 判断数据是否适合比较；质量不足时输出 `WARNING`，不伪造可靠结论。
- 支持身体相对坐标语义、左右特征镜像规范化、线性时间归一化和 NumPy 实现的约束 DTW。
- 输出动作整体差异、差异最大的特征、差异最大的归一化时间区间和可追溯 CSV。
- 支持多个用户确认参考片段聚合为均值轨迹、标准差轨迹和置信带模板。

参考动作由用户选择和确认。系统比较的是动作轨迹、角度、速度与时序的相对差异。

DTW 距离小不代表动作一定更好；DTW 距离大也不代表动作一定错误。它只表示该动作与指定参考动作的运动学差异程度。

## 第四阶段功能

第四阶段实现的是徒手深蹲专项视觉运动学分析，不是医疗诊断，也不是绝对动作标准判定。

- 支持单人、单摄像头、连续或单次徒手深蹲。
- 支持 `side`、`front`、`front_left`、`front_right` 和 `unknown` 视角。
- 实时模式支持站立校准、状态机计数、专项面板和数据质量提示。
- 离线模式可从 `outputs/sessions/<session_id>/` 识别深蹲重复，导出 CSV、JSON、PNG 和 Markdown 报告。
- 每次深蹲输出开始、最低点、结束时间，下降/起身时长，膝/髋角范围，躯干倾斜范围，骨盆相对下降幅度，左右对称性代理和数据质量等级。
- 侧面视角输出深度、髋膝活动范围、躯干倾斜和动作节奏等代理指标。
- 正面视角输出左右差异、骨盆横向偏移、躯干侧倾等代理指标。
- 可复用第三阶段个人参考动作库，对个人参考深蹲进行相对轨迹比较。

深蹲状态机使用配置文件 `configs/squat_basic_v1.yaml` 中的训练规则 / 视觉代理规则进行分段稳定化。这些阈值只用于重复计数和动作切片，不是医学或普适标准。

## 第五阶段功能

第五阶段实现的是篮球投篮专项视觉运动学分析，不判断投篮姿势是否标准，也不预测命中率。

- 第一版支持单人、固定机位、单次定点投篮或原地跳投。
- 用户必须指定 `shot_type`、`shooting_side` 和 `camera_view`。
- 支持手动裁剪投篮片段；自动候选只作为建议，不能直接当作确定投篮。
- 可识别 `SETUP`、`DIP`、`RISE`、`ARM_EXTENSION`、`RELEASE_PROXY`、`FOLLOW_THROUGH`、`RECOVERY` 等主要阶段，阶段不确定时会在报告中保留提示。
- 可提取投篮侧膝/髋/肩/肘/腕、骨盆、躯干等运动学代理特征。
- 可输出膝、髋、骨盆、肩、肘、腕的峰值事件和事件先后顺序。
- `release_proxy_time` 是基于人体关键点的出手代理时刻，不是篮球真实离手时刻。
- 可比较多次投篮的一致性；该一致性只表示重复动作之间的运动学相似程度。
- 可与个人参考投篮动作进行 DTW 对齐和相对差异分析。
- 预留 `BallTracker` 接口，但本阶段不实现篮球检测模型，不伪装球轨迹检测。

篮球分析规则来自 `configs/basketball_shot_v1.yaml`。其中事件序列模板是分析参考，不是唯一正确动作模式。

## 运动学信息面板

按 `3`，或启动时加 `--metrics-overlay` 后，右上角会显示：

- `POSE: YES / NO`
- `FPS`
- `SESSION: RECORDING / IDLE`
- 镜像状态
- 当前右肘角
- 当前右膝角
- 当前右腕速度代理值
- 当前骨盆速度代理值
- 当前全身运动能量代理值

不可用指标显示为 `N/A`。这些指标只表示视觉姿态数据派生出的运动学代理量。

如果使用 `--analysis-mode squat`，右侧下方还会出现深蹲专项面板，显示左右膝角、躯干前倾代理角、骨盆位移代理、当前状态和次数。

如果使用 `--analysis-mode basketball`，右侧下方还会出现篮球专项面板，显示投篮侧膝角、投篮侧肘角、骨盆速度、投篮侧手腕速度和出手代理时刻。

如果启动时加 `--hyrox-debug`，画面上方还会额外显示一组 HYROX 调试值：`visible`、`lknee`、`rknee`、`lhip`、`rhip`、`torso`。

如果启动时选择任一已注册的 `--hyrox-action`，画面上方会显示对应的 HYROX 分析面板：`action`、`phase`、`reps` 和最多 2 条中文纠正提示。当前 `phase` 使用连续帧确认后的稳定阶段，不会因为单帧抖动立即切换。

## HYROX 通用姿态特征

`hyrox/features.py` 的 `extract_basic_pose_features(...)` 为现有动作和后续动作分析器提供同一份无状态特征。第一轮扩展后包含：

- 关节角度：左右肘、肩、髋、膝角，以及已有的最小/最大聚合角度。
- 身体姿态：躯干相对竖直方向的倾角、归一化肩/髋高度差、身体中心和身体高度。
- 手脚位置：左右手腕/脚踝高度、手腕相对同侧髋和肩的垂直差、双腕及双踝归一化距离。
- 可见度：全身核心点、上肢、下肢、左侧和右侧的独立综合分数。

坐标和距离以画面宽高归一化；屏幕 `y` 轴向下，因此 `*_wrist_to_*_y` 为正表示手腕低于参照点，为负表示手腕高于参照点。`shoulder_tilt`、`hip_tilt` 为“右侧 y 减左侧 y”再除以画面高度，正值表示画面中的右侧点更低。无法计算的静态字段返回 `None`；可见度分数无有效点时保持兼容并返回 `0.0`。

本轮没有加入速度字段。当前函数没有上一帧和时间间隔上下文，保持无状态可避免不同视频、摄像头或会话之间串帧；需要速度特征时应由明确归属某个会话的缓存层计算。

第一轮回归验收：

```powershell
python -m pytest tests/test_hyrox_foundation.py tests/test_wall_ball.py -q
```

## HYROX Lunge 实时模式

HYROX 实时动作分析支持 8 个动作；本节说明弓步 `lunge` 的专项规则：

```powershell
python main.py --hyrox-action lunge --camera-view side
python main.py --hyrox-action lunge --hyrox-sensitivity high
python main.py --hyrox-action lunge --hyrox-debug
python main.py --hyrox-action lunge --hyrox-config configs/hyrox/lunge.yaml
```

- `--hyrox-action lunge`：启用弓步动作分析。
- `--hyrox-sensitivity low`：更保守，阶段确认更慢，误报更少。
- `--hyrox-sensitivity medium`：默认档，连续 `3` 帧确认阶段。
- `--hyrox-sensitivity high`：更敏感，适合动作幅度较小的测试。
- `--hyrox-config configs/hyrox/lunge.yaml`：从配置文件读取弓步阈值，方便通过视频回放调规则而不改代码。

当前弓步分析会输出：

- `stand / descent / bottom / ascent / unknown` 稳定阶段。
- `bottom -> stand` 的重复计数。
- `LOW_VISIBILITY`、`NOT_DEEP_ENOUGH`、`LEAN_TOO_MUCH`、`STAND_EXTENSION` 4 类中文提示。
- 动作面板会显示当前配置名，便于确认这次运行实际用了哪套阈值。
- `bottom` 在已有站立基线时会同时检查膝角和髋部下降量，避免只弯膝但身体没有下降时误判到底部。

### Lunge 当前评判标准（默认 `medium`）

以下标准是程序当前实际使用的视觉判定规则，阈值来自 `configs/hyrox/lunge.yaml`：

| 判定项 | 当前规则 |
| --- | --- |
| 有效画面 | 核心关键点综合可见度不低于 `0.45`；否则进入 `low_visibility / no_pose`，暂停动作判断并清除本次未完成的计数过程 |
| 站立位 | 左右膝角中的较小值不低于 `150°` |
| 最低位 | 左右膝角中的较小值不高于 `115°`；建立站立基线后，髋中心还须相对基线下降至少 `0.035` 个归一化画面高度 |
| 完成 1 次 | 必须稳定经历 `bottom -> stand`；默认每个候选阶段连续 `3` 帧才确认，且两次计数至少间隔 `400 ms` |
| 深度提示 | 最低位膝角仍大于 `100°` 时提示 `NOT_DEEP_ENOUGH`；该提示不阻止本次计数 |
| 躯干提示 | 躯干相对竖直方向前后倾绝对值大于 `20°` 时提示 `LEAN_TOO_MUCH` |
| 站起伸展提示 | 完成计数时，最小膝角小于 `165°` 或最小髋角小于 `160°`，提示 `STAND_EXTENSION`；该提示不撤销已完成的计数 |

因此，当前程序的“计数通过”和“动作无提示”是两个层级：达到最低位并回到站立位即可计数，但深度、躯干或完全伸展仍可能产生纠正提示。

稳定性规则：

- 同一候选阶段需要连续多帧才会确认，默认 `medium=3` 帧。
- 完成一次 rep 后有 `400ms` 冷却时间，避免抖动重复计数。
- `LOW_VISIBILITY` 出现时会独占提示，并暂停其他动作判断。
- 灵敏度会在 YAML 阈值基础上生效：`low` 更保守，`high` 放宽动作幅度并减少确认帧数。

## HYROX Farmers Carry 姿态监控

Farmers Carry 不以重复次数为主要结果，而是监控行走时的负重姿态稳定性。推荐正面或斜前方拍摄：

```powershell
python main.py --hyrox-action farmers_carry --camera-view front
python main.py --hyrox-action farmers_carry --camera-view front --hyrox-debug
python tools/replay_hyrox_video.py --video "HYROX视频\农夫行走.mp4" --hyrox-action farmers_carry --camera-view front --headless
```

阶段为 `ready / carrying / rest / unknown`，`rep_count` 固定为 0。当前方法使用身体中心短时水平变化、
踝间距变化和膝角变化近似判断是否在行走；手腕相对髋部位置判断手臂是否下垂。默认连续 3 帧确认阶段，
静止超过 `1200 ms` 转为 rest。

| 评价项 | 默认规则或提示 |
|---|---|
| 可见度 | `visible_score < 0.55` 时只提示 `LOW_VISIBILITY` |
| 左右倾斜 | 肩或髋归一化高度差超过 `0.08`，提示 `LEAN_LEFT_RIGHT` |
| 肩部平衡 | 肩高度差超过 `0.08`，提示 `SHOULDERS_UNEVEN` |
| 手臂位置 | 双腕未保持在髋部附近下方，提示 `ARMS_NOT_DOWN` |
| 侧面躯干 | 躯干倾斜绝对值超过 `25°`，提示 `TORSO_LEAN` |
| 搬运稳定 | 垂直晃动、躯干角或肩髋倾斜短时变化过大，提示 `UNSTABLE_CARRY` |

正面档案主要保留左右平衡和手臂位置；侧面档案主要保留躯干倾角、手臂位置和行走稳定提示。
程序不检测哑铃/壶铃，不判断重量、200 米距离或比赛是否完成。

## HYROX Wall Ball 人体动作 MVP

Wall Ball 第一版只分析人体的深蹲、站起和投掷伸展姿态，不检测药球、目标命中或目标高度：

```powershell
python main.py --hyrox-action wall_ball --camera-view front
python main.py --hyrox-action wall_ball --hyrox-debug
python main.py --hyrox-action wall_ball --hyrox-sensitivity high
python main.py --hyrox-action wall_ball --hyrox-config configs/hyrox/wall_ball.yaml
```

阶段为 `stand / squat_down / bottom / throw_extension / reset`。一次计数要求先稳定站立、进入 `bottom`，满足配置中的深度条件，再回到站立或投掷伸展。反馈包括：

- `SQUAT_NOT_DEEP`：下蹲深度不够。
- `KNEES_CAVE_IN`：基于正面 2D 膝/踝宽度的低置信度内扣警告，不作为医学或三维动作结论。
- `NOT_FULL_EXTENSION`：站起时髋膝未充分伸展。
- `LOW_VISIBILITY`：关键点不足或全身未入镜；出现时独占反馈。

`configs/hyrox/wall_ball.yaml` 可调整站立、最低位、手腕高于肩、肘部伸展、防抖和冷却阈值。由于药球可能遮挡一侧手腕或髋部，投掷阶段允许使用可见度更好的一侧上肢/髋部，双膝伸展仍需同时满足。

### Wall Ball 当前评判标准（默认 `medium`）

以下标准是程序当前实际使用的视觉判定规则，阈值来自 `configs/hyrox/wall_ball.yaml`：

| 判定项 | 当前规则 |
| --- | --- |
| 有效画面 | 核心关键点综合可见度不低于 `0.45`；否则进入 `low_visibility / no_pose`，暂停判断并清除本次未完成的计数过程 |
| 起始站立位 | 最小膝角不低于 `150°`，且可见度较好一侧的髋角不低于 `145°` |
| 最低位候选 | 最小膝角不高于 `110°` |
| 有效深度 | `hip_knee_depth >= -0.05`，即髋中心视觉高度达到配置允许的膝部高度范围；仅到达最低位候选但未满足此条件不计数 |
| 投掷伸展位 | 最小膝角不低于 `150°`、较好一侧髋角不低于 `145°`、较好一侧肘角不低于 `125°`，且手腕高于肩至少 `0.03` 个归一化画面高度 |
| 完成 1 次 | 必须先稳定站立，再进入满足有效深度的 `bottom`，随后稳定回到 `stand` 或 `throw_extension`；默认连续 `3` 帧确认阶段，计数冷却为 `400 ms` |
| 深度提示 | 处于最低位但未达到有效深度时提示 `SQUAT_NOT_DEEP`，且本次尝试不计数 |
| 膝内扣提示 | 下蹲或最低位时，踝宽至少为 `0.08`，且 `膝宽 / 踝宽 < 0.72`，提示低置信度 `KNEES_CAVE_IN`；不阻止计数 |
| 完全伸展提示 | 本次尝试结束时最小膝角小于 `165°`，或较好一侧髋角小于 `160°`，提示 `NOT_FULL_EXTENSION`；该提示不撤销已完成的计数 |

当前版本只依据人体关键点评判动作过程；不检测药球，因此不能确认球是否触墙、是否达到官方目标高度、球重是否正确或整场是否完成规定次数。

当前仓库样例的无界面验收命令如下；使用默认 `medium + one-euro` 时应处理 `145` 帧、最终阶段为 `stand`、计数为 `3`：

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\投掷药球.mp4" --hyrox-action wall_ball --camera-view front --headless
```

## HYROX Rowing 动作 MVP

Rowing 第一版建议从侧面拍摄，只依据人体姿态近似分析划船节奏：

```powershell
python main.py --hyrox-action rowing --camera-view side
python main.py --hyrox-action rowing --hyrox-debug
python main.py --hyrox-action rowing --hyrox-config configs/hyrox/rowing.yaml
```

阶段为 `catch / drive / finish / recovery / unknown`。只有按顺序稳定完成 `catch → drive → finish → recovery → catch` 才计为一个 stroke，`rep_count` 即 stroke 数；默认连续 `3` 帧确认阶段，两次计数至少间隔 `500 ms`。

默认阈值和提示：

| 判定项 | 当前规则 |
| --- | --- |
| 有效画面 | 综合可见度至少 `0.55`，否则进入 `unknown` 并仅提示 `LOW_VISIBILITY` |
| Catch | 最小膝角不高于 `105°` |
| Drive | 膝角逐帧增大，表示腿部开始伸展 |
| Finish | 最小膝角至少 `145°`，且平均肘角不高于约 `145°`，表示拉柄靠近躯干 |
| Recovery | 膝角开始减小或结束阶段后手臂重新前伸 |
| 后仰提示 | Finish 时躯干倾斜绝对值超过 `45°`，提示 `TOO_MUCH_BACK_LEAN` |
| 早拉手臂 | Drive 尚未充分伸腿而平均肘角小于 `120°`，提示 `EARLY_ARM_PULL` |
| 恢复过快 | Recovery 稳定阶段不足 `120 ms` 就回到 Catch，提示 `RUSHED_RECOVERY` |

其他反馈包括 `NO_FULL_LEG_DRIVE` 和 `NOT_SEATED_OR_BAD_VIEW`。侧面视角之外仍可运行，但二维关键点更容易产生角度歧义。当前版本不读取划船机屏幕，不判断 1000 米、阻力或脚踏板设置。

仓库样例回放验收基线为处理 `512` 帧、最终阶段 `catch`、`reps: 5`：

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\划船机.mp4" --hyrox-action rowing --camera-view side --headless --save-debug-csv outputs\hyrox_rowing.csv
```

## HYROX SkiErg 动作 MVP

SkiErg 第一版优先支持正面或斜前方视角，只分析人体上举、下拉、底部折叠和回程姿态：

```powershell
python main.py --hyrox-action skierg --camera-view front
python main.py --hyrox-action skierg --hyrox-debug
python main.py --hyrox-action skierg --hyrox-config configs/hyrox/skierg.yaml
```

阶段为 `top / pull_down / bottom / return / unknown`。只有稳定完成 `top → pull_down → bottom → return → top` 才计一个 pull，`rep_count` 即 pull 数；默认连续 `3` 帧确认阶段，计数冷却为 `350 ms`。

| 判定项 | 当前规则 |
| --- | --- |
| 有效画面 | 上半身综合可见度至少 `0.50`，否则进入 `unknown` 并提示 `LOW_VISIBILITY` |
| Top | 双腕均高于同侧肩至少 `0.03` 个画面高度，躯干接近直立 |
| Pull Down | 手腕逐帧向下移动，躯干开始折叠 |
| Bottom | 平均手腕低于胸部中心至少 `0.05`，并出现躯干折叠或适度屈膝 |
| Return | 手腕从底部重新上升，躯干恢复直立 |
| 髋折叠提示 | 手腕已经到达低位但躯干倾角不足 `15°`，提示 `NO_HIP_HINGE` |
| 下蹲提示 | Bottom 时平均膝角不高于 `110°`，提示 `TOO_MUCH_SQUAT` |
| 不对称提示 | 左右手腕归一化高度差超过 `0.08`，提示 `ASYMMETRIC_PULL` |
| 回程节奏 | Return 持续不足 `100 ms` 就回到 Top，提示 `RUSHED_RETURN` |

若回程未充分上举就再次下拉，会提示 `ARMS_NOT_HIGH_ENOUGH`。当前版本不检测 SkiErg 机器、平台站位、1000 米、阻力或完成手势。

仓库样例验收基线为处理 `103` 帧、最终阶段 `return`、`reps: 1`：

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\滑雪机.mp4" --hyrox-action skierg --camera-view front --headless --save-debug-csv outputs\hyrox_skierg.csv
```

## HYROX Burpee Broad Jump 动作 MVP

第一版建议使用侧面约 45° 视角，确保全身、双手腕和双脚落地位置持续入镜：

```powershell
python main.py --hyrox-action burpee_broad_jump --camera-view side
python main.py --hyrox-action burpee_broad_jump --hyrox-debug
python main.py --hyrox-action burpee_broad_jump --hyrox-config configs/hyrox/burpee_broad_jump.yaml
```

阶段包括 `stand / hands_down / chest_down / step_or_jump_in / broad_jump_takeoff / flight_or_move / landing / reset / unknown`。稳定经历 `chest_down → broad_jump_takeoff → landing` 计为一次，`rep_count` 不代表官方 80 米是否完成。

| 判定项 | 当前规则 |
| --- | --- |
| 有效画面 | 全身综合可见度至少 `0.55`，否则进入 `unknown` 并提示 `LOW_VISIBILITY` |
| Chest Down | 身体垂直高度不超过 `0.35`，且躯干相对竖直方向倾角绝对值至少 `60°` |
| 起跳 | 从收腿阶段出现膝髋伸展，或身体快速拉高且躯干接近直立 |
| 前移 | 起跳后身体中心出现水平或腾起位移 |
| Broad Jump | 起跳到落地的身体中心水平位移至少 `0.08`；不足时提示 `NO_BROAD_JUMP` |
| 双脚错位 | 双踝视觉高度差超过 `0.08` 时给出低置信度 `FEET_STAGGERED` |
| 落地碎步 | 落地后 `500 ms` 内检测到多次双踝快速位移，提示 `EXTRA_STEPS` |

其他提示包括 `CHEST_NOT_LOW` 和 `HIPS_TOO_HIGH_IN_BOTTOM`。胸部触地、双脚精确误差、手脚精确距离和官方 80 米都无法由单摄像头可靠确认；当前实现不做地面检测，也不引入目标检测模型。

仓库样例验收基线为处理 `501` 帧、最终阶段 `unknown`、`reps: 2`：

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\波比跳远.mp4" --hyrox-action burpee_broad_jump --camera-view side --headless --save-debug-csv outputs\hyrox_burpee_broad_jump.csv
```

## HYROX Sled Push 动作 MVP

第一版建议使用侧面或斜侧面视角，确保肩、髋、膝和脚踝入镜：

```powershell
python main.py --hyrox-action sled_push --camera-view side
python main.py --hyrox-action sled_push --hyrox-debug
python main.py --hyrox-action sled_push --hyrox-config configs/hyrox/sled_push.yaml
```

阶段为 `setup / drive / step / reset / unknown`。Sled Push 不按官方次数计数；`rep_count` 和 debug 中的 `step_count` 仅表示视觉检测到的明显步伐。

| 判定项 | 当前规则 |
| --- | --- |
| 有效画面 | 全身综合可见度至少 `0.55`，否则进入 `unknown` 并提示 `LOW_VISIBILITY` |
| Drive 姿态 | 躯干前倾至少 `25°`；推荐区间为 `25°–65°` |
| 躯干过直 | 前倾小于 `20°` 时提示 `TORSO_TOO_UPRIGHT` |
| 躯干过低 | 前倾超过 `70°` 时提示 `TORSO_TOO_LOW`，但仍可保持驱动阶段 |
| 明显步伐 | 双踝高度或双踝距离的帧间变化达到 `0.04`，且两次计数至少间隔 `250 ms` |
| 腿部驱动 | 一个驱动过程的平均膝角伸展不足 `20°` 时提示 `NO_LEG_DRIVE` |

其他提示包括 `SHORT_STEPS` 和低置信度的 `HIP_TOO_HIGH_OR_BACK_ROUND`。当前版本不检测雪橇，不判断过线、lane、50 米、分段距离或雪橇重量。

仓库样例验收基线为处理 `247` 帧、最终阶段 `drive`、检测步数 `3`：

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\推雪橇.mp4" --hyrox-action sled_push --camera-view side --headless --save-debug-csv outputs\hyrox_sled_push.csv
```

## HYROX Sled Pull 动作 MVP

第一版建议使用侧面或斜侧面视角，确保手臂、躯干、髋、膝和脚入镜：

```powershell
python main.py --hyrox-action sled_pull --camera-view side
python main.py --hyrox-action sled_pull --hyrox-debug
python main.py --hyrox-action sled_pull --hyrox-config configs/hyrox/sled_pull.yaml
```

阶段为 `ready / reach / pull / recover / unknown`。稳定完成 `reach → pull → recover` 计为一次视觉拉动，`rep_count` 即 `pull_count`，不代表官方 50 米是否完成。

| 判定项 | 当前规则 |
| --- | --- |
| 有效画面 | 综合可见度至少 `0.55`，否则进入 `unknown` 并提示 `LOW_VISIBILITY` |
| Reach | 双肘接近伸展，平均肘角达到约 `145°` |
| Pull | 从前伸参考位开始屈肘，拉动幅度至少 `25°` 才属于清晰拉动 |
| Recover | 拉动后肘角重新增大，手臂再次前伸 |
| 非站姿 | 平均膝角不高于 `95°`，或身体中心 y 达到 `0.75`，提示 `NOT_STANDING` |
| 后仰提示 | 躯干倾斜绝对值超过 `35°`，提示 `OVER_LEAN_BACK` |
| 髋腿参与 | 清晰拉动期间髋膝角变化不足 `8°`，提示 `ARMS_ONLY_PULL` |
| 左右同步 | 双腕归一化高度差超过 `0.08`，提示 `ASYMMETRIC_PULL` |

拉动幅度未达到阈值时提示 `NO_CLEAR_PULL` 且不增加计数。当前版本不检测绳子或雪橇，不判断绳子是否出 lane、雪橇过线、50 米、分段距离或重量。

仓库样例验收基线为处理 `332` 帧、最终阶段 `recover`、`reps: 2`：

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\拉雪橇.mp4" --hyrox-action sled_pull --camera-view side --headless --save-debug-csv outputs\hyrox_sled_pull.csv
```

## 文档同步维护约定

以后新增或修改动作、阶段、计数条件、反馈规则、默认阈值、命令行参数或运行流程时，必须在同一次变更中同步更新 `README.md` 和 `使用说明.md`。涉及评判逻辑时，应同时写明计数条件、提示条件、默认配置来源、已知限制及可复现的使用命令，避免代码与文档不一致。

## HYROX 视频回放测试工具

本地视频回放工具支持全部 8 个 HYROX 动作，并通过 `create_action_analyzer` 注册入口复用实时分析器。它只是离线调试工具，不替代摄像头实时主流程。

运行示例：

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\负重箭步蹲.mp4" --hyrox-action lunge --camera-view side
python tools/replay_hyrox_video.py --video "HYROX视频\负重箭步蹲.mp4" --hyrox-action lunge --camera-view side --speed 0.5
python tools/replay_hyrox_video.py --video "HYROX视频\负重箭步蹲.mp4" --hyrox-action lunge --camera-view side --save-debug-csv outputs\hyrox_replay_lunge.csv
python tools/replay_hyrox_video.py --video "HYROX视频\负重箭步蹲.mp4" --hyrox-action lunge --camera-view side --hyrox-config configs/hyrox/lunge.yaml
python tools/replay_hyrox_video.py --video "HYROX视频\负重箭步蹲.mp4" --hyrox-action lunge --camera-view side --headless
python tools/replay_hyrox_video.py --video "HYROX视频\投掷药球.mp4" --hyrox-action wall_ball --camera-view front --headless
python tools/replay_hyrox_video.py --video "HYROX视频\投掷药球.mp4" --hyrox-action wall_ball --camera-view front --save-debug-csv outputs\hyrox_wall_ball.csv
python tools/replay_hyrox_video.py --video "HYROX视频\划船机.mp4" --hyrox-action rowing --camera-view side --headless
python tools/replay_hyrox_video.py --video "HYROX视频\滑雪机.mp4" --hyrox-action skierg --camera-view front --headless
python tools/replay_hyrox_video.py --video "HYROX视频\波比跳远.mp4" --hyrox-action burpee_broad_jump --camera-view side --headless
python tools/replay_hyrox_video.py --video "HYROX视频\推雪橇.mp4" --hyrox-action sled_push --camera-view side --headless
python tools/replay_hyrox_video.py --video "HYROX视频\拉雪橇.mp4" --hyrox-action sled_pull --camera-view side --headless
```

说明：

- 固定复用 `MediaPipeBackend` 做逐帧姿态检测。
- 固定复用 `hyrox/features.py` 和 `hyrox/actions/` 下的实时分析器，不会为视频另写一套动作逻辑。
- 回放窗口会显示原视频、骨架、当前阶段、次数和反馈提示。
- 按 `Q` 或 `ESC` 可退出回放。
- `--speed` 可设为 `0.5`、`1.0`、`2.0` 等正数，控制回放速度。
- `--smoothing` 默认使用与实时主入口一致的 `one-euro`，也可选择 `ema` 或 `none`。
- `--headless` 不打开窗口，适合批量验收；结束时会输出处理帧数、最终阶段和次数。
- `--save-debug-csv` 会把每帧特征值、阶段结果和反馈写入 CSV；固定列至少包含帧号、时间戳、动作、阶段、次数、反馈码、可见度、躯干角、左右膝肘角和身体中心，不适用的字段留空。
- `--hyrox-action` 支持 `lunge`、`wall_ball`、`farmers_carry`、`rowing`、`skierg`、`burpee_broad_jump`、`sled_push`、`sled_pull`；留空的 `--hyrox-config` 会按动作选择对应 YAML。
- `--hyrox-config` 和实时模式共用同一份配置文件，便于拿本地回放调阈值，再回到摄像头实时验证。
- `--camera-view` 应与视频实际机位一致；CSV 会同时记录原始 `camera_view` 和归类后的 `view_profile`。
- 8 份独立 YAML 的字段、推荐视角和单摄像头限制详见 `configs/hyrox/README.md`；自定义配置优先，缺失阈值使用动作内置默认值。

## 会话输出

按 `C` 开始采集，再按 `C` 停止保存。退出程序时如果会话仍在进行，会自动安全保存。

输出目录示例：

```text
outputs/sessions/2026-07-04_162530/
├── metadata.json
├── landmarks.csv
├── kinematics.csv
├── summary.json
├── angle_curves.png
├── velocity_curves.png
└── sequence_summary.json
```

文件说明：

- `metadata.json`：会话 ID、开始/结束时间、摄像头、分辨率、平均 FPS、镜像、平滑参数、模型名、检测帧统计。
- `landmarks.csv`：长表格式关键点数据，每帧每个已启用 profile 的关键点一行，包含 image/world/smoothed 坐标、可见度和 presence；保存时使用开始会话那一刻的当前 profile，`main.py` 默认 `full`，`src.realtime_pose` 默认 `no-face`。
- `kinematics.csv`：每帧关节角、速度、角速度、运动能量代理值和姿态检测状态。
- `summary.json`：角度统计、速度统计、有效姿态帧比例、运动能量代理峰值、可用峰值事件。
- `sequence_summary.json`：峰值事件时间和通用顺序比较结果。
- `angle_curves.png`：左右肘、膝、髋角曲线。
- `velocity_curves.png`：骨盆、左右手腕、左右脚踝速度代理曲线。

## 参考动作与比较输出

参考动作目录示例：

```text
outputs/references/<reference_id>/
├── reference.json
├── source_metadata.json
├── clip_kinematics.csv
├── clip_landmarks.csv
├── features.csv
└── feature_processing.json
```

比较结果目录示例：

```text
outputs/comparisons/<comparison_id>/
├── metadata.json
├── comparison_summary.json
├── aligned_features.csv
├── feature_errors.csv
├── dtw_path.csv
├── angle_comparison.png
├── velocity_comparison.png
├── phase_difference.png
└── report.md
```

`comparison_summary.json` 会记录参考动作 ID、候选会话片段范围、对齐方法、参与计算的特征数量、DTW 距离、差异最大的特征、差异最大的时间区间、数据质量状态和是否应用镜像规范化。

## 深蹲报告输出

输出目录示例：

```text
outputs/squat_reports/<report_id>/
├── metadata.json
├── squat_reps.csv
├── squat_frames.csv
├── squat_summary.json
├── rep_timeline.png
├── angle_curves_by_rep.png
├── symmetry_curves.png
├── annotated_keyframes/
│   ├── rep_001_start.png
│   ├── rep_001_bottom.png
│   └── rep_001_end.png
├── squat_reference_comparison.csv
├── squat_reference_alignment.png
└── report.md
```

如果没有提供 `--reference`，参考比较文件不会生成。保存的关键帧图基于会话中的运动学轨迹标注关键时间点；当前会话文件不包含原始视频帧时，不会伪造摄像头画面。

## 篮球报告输出

输出目录示例：

```text
outputs/basketball/reports/<report_id>/
├── metadata.json
├── shot_summary.json
├── shot_events.csv
├── shot_features.csv
├── chain_sequence.json
├── phase_timeline.png
├── angle_curves.png
├── velocity_curves.png
├── event_sequence.png
├── arm_path.png
├── reference_alignment.png
├── reference_feature_errors.csv
├── keyframes/
│   ├── setup.png
│   ├── dip.png
│   ├── rise.png
│   ├── arm_extension.png
│   ├── release_proxy.png
│   └── follow_through.png
└── report.md
```

如果没有提供 `--reference`，参考对齐文件不会生成。`arm_path.png` 是相机平面轨迹代理图；侧面或未知视角下不会给出高置信横向手臂对齐结论。

## 已实现功能

- `main.py --backend mediapipe` 提供实时优先主链路，默认使用 MediaPipe 33 点姿态检测。
- `main.py --backend auto` 支持动作级 backend 自动选择，未知动作默认 MediaPipe。
- `main.py --backend yolo-pose` 提供 YOLO Pose COCO 17 点姿态 backend；`--yolo-device auto` 在当前 RTX GPU 环境下会自动使用 GPU。
- `main.py` 支持运行中按 `B` 在 MediaPipe 和 YOLO Pose 之间热切换，摄像头保持打开。
- `main.py` 默认启用短暂 pose 丢失保持和手部遮挡跳点保护，并在 metrics 中记录保护触发次数。
- `src/backends/` 提供内部兼容用 `PoseResult` / `PoseBackend`、MediaPipe backend 和 YOLO Pose backend。
- `src/pose/` 提供对外统一的 `NormalizedPose`、17 点公共命名表和 MediaPipe / YOLO Pose adapters。
- `src/utils/smoothing.py` 支持 `none`、`ema`、`one-euro` 三种关键点平滑方式；One Euro Filter 是当前推荐实时模式。
- `src/utils/angle_utils.py` 提供三点夹角、膝、髋、肘、肩和躯干倾角计算，暂不绑定具体运动项目规则。
- `src/realtime/feedback_engine.py` 提供基础实时反馈：`person_lost`、`low_confidence`、`keypoints_unstable`、`angle_available` 和防抖后的反馈文本。
- `src/utils/metrics.py` 统计实时 FPS、平均 FPS、推理耗时、P95 推理耗时、端到端延迟、成功率、丢人次数、关键点 jitter 和角度 jitter，并可保存 CSV。
- `main.py` 支持 `--record` 保存带骨架和文字的视频，`--record-raw` 保存原始画面，`--input-video` 读取视频复跑同一套逻辑，`--save-metrics` 保存指标。
- `src/detectors/yolo_person_detector.py` 提供实验性 YOLO person bbox 检测；默认不加载 YOLO，不影响 MediaPipe baseline。
- `src/utils/roi.py` 提供 bbox 扩大、裁剪、平滑和 ROI 关键点坐标还原。
- `src/fusion/yolo_roi_mediapipe.py` 提供实验性 YOLO 低频 bbox + MediaPipe ROI 检测链路；当前不作为默认策略。
- metrics 额外记录 `person_detector`、`fusion`、`detector_every_n`、ROI 成功率、YOLO 检测耗时、bbox 复用/丢失次数、全图回退次数、`stabilized_hold_count` 和 `occlusion_guard_count`。
- `main.py` 的 MediaPipe backend 使用 Tasks API `VIDEO` 模式同步逐帧推理；内部高级入口 `src.realtime_pose` 使用 `LIVE_STREAM` callback 异步接收结果。
- 每帧传入严格递增的毫秒时间戳。
- 默认摄像头 `0`，Windows 下优先使用 `cv2.CAP_DSHOW`，失败后回退普通 `VideoCapture`。
- 默认镜像显示，并支持运行时切换。
- 绘制 33 个关键点和骨架连接线；肩、肘、腕、髋、膝、踝会高亮。
- 显示滑动窗口 FPS、摄像头编号、检测状态、镜像、录制和会话状态。
- 支持截图、录制、全骨架/投篮关键关节显示模式切换。
- 支持按 `A` 在摄像头不中断的情况下切换 8 个 HYROX 动作，并按 `V` 切换正面/侧面评价档案。
- 提供多摄像头同时打开和同步偏差检查基础层；正式双视角动作融合尚未启用。
- 支持运动学会话采集、CSV/JSON/PNG 导出。
- 支持个人参考动作库、动作片段裁剪、线性重采样、约束 DTW 对齐和比较报告导出。
- 支持深蹲站立校准、状态机计数、离线报告、视角限制说明和个人参考深蹲比较。
- 支持篮球投篮离线裁剪、动作阶段、出手代理、动力链峰值时序、一致性分析和个人参考投篮比较。
- 模型缺失、摄像头无法打开、读帧失败、检测调用异常、无人姿态时都有明确提示。

## 明确限制

当前版本只提供视觉姿态估计和运动学代理指标。

当前版本不能直接测量：

- 地面反作用力。
- 真实关节力矩。
- 肌肉发力大小。
- 肌电活动。
- 医学风险或运动损伤。
- 投篮、射门、健身动作是否标准。
- 个人参考动作是否等同于专业标准动作。
- 深蹲过程中的真实受力、真实关节力矩或肌肉发力。
- 篮球真实离手时刻，除非未来接入并验证篮球检测。
- 投篮命中率或投篮技术是否绝对标准。

当前版本也不实现：

- 投篮姿势评分。
- 关节角医学诊断。
- 动作识别。
- 专项运动标准判断。
- 网页界面。
- 新姿态模型训练。
