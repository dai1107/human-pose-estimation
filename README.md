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

## 目录结构

```text
.
├── README.md
├── main.py
├── requirements.txt
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
│   ├── reference_features.yaml
│   ├── reference_quality.yaml
│   ├── squat_basic_v1.yaml
│   ├── squat_camera_views.yaml
│   ├── basketball_shot_v1.yaml
│   └── basketball_views.yaml
├── src/
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

基础导入测试：

```powershell
python -m src.import_test
```

单元测试：

```powershell
python -m pytest -q
```

启动当前主实时入口：

```powershell
python main.py
```

默认平滑方式是 One Euro Filter，也可以显式指定：

```powershell
python main.py --backend mediapipe --smoothing one-euro
```

`main.py` 默认使用 `--backend auto`。未知动作走 MediaPipe；如果传入已验证的 HYROX 动作名或视频文件名，会按动作级策略选择 MediaPipe 或 YOLO Pose。

实时窗口打开后，可在普通主链路中按 `B` 在 `mediapipe` 与 `yolo-pose` 之间切换，摄像头不会重新打开。该热切换仅支持 `--fusion none --person-detector none`；切换时会重置关键点平滑器，YOLO Pose 仍按 `--yolo-device auto` 自动使用 GPU。热切换适合实时观察效果，公平 metrics 对比仍建议用固定 backend 分别复跑同一段视频。

`main.py` 默认启用两项实时稳定性保护：短暂丢失 pose 时保留上一帧有效骨架最多 `5` 帧；手腕/手指靠近肩、肘、髋、膝等身体点并触发异常跳变时，短暂保留上一帧稳定位置。画面左上角会显示 `tracking: HOLD` 或 `occlusion_guard`，metrics 会记录 `stabilized_hold_count` 和 `occlusion_guard_count`。可用 `--pose-hold-frames 0` 关闭短暂保留，用 `--no-occlusion-guard` 关闭遮挡保护。

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

新运动项目接入时，不直接凭肉眼观感改默认策略。建议先录制 3 到 5 段代表视频，然后分别用 `mediapipe`、`yolo-pose` 复跑，必要时再单独测 `yolo-roi-mediapipe`。优先看成功率、关键部位缺失率、关键点/角度抖动、P95 推理耗时和端到端延迟，再把推荐 backend 写入 [src/utils/backend_policy.py](<C:\Users\dell\Desktop\pose estimation\src\utils\backend_policy.py:1>)。

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
- `--hyrox-action`：启用 HYROX 实时动作分析，当前支持 `none`、`lunge`。
- `--hyrox-sensitivity`：HYROX 动作灵敏度，支持 `low`、`medium`、`high`，默认 `medium`。
- `--hyrox-config`：HYROX Lunge 配置文件路径，默认 `configs/hyrox/lunge.yaml`；文件缺失时自动回退到默认配置。
- `--metrics-overlay`：启动时显示通用运动学信息面板。
- `--session-autostart`：启动后自动开始一次 session 采集。
- `--analysis-mode`、`--camera-view`、`--shot-type`、`--shooting-side`：深蹲 / 篮球专项实时模式参数。
- `--smoothing`：`one-euro`、`ema` 或 `none`，默认 `one-euro`。
- `--pose-hold-frames`：短暂丢 pose 时保留上一帧有效骨架的帧数，默认 `5`，设为 `0` 可关闭。
- `--occlusion-guard` / `--no-occlusion-guard`：开启或关闭手部遮挡跳点保护，默认开启。
- `--record`：保存带骨架和文字的标注视频。
- `--record-raw`：启动后立即保存原始输入画面；运行中也可按 `T` 开始或停止原始视频录制，用于后续公平复跑。
- `--save-metrics`：把本次运行的性能和稳定性指标追加到 CSV。
- `--headless`：不打开 OpenCV 窗口，适合批量视频评估。

运行时快捷键：

```text
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

如果启动时加 `--hyrox-action lunge`，画面上方还会显示 HYROX 弓步分析面板：`action`、`phase`、`reps` 和最多 2 条中文纠正提示。当前 `phase` 使用连续帧确认后的稳定阶段，不会因为单帧抖动立即切换。

## HYROX Lunge 实时模式

当前 HYROX 实时动作分析第一版支持弓步 `lunge`：

```powershell
python main.py --hyrox-action lunge
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

稳定性规则：

- 同一候选阶段需要连续多帧才会确认，默认 `medium=3` 帧。
- 完成一次 rep 后有 `400ms` 冷却时间，避免抖动重复计数。
- `LOW_VISIBILITY` 出现时会独占提示，并暂停其他动作判断。

## HYROX 视频回放测试工具

第五轮新增了本地视频回放工具，用于验证同一个 `LungeAnalyzer` 在回放视频上的表现。它只是测试工具，不替代摄像头实时主流程。

运行示例：

```powershell
python tools/replay_hyrox_video.py --video "HYROX视频\弓步.mp4" --hyrox-action lunge
python tools/replay_hyrox_video.py --video "HYROX视频\弓步.mp4" --hyrox-action lunge --speed 0.5
python tools/replay_hyrox_video.py --video "HYROX视频\弓步.mp4" --hyrox-action lunge --save-debug-csv outputs\hyrox_replay_lunge.csv
python tools/replay_hyrox_video.py --video "HYROX视频\弓步.mp4" --hyrox-action lunge --hyrox-config configs/hyrox/lunge.yaml
```

说明：

- 固定复用 `MediaPipeBackend` 做逐帧姿态检测。
- 固定复用 `hyrox/features.py` 和 `hyrox/actions/lunge.py`，不会单独维护第二套分析逻辑。
- 回放窗口会显示原视频、骨架、当前阶段、次数和反馈提示。
- 按 `Q` 或 `ESC` 可退出回放。
- `--speed` 可设为 `0.5`、`1.0`、`2.0` 等正数，控制回放速度。
- `--save-debug-csv` 会把每帧特征值、阶段结果和反馈写入 CSV，便于回看阈值与状态机输出。
- `--hyrox-config` 和实时模式共用同一份配置文件，便于拿本地回放调阈值，再回到摄像头实时验证。

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
- `src/backends/` 提供统一 `PoseResult` / `PoseBackend` 基础结构、MediaPipe backend 和 YOLO Pose backend。
- `src/utils/smoothing.py` 支持 `none`、`ema`、`one-euro` 三种关键点平滑方式；One Euro Filter 是当前推荐实时模式。
- `src/utils/angle_utils.py` 提供三点夹角、膝、髋、肘、肩和躯干倾角计算，暂不绑定具体运动项目规则。
- `src/realtime/feedback_engine.py` 提供基础实时反馈：`person_lost`、`low_confidence`、`keypoints_unstable`、`angle_available` 和防抖后的反馈文本。
- `src/utils/metrics.py` 统计实时 FPS、平均 FPS、推理耗时、P95 推理耗时、端到端延迟、成功率、丢人次数、关键点 jitter 和角度 jitter，并可保存 CSV。
- `main.py` 支持 `--record` 保存带骨架和文字的视频，`--record-raw` 保存原始画面，`--input-video` 读取视频复跑同一套逻辑，`--save-metrics` 保存指标。
- `src/detectors/yolo_person_detector.py` 提供实验性 YOLO person bbox 检测；默认不加载 YOLO，不影响 MediaPipe baseline。
- `src/utils/roi.py` 提供 bbox 扩大、裁剪、平滑和 ROI 关键点坐标还原。
- `src/fusion/yolo_roi_mediapipe.py` 提供实验性 YOLO 低频 bbox + MediaPipe ROI 检测链路；当前不作为默认策略。
- metrics 额外记录 `person_detector`、`fusion`、`detector_every_n`、ROI 成功率、YOLO 检测耗时、bbox 复用/丢失次数、全图回退次数、`stabilized_hold_count` 和 `occlusion_guard_count`。
- MediaPipe Tasks API `PoseLandmarker`，运行模式为 `LIVE_STREAM`。
- 通过 callback 异步接收最新姿态结果。
- 每帧传入严格递增的毫秒时间戳。
- 默认摄像头 `0`，Windows 下优先使用 `cv2.CAP_DSHOW`，失败后回退普通 `VideoCapture`。
- 默认镜像显示，并支持运行时切换。
- 绘制 33 个关键点和骨架连接线；肩、肘、腕、髋、膝、踝会高亮。
- 显示滑动窗口 FPS、摄像头编号、检测状态、镜像、录制和会话状态。
- 支持截图、录制、全骨架/投篮关键关节显示模式切换。
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
