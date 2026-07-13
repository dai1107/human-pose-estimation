# HYROX 姿态分析

本项目提供两种使用方式：

- **网页版**：在电脑或手机浏览器中使用本机摄像头，支持多人匿名会话、实时骨架与动作反馈、上传视频分析，以及 JSON/CSV 报告下载。
- **桌面版**：通过 OpenCV 窗口使用摄像头或视频文件，适合本机调试、录像、指标导出和模型对比。

当前聚焦 8 项 HYROX 动作的人体姿态识别、重复计数和实时指导；独立深蹲与篮球投篮分析模式已移除。

## 支持的动作

| 参数 | 动作 | 主要输出 |
|---|---|---|
| `lunge` | 负重箭步蹲 | 阶段、次数、步幅与膝部稳定性提示 |
| `wall_ball` | Wall Ball | 下蹲/起身/投掷伸展阶段、次数和深度提示 |
| `rowing` | Rowing | 划船阶段、次数与躯干/手臂时序提示 |
| `skierg` | SkiErg | 拉动周期、次数与髋铰链提示 |
| `burpee_broad_jump` | Burpee Broad Jump | 波比与跳远阶段、次数和落地提示 |
| `sled_push` | Sled Push | 推行状态、躯干角度与步态提示 |
| `sled_pull` | Sled Pull | 拉动状态、躯干角度与左右对称提示 |
| `farmers_carry` | Farmers Carry | 搬运状态、躯干稳定与左右对称提示 |

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

主要模型文件：

- `models/pose_landmarker_full.task`：MediaPipe Pose；
- `models/hand_landmarker.task`：可选手部关键点；
- `yolo11n-pose.pt`：YOLO Pose；
- `yolo11n.pt`：可选人体检测器。

## 网页版快速开始

本机使用可双击 `启动网页.bat`，或在命令行运行：

```powershell
.venv\Scripts\python.exe start_web.py
```

程序会打开 `http://127.0.0.1:5000`。选择“本机摄像头”，点击“开启摄像头”并允许浏览器的视频权限，然后点击“开始实时分析”。

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
- 分析结果可下载为 JSON/CSV，停止后最多保留 10 分钟；
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

## 视频回放工具

```powershell
python -m tools.replay_hyrox_video `
  --input-video "HYROX视频\药球.mp4" `
  --hyrox-action wall_ball `
  --camera-view front
```

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
python -m py_compile main.py src\realtime_pose.py webui\app.py webui\realtime.py start_web.py start_public_web.py
node --check webui\static\app.js
```

## 限制

- 当前结论来自人体关键点，是视觉运动学代理，不是医疗诊断；
- Wall Ball 不检测药球、目标命中或目标高度；
- Sled Push / Pull 不检测器械或真实负载；
- Rowing / SkiErg 不读取器械阻力、功率或距离；
- Farmers Carry 不检测哑铃重量；
- 拍摄视角、遮挡、光照和多人干扰会影响识别质量；
- 网页版虽已完成会话隔离和容量限制，但尚未完成实体设备兼容验收、50 路压力测试和正式公网部署。
