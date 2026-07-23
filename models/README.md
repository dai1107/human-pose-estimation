# 模型文件

该目录存放 MediaPipe、手部和可选 RTMW 模型；YOLO 权重位于项目根目录。发布包会
包含 MediaPipe、手部和 YOLO 小型权重，但约 229 MB 的 RTMW-X/L ONNX 文件需要
单独下载。网页版提供三种可手动选择的姿态方案：纯 MediaPipe、
YOLO + MediaPipe、YOLO + RTMW WholeBody；“自动选择”只是按动作选择标准后端，
不是第四种模型。

## MediaPipe

Pose Landmarker 产品文件：

```text
models/pose_landmarker_lite.task
models/pose_landmarker_full.task
```

Pose Landmarker 官方下载地址：

```text
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

网页本机 Worker 提供 `自动基准 / Full / Lite` 三种档位。自动基准在真实摄像头帧上比较两种模型，但还受离线黄金精度门控制；当前 Lite 只有 `1/8` 个黄金动作通过严格等价门，不能被自动选择。Full 是正式默认档，Lite 仅在用户明确选择速度优先实验时启用；Heavy 不进入产品包。

Hand Landmarker 是可选模型。使用纯 MediaPipe 或 YOLO + MediaPipe 并开启
“显示手指节点”时，它负责补充五指覆盖层；页面显示每只手的 20 个非手腕关节点。
选择 RTMW WholeBody 时会优先使用 RTMW 自带的双手各 21 点，不需要额外运行
Hand Landmarker。

Hand Landmarker 推荐文件名：

```text
models/hand_landmarker.task
```

Hand Landmarker 官方下载地址：

```text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## YOLO

项目根目录包含：

```text
yolo11n-pose.pt
yolo11n.pt
```

`yolo11n-pose.pt` 用于 YOLO Pose、网页 YOLO + MediaPipe 身份锁定，以及 RTMW
前置目标锁定；`yolo11n.pt` 用于桌面版可选的人体检测 ROI。先安装 YOLO 依赖：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-yolo.txt
```

网页手动选择 `纯 MediaPipe` 时不会加载任何 YOLO 权重；手动选择
`YOLO + MediaPipe` 时，所有动作都会固定运行 YOLO 锁定与 MediaPipe 同人补点，
不再只对特定动作隐式融合。自动模式仍可按动作选择标准后端。

## RTMW WholeBody

网页版高级设置提供显式的 `YOLO + RTMW WholeBody（高精度 133 点）` 选项。YOLO 先锁定一个目标运动员，RTMW-X/L 再在该人物框中输出身体、脚部、面部和双手共 133 点。系统使用最多 13 个共有身体点复核 YOLO 与 RTMW 是否指向同一个人，身份复核通过后才接受 RTMW 结果。

预期文件：

```text
models/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx
```

OpenMMLab 官方压缩包：

```text
https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip
```

解压后的 ONNX 文件约 229 MB，已在 `.gitignore` 中忽略。如果模型或 ONNX Runtime 不可用，网页会明确显示 `YOLO + MediaPipe（RTMW 降级）`，不会直接终止当前分析。

只安装一种 ONNX Runtime：

```powershell
# 通用 CPU 环境
.venv\Scripts\python.exe -m pip install -r requirements-rtmw-cpu.txt

# NVIDIA GPU 环境（Windows，CUDA 13 / cuDNN 9）
.venv\Scripts\python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu
.venv\Scripts\python.exe -m pip install -r requirements-rtmw-gpu.txt
```

不要在同一环境中同时保留 `onnxruntime` 和 `onnxruntime-gpu`，两者提供相同的 Python 包，可能相互覆盖 Provider 文件。

检查当前运行环境：

```powershell
.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.get_device()); print(ort.get_available_providers())"
```

GPU 环境应至少包含 `CUDAExecutionProvider`；只有 `CPUExecutionProvider` 时仍可运行，但 RTMW-X/L 的处理速度会明显降低。自 2026-07-21 起，正式网页摄像头、上传视频和默认桌面链路只使用 MediaPipe Pose，YOLO/RTMW 仅供显式实验、离线比较、缓存生成和研究消融；安装 RTMW 不会让产品网页出现模型切换项。实验运行时模型仍在服务器执行，手机摄像头、编码性能与网络质量会影响端到端体验。
