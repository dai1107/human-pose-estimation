# 模型文件

该目录存放不随源码分发的姿态和手部模型。网页版提供三种可手动选择的姿态方案：MediaPipe、YOLO Pose、YOLO + RTMW WholeBody；“自动选择”只是在标准后端之间按动作选择，不是第四种模型。

## MediaPipe

Pose Landmarker 推荐文件名：

```text
models/pose_landmarker_full.task
```

Pose Landmarker 官方下载地址：

```text
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

Hand Landmarker 是可选模型。使用 MediaPipe 或 YOLO Pose 并开启“显示手指节点”时，它负责补充五指覆盖层；页面显示每只手的 20 个非手腕关节点。选择 RTMW WholeBody 时会优先使用 RTMW 自带的双手各 21 点，不需要额外运行 Hand Landmarker。

Hand Landmarker 推荐文件名：

```text
models/hand_landmarker.task
```

Hand Landmarker 官方下载地址：

```text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

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

GPU 环境应至少包含 `CUDAExecutionProvider`；只有 `CPUExecutionProvider` 时仍可运行，但 RTMW-X/L 的实时帧率会明显降低。手机浏览器不直接加载该模型，YOLO 和 RTMW 都在服务器执行；手机摄像头、编码性能与网络质量仍会影响端到端体验。
