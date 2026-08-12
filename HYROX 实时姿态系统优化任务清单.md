# HYROX 实时姿态系统优化任务清单

## 总体目标

保持现有所有功能，不修改 HYROX 8 项动作的业务语义，不改变现有 `VALID / NO_REP / UNSURE`、报告、DTW、网页和桌面功能。

最终链路调整为：

```
普通手机 / 电脑摄像头 / 上传视频
              ↓
        最新帧采集
              ↓
       MediaPipe Pose
              ↓
       2D + World 3D
              ↓
    低延迟 One Euro 平滑
              ↓
     轻量 3D 可靠性处理
              ↓
      角度 / 接触 / 状态机
              ↓
     HYROX 规则与实时反馈
```

必须优先满足：

```
视频播放速度 > 推理完整帧率
```

即：

> 宁可跳过部分“推理帧”，也不能因为推理跟不上而让视频变慢。

MediaPipe 本身就是按这种低延迟思路设计 LIVE_STREAM 的。

## 当前进度（2026-08-11）

第 1–12 轮的工程实现和现有数据验证已完成；第 13 轮离线 OpenCap 参考与第 14 轮目标设备最终验收尚未执行。当前正式 HYROX 阈值和 `VALID / NO_REP / UNSURE` 语义保持不变。第 12 轮严格角度非回归没有全部通过，因此 3D/canonical 3D 仍是验证或辅助证据，没有替换正式 2D 规则。

| 轮次 | 状态 | 主要结果 |
|---|---|---|
| 1 | 已完成 | 统一性能指标、CSV、P50/P95 与 `playback_speed_ratio` |
| 2–3 | 已完成 | 视频/摄像头 Latest-Frame、单槽覆盖、异步 MediaPipe 与正常播放时钟 |
| 4–5 | 已完成 | 显示/分析双滤波、真实时间戳、推理专用自适应分辨率与低开销默认值 |
| 6–10 | 已完成第一版 | 统一 `JointMetric`、3D 可靠性、人体坐标、地板/接触证据和过期姿态门控 |
| 11 | 已完成第一版 | `RealtimeBudgetController` 按固定顺序动态降载 |
| 12 | 已完成框架与现有数据基线 | 六通道角度、旧版/新版比较和覆盖审计；严格非回归未通过 |
| 13–14 | 待完成 | OpenCap 离线参考、真实设备/高速录像和最终发布验收 |

------

## 第 1 轮：先建立延迟与性能基线

**完成状态（2026-08-11）：已完成。** `RealtimeMetrics` 与原 `--save-metrics` CSV 已统一记录帧率、分阶段耗时、时间戳、姿态年龄、跳帧/渲染计数、队列深度、播放速度比及 P50/P95。

**这一轮不改算法，只加监控。**

Codex 任务：

-  保留现有所有功能和输出。
-  给摄像头和上传视频增加统一性能统计。
-  每隔约 1 秒记录一次，不要每帧打印日志。
-  新增以下指标：

```
source_fps
display_fps
inference_fps

capture_ms
preprocess_ms
inference_ms
postprocess_ms
rule_ms
render_ms

frame_timestamp_ms
pose_timestamp_ms
pose_result_age_ms

frames_read
frames_inferred
frames_skipped
frames_rendered

queue_depth
```

特别增加：

```
playback_speed_ratio
```

定义：

```
视频时间前进量 / 实际墙钟时间
```

正常播放时应接近：

```
1.0
```

建议验收目标：

```
0.95 ~ 1.05
```

另外统计：

```
P50 inference latency
P95 inference latency

P50 pose_result_age
P95 pose_result_age
```

现有 `--save-metrics` 保留，直接扩展原来的 CSV，不另外造一套统计系统。

**这一轮验收：**

原有 pytest、golden regression 全部通过，并得到一份修改前性能基线。

------

# 第 2 轮：彻底解决“上传视频变慢放”

**完成状态（2026-08-11）：已完成第一版。** 有窗口的桌面视频回放使用源视频时钟和单槽 `LatestFrameVideo`，推理只消费最新可用帧；网页上传/示例展示按视频自身 FPS 节流，分析速度不再决定播放速度。无窗口批处理仍可选择同步逐帧完整分析。

这是最高优先级。

现在需要把：

```
读一帧
↓
等待 MediaPipe
↓
等待规则
↓
显示
↓
读下一帧
```

改成：

```
                    ┌→ MediaPipe worker
视频解码 → 最新帧 ──┤
                    └→ 正常速度显示
```

核心原则：

> **播放线程绝对不能等待 Pose 推理。**

Codex 实现：

-  建立 `LatestFrameBuffer`。
-  Buffer 最大只保存 **1～2 帧**。
-  禁止形成几十帧的推理队列。
-  新帧到来时，如果旧帧还没有推理，直接覆盖旧帧。
-  推理线程始终读取“最新可用帧”。
-  视频显示根据原视频 FPS / timestamp 前进。
-  不允许 MediaPipe 推理速度控制视频播放速度。

例如 30 FPS 视频：

```
Video:
1 2 3 4 5 6 7 8 9 10

如果 MediaPipe 只能 15 FPS：

Inference:
1   3   5   7   9
```

而不能：

```
1
等推理
2
等推理
3
等推理
```

所以最终：

```
播放：30 FPS
推理：15 FPS
```

是允许的。

而：

```
播放：15 FPS
推理：15 FPS
```

是不允许的。

对于 HYROX，默认先设置：

```
realtime:
  target_pose_fps: 15
  max_pose_fps: 20
  queue_size: 1
```

这些是**工程初值，不是固定科学参数**，后面通过真实视频测试确定。

Wall Ball、Burpee 等快速动作可以以后单独测试是否需要提高到约 20 FPS。

------

# 第 3 轮：实时摄像头改成真正的 Latest-Frame 模式

**完成状态（2026-08-11）：已完成。** 桌面摄像头采用单槽 `LatestFrameCamera`、MediaPipe `LIVE_STREAM/detect_async()` 和仅保存最新结果的 callback；浏览器 Worker 同样只有一个 pending 槽，迟到结果不能覆盖新帧。

实时摄像头也使用同样原则：

```
Camera thread
      ↓
Latest frame
      ↓
MediaPipe detect_async()
      ↓
callback
      ↓
LatestPoseResult
```

MediaPipe 官方明确说明 `detect_async()` 是为了 live stream 设计，会立即返回，而且当计算繁忙时允许丢弃输入帧，从而降低整体延迟。

Codex 完成：

-  LIVE_STREAM 继续使用 `detect_async()`。
-  timestamp 必须严格递增。
-  callback 中禁止做耗时业务。
-  callback 只保存：

```
pose_result
timestamp
confidence
```

-  HYROX 状态机、报告、语音、绘制不要全部堆进 callback。
-  主线程只读取最新 PoseResult。
-  禁止结果队列积压。
-  如果 Pose 太旧，不画错误位置的骨架。

增加：

```
pose_age_ms = current_frame_ts - pose_result_ts
```

例如：

```
pose_age < threshold
    → 正常显示

pose_age 太大
    → 不使用旧关键点进行规则判定
```

建议一开始：

```
max_pose_age_ms: 120
```

之后通过实测调整。

------

# 第 4 轮：降低关节节点“跟手延迟”

**完成状态（2026-08-11）：已完成。** 显示和分析使用独立 One Euro 状态与参数，均使用真实观测时间戳；显示流允许低延迟 raw blend，正式分析流禁止预测。`--landmark-lag-debug` 可同时绘制原始与过滤关键点。

这一轮解决你现在明显感觉到的：

> 人已经动了，骨架过一会才跟上。

重点不是单纯“关闭平滑”。

你现在已有 One Euro，应保留。

Google/MediaPipe 自身也采用视频跟踪和平滑来减少计算和 jitter；不同模型复杂度本身就是精度/延迟之间的权衡。

改成两套独立配置：

```
RAW LANDMARK
     │
 ┌───┴─────────┐
 ↓             ↓
DISPLAY       ANALYSIS
FILTER        FILTER
 ↓             ↓
骨架显示      规则/角度
```

Codex 完成：

-  保留现有显示流和正式分析流分离。
-  Display One Euro 使用更低延迟参数。
-  Analysis One Euro 可以稍强，保证规则稳定。
-  One Euro 必须使用真实 timestamp，而不是假设固定 FPS。
-  速度越快时，滤波自动减弱。
-  静止时加强滤波减少抖动。
-  不允许简单使用很大的滑动平均窗口。

禁止：

```
10 帧 moving average
20 帧 moving average
```

因为这会直接制造明显延迟。

增加调试模式：

```
raw landmark
filtered landmark
```

同时画出，方便观察平滑到底造成了多少帧延迟。

------

# 第 5 轮：降低 MediaPipe 推理开销

**完成状态（2026-08-11）：已完成第一版。** 正式后端保持 MediaPipe、`num_poses=1`、关闭 segmentation 默认开销、复用模型对象；推理图像可缩小而显示图像保持原分辨率，持续超预算时按 640→512→416→320 逐级降低纯推理宽度。

保持：

```
MediaPipe = 产品默认模型
YOLO/RTMW = 实验模型
```

不要更换主模型。

OpenCap Monocular 使用 WHAM + 相机/姿态优化 + OpenSim，本身更适合高精度后处理，不适合直接替换你的低延迟主链。

Codex 完成：

-  `num_poses=1`。
-  segmentation 始终关闭。
-  避免每帧重复初始化模型。
-  避免每帧重复创建大对象。
-  避免不必要的 numpy copy。
-  JSON/CSV 写入移出实时关键路径。
-  debug 日志不得每帧写磁盘。
-  WebSocket/HTTP 返回不要发送不必要的大图像数组。
-  推理图像允许低于显示图像。

例如：

```
Camera:
1280×720

Inference:
640×360 / 640×480

Display:
1280×720
```

关键点是归一化坐标，所以可以重新映射回原图。

增加配置：

```
pose:
  inference_width: 640
  adaptive_resolution: true
```

如果 P95 推理时间过高：

```
降低 inference resolution
```

但：

```
不降低视频播放 FPS。
```

------

# 第 6 轮：建立真正统一的 2D / 3D 角度系统

**完成状态（2026-08-10）：已完成。** 已建立统一 `JointMetric`，集中膝、髋、踝、肩、肘的 raw/smooth 2D/3D 计算、可靠来源选择与视角可观测性；正式 HYROX 阈值仍保持原 2D 输入，可靠 3D 先作为可验证的统一选择值输出。

这是解决“角度到底对不对”的核心。

MediaPipe 本身会输出：

```
image landmarks
world landmarks
```

官方说明 world landmarks 是以髋部中点为原点的米制 3D 坐标。

所以不要继续简单地：

```
有 world landmarks
= 所有规则直接换成 3D
```

改成统一结构：

```
JointMetric:
    raw_2d
    smooth_2d

    raw_3d
    smooth_3d

    selected_value

    source
    confidence
    observable
```

其中：

```
source =
3D
2D
UNAVAILABLE
```

Codex 完成：

-  统一膝、髋、踝、肩、肘等角度定义。
-  所有角度计算集中到 `src/biomechanics/`。
-  禁止每个 HYROX 动作自己写一套重复角度公式。
-  普通三点角支持 2D 和 3D 两种输入。
-  3D 可靠时优先使用 3D。
-  3D 不可靠时使用现有 2D 规则。
-  视角不适合时继续返回 `UNSURE / not_observable`。
-  不删除现在的 `camera_view`。

例如：

```
Side view
→ 2D knee flexion 很可靠

Front view
→ 不把 2D knee flexion 当真实三维膝角

World landmarks 可靠
→ 使用 3D knee angle

World landmarks 不可靠
→ 使用合适视角的 2D
```

继续保留你目前已有的：

```
CAMERA_VIEW_LIMITED
```

这一机制非常重要。

------

# 第 7 轮：加入“OpenCap Lite”式轻量 3D 可靠性层

**完成状态（2026-08-10）：已完成第一版。** 已加入历史中位骨长、左右骨长一致性、孤立关键点速度异常和基于踝/足跟/足尖垂直关系、速度、连续稳定帧的足部接触置信度；这些结果只影响 3D 可靠性或作为辅助证据，不修正骨骼位置、不替换正式规则。

这里**不要加入 WHAM、SMPL、OpenSim**。

只借鉴 OpenCap 最有价值的思想：

> 不要完全相信每一帧网络输出，而是利用人体结构和时序约束判断 3D 是否可信。

OpenCap Monocular 本身通过 WHAM 后继续做相机/姿态优化和生物力学约束，这也是它比直接 3D 回归更可靠的重要原因。

你的实时版只做便宜约束。

### A. Bone Length Consistency

保存一段时间内：

```
shoulder-elbow
elbow-wrist
hip-knee
knee-ankle
```

骨长中位数。

计算：

```
current_length
vs
historical_median
```

如果突然变化很多：

```
3D confidence ↓
```

第一版：

> **只用于 confidence，不强行修改骨骼位置。**

------

### B. 左右骨长一致性

例如：

```
left femur
right femur
```

差异异常：

```
3D confidence ↓
```

------

### C. 速度异常检测

如果某关键点在一帧间：

```
突然跳几十厘米
```

而相邻关节没动：

```
landmark unreliable
```

不允许直接进入动作规则。

------

### D. 轻量脚接触

利用：

```
heel
foot_index
ankle
```

计算：

```
vertical position
+
velocity
+
连续帧稳定性
```

得到：

```
foot_contact_confidence
```

可以辅助：

- Lunge
- Burpee
- Sled
- Farmer Carry

但第一版：

> 不替换已有正式规则，只增加证据。

物理合理的地面/足部接触约束也是 OpenCap、PhysCap 这类方法提升时序稳定性的重要思路。

------

# 第 8 轮：建立人体局部 3D 坐标系

**完成状态（2026-08-10）：已完成第一版。** 已用髋中点为原点、左右髋方向为 X、髋到肩中心方向为 Y、叉乘方向为 Z 建立正交人体坐标系，输出坐标轴、canonical landmarks、可靠性以及 `legacy_angle`/`canonical_3d_angle`。当前正式 HYROX 阈值仍使用 legacy 角度，不允许 canonical 角度直接替换。

不要直接上“真实世界全局坐标”。

先建立：

```
Body Coordinate System
```

例如：

```
Origin:
左右髋中点

Left-Right axis:
left_hip → right_hip

Up axis:
hip_center → shoulder_center

Forward axis:
cross(left-right, up)
```

得到：

```
X = 左右
Y = 上下
Z = 前后
```

然后把 WorldLandmarks 旋转到统一人体坐标。

用途：

```
trunk lean
pelvis orientation
hip flexion
knee flexion
左右偏移
身体前倾
```

这样同一个人：

```
Camera 0°
Camera 30°
Camera 45°
```

计算逻辑会更加统一。

但是：

> **第一版不要修改现有 HYROX 阈值。**

只同时输出：

```
legacy_angle
canonical_3d_angle
```

经过验证以后再逐动作替换。

------

# 第 9 轮：优化地板和触地检测

**完成状态（2026-08-10）：已完成第一版。** 已新增时序 `GroundEstimator`，使用第 7 轮足部接触置信度筛选踝/足跟/足尖稳定样本，以历史中位数估计图像地面并输出 `ground_confidence`；同时融合 2D 地面距离、3D 垂直关系和足部稳定置信度形成辅助 `contact_evidence`。现有局部地板线、虚拟 K/C 点和正式触地规则均保留，证据不足继续为 `UNSURE`。

你目前已有：

```
局部地板线
虚拟膝盖点 K
虚拟胸部点 C
```

应该继续保留，不能直接删除。

新增：

```
GroundEstimator
```

输入：

```
heel
foot_index
ankle
foot_contact_confidence
```

使用若干稳定帧估计地面，而不是每帧重新计算。

输出：

```
ground_confidence
```

然后：

```
2D contact proxy
+
ground confidence
+
3D vertical relation
```

共同决定：

```
contact evidence
```

第一版仍然遵循：

```
证据充分 → VALID / NO_REP

证据不足 → UNSURE
```

不要因为加了 3D 就把不确定情况硬判。

------

# 第 10 轮：解决“旧骨架画在新视频帧上”

**完成状态（2026-08-10）：已完成第一版。** `PoseResult` 已在桌面、Web、本地浏览器与缓存链路统一携带 `frame_id` 和 `timestamp_ms`；桌面显示采用帧差、墙钟年龄、源视频时间轴年龄三重门控，Web 本地推理也会把源姿态年龄传给服务端。80–120 ms 的骨架仅用于淡出，超过 120 ms 不再绘制，也不会进入正式规则、角度或计数。短时 landmark prediction 严格限定为显示实验功能，默认关闭，预测关键点不会上传到分析协议。

这是关节显示延迟经常被忽略的一类问题。

每一个 PoseResult 必须保存：

```
frame_id
timestamp_ms
```

显示时计算：

```
current_frame_timestamp
-
pose_timestamp
```

禁止：

```
Frame 350
+
Frame 345 的 skeleton
```

却不做任何判断。

增加：

```
pose_result_age_ms
```

超过阈值：

```
骨架降低置信度
或暂时不画
```

而不是让用户看到明显滞后的关节点。

之后如果仍然觉得显示滞后，可以增加一个**仅用于显示**的实验功能：

```
短时 landmark prediction
```

例如根据最近速度预测约：

```
1 frame
```

但这个预测：

```
只能画骨架
不能进入 HYROX 规则
不能进入正式角度
不能计数
```

第一版默认关闭。

------

# 第 11 轮：建立自动负载控制

**完成状态（2026-08-11）：已完成第一版。** 已新增 `RealtimeBudgetController`，使用滚动 P95 推理耗时、P95 姿态结果年龄和队列饱和度闭环控制推理负载；降级顺序固定为降低 pose FPS（20→15→12）、降低纯推理分辨率、降低可选额外分析频率。视频读取、播放和渲染时钟不受控制器影响。

新增：

```
RealtimeBudgetController
```

根据：

```
P95 inference_ms
pose_result_age_ms
queue depth
```

动态控制：

```
pose inference FPS
```

例如：

```
机器很快
→ 20 FPS inference

机器一般
→ 15 FPS inference

机器较慢
→ 12 FPS inference
```

原则永远是：

```
优先降低 inference FPS
```

而不是：

```
降低视频播放 FPS
```

如果仍然超预算：

```
降低 inference resolution
```

顺序固定为：

```
① 跳过推理帧
↓
② 降低推理分辨率
↓
③ 降低额外分析频率
↓
④ 最后才考虑更大的功能降级
```

不要让：

```
render loop
video clock
```

跟着 MediaPipe 一起变慢。

------

# 第 12 轮：角度准确性验证

**完成状态（2026-08-11）：已完成验证框架与现有数据基线。** 现有人工标注链路已扩展为同时输出 raw/filtered 2D、raw/filtered 3D、canonical 3D 和正式 selected-rule angle，并输出 MAE、median AE、P90、P95、最低点/完全伸展事件误差及显式旧版/新版非回归比较。现有 150 条人工标注覆盖 Lunge、Wall Ball、Burpee、Rowing，但没有经标定的 30°/45°斜侧素材，也没有成对的旧版/新版程序事件帧；报告会将这两项标为未完成证据，而不会伪造通过。

你现有：

```
manual_angle_annotation.py
compare_manual_angles.py
```

不要重写。

继续扩展现有验证系统。

选择至少：

```
Lunge
Squat-like Wall Ball
Burpee
Rowing
```

然后选择：

```
侧面
30°斜侧
45°斜侧
正面
```

每个测试：

```
raw 2D
filtered 2D

raw 3D
filtered 3D

canonical 3D
selected rule angle
```

输出：

```
MAE
median AE
P90
P95

event timing error

lowest-point error
full-extension error
```

最重要的是比较：

```
旧版
vs
新版
```

不能只看新版数字。

------

# 第 13 轮：OpenCap Monocular 只作为离线参考

这一轮可以稍后再做，不属于实时主链。

建立：

```
tools/opencap_validation/
```

输入同一个视频：

```
              Video
                │
       ┌────────┴────────┐
       ↓                 ↓
Your MediaPipe       OpenCap
       ↓                 ↓
Realtime 3D        Refined 3D
       └────────┬────────┘
                ↓
             Compare
```

比较：

```
动作最低点时刻
完全伸展时刻

膝角趋势
髋角趋势
躯干趋势

动作周期
```

OpenCap 官方实现包含 WHAM、camera/pose optimization 和 OpenSim IK，因此很适合做这种研究参考，而不是塞进实时链。

注意：

> 不要直接认为 OpenCap OpenSim 的 `knee_flexion` 与你的三关键点膝角定义完全相同。

必须统一角度定义以后才能算 MAE。

------

# 第 14 轮：最终回归与验收

所有优化完成后必须满足四组测试。

### 功能回归

原功能全部保留：

```
8 HYROX actions
VALID
NO_REP
UNSURE

camera_view
CAMERA_VIEW_LIMITED

Web
Desktop

upload video
camera

JSON
CSV
text report

voice feedback
DTW
manual validation
experimental backends
```

不能删除。

### 视频性能

重点要求：

```
上传视频播放速度 ≈ 原视频速度
```

建议：

```
playback_speed_ratio
0.95 ~ 1.05
```

并且：

```
queue_depth <= 1~2
```

不允许随着视频时间增长而持续积压。

### 实时延迟

建议目标：

```
median pose_result_age < 60~80 ms

P95 pose_result_age < 120 ms
```

这是工程目标，不是硬性科学标准；如果机器达不到，就通过降低 inference FPS / resolution 达到最低稳定延迟。

最重要的是：

```
新版骨架延迟
<
旧版骨架延迟
```

最好降低：

```
≥ 30%
```

作为初步优化目标。

### 精度

必须满足：

```
rep count 不下降
NO_REP 不明显恶化
UNSURE 机制保留

manual angle MAE 不恶化
P90/P95 不恶化

最低点事件误差不恶化
完全伸展事件误差不恶化
```

只有经过验证的 3D 指标才允许替换现有 2D 正式规则。

------

# 建议最终架构

Codex 最后应该把程序整理成大致如下：

```
Camera / Video
      ↓
FrameSource
      ↓
LatestFrameBuffer
      │
      ├──────────────→ Renderer
      │
      ↓
PoseInferenceWorker
MediaPipe
      ↓
LatestPoseResult
      ↓
Timestamp Sync
      ↓
┌─────────────────────────┐
│ 2D landmarks            │
│ world landmarks         │
└─────────────────────────┘
      ↓
One Euro
      ↓
3D Reliability Layer
├─ visibility
├─ bone consistency
├─ velocity sanity
├─ foot contact
└─ body coordinate
      ↓
Metric Selector
├─ reliable 3D
├─ view-appropriate 2D
└─ UNSURE
      ↓
现有 HYROX FSM
      ↓
现有 Rule Engine
      ↓
VALID / NO_REP / UNSURE
      ↓
Feedback / Report
```

这里最关键的是：

```
Renderer
≠
Pose inference clock
```

和：

```
Video playback clock
≠
Pose inference clock
```

这样才能从架构上解决你现在**“推理越慢 → 视频越慢放 → 骨架越来越滞后”**的问题，而不是靠单纯换模型解决。
