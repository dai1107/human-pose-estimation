# HYROX 姿态分析

本项目提供两种使用方式：

- **网页版**：在电脑或手机浏览器中使用本机摄像头，支持多人匿名会话、实时骨架与动作反馈、逐次动作语音改进提示、上传视频分析，以及文字/JSON/CSV 报告下载。
- **桌面版**：通过 OpenCV 窗口使用摄像头或视频文件，适合本机调试、录像、指标导出和模型对比。

当前聚焦 8 项 HYROX 动作的人体姿态识别、人体规则有效计数、距离动作分析周期和实时指导；独立深蹲与篮球投篮分析模式已移除。正式网页、桌面摄像头和默认视频分析统一使用 MediaPipe Pose；YOLO Pose、YOLO + MediaPipe 与 YOLO + RTMW 仅保留为显式实验、离线比较和研究消融入口，不会被产品 `auto` 自动加载。实时关键点默认使用 One Euro 平滑，参考动作比较支持 DTW 对齐。

## 文档导航

- [完整使用说明](使用说明.md)：网页版、桌面版、拍摄、计数语义、调试与故障排查；
- [网页版使用说明](网页版使用说明.md)：浏览器摄像头、匿名会话、隐私、报告和部署；
- [模型安装说明](models/README.md)：MediaPipe、手部模型、RTMW 权重与 CPU/GPU 环境；
- [动作配置说明](configs/hyrox/README.md)：默认阈值、触地/脚部事件、可观测性与自定义配置；
- [变更记录](CHANGELOG.md)：版本变更摘要；
- [姿态模型与延迟优化实施记录](姿态模型使用更改和延迟优化.md)：产品后端收敛、最新帧流水线、3D Assist 与逐轮验证；
- [完整延迟审计与高速录像验收](延迟审计与高速录像验收.md)：网页/桌面逐段时间线、瓶颈报告与 120/240 FPS 外部 sensor-to-photon 验收；
- [发布与升级说明](RELEASING.md)：版本规则、构建、发布和 schema 兼容策略；
- [程序成熟度审计与实施清单](程序成熟度审计与实施清单.md)：已完成工程化项目、当前限制与后续优先级。

## 2026-07-23 改进摘要

今天完成的产品链路与工程改进如下：

- **产品后端收敛**：网页摄像头、上传视频、默认桌面运行和 `auto` 统一使用 MediaPipe Pose；YOLO Pose、YOLO + MediaPipe 与 YOLO + RTMW 只保留在显式实验、离线比较和研究入口。
- **固定示例零模型推理**：8 个内置示例使用 v2 指纹校验缓存，逐帧保存 image landmarks、可用的 world landmarks 和手部关键点，并校验视频、姿态模型、手部模型及必要 YOLO 模型。页面显示“预计算示例结果”，姿态推理时间为 `0.0 ms`；箭步蹲的 YOLO 人物锁定与对应 world landmarks 身份匹配仅发生在离线缓存生成阶段。缓存缺失、损坏、版本过旧或指纹不一致时才安全回退真实推理。
- **实时摄像头低延迟**：桌面采用最新帧优先；网页由 `requestVideoFrameCallback` 驱动本机 MediaPipe Worker，忙时只保留最新待处理帧，本机结果立即绘制，只把 image/world landmarks 发给 Python 执行 HYROX 规则。默认“自动基准”会用真实摄像头帧在约 3 秒内比较 Full/Lite 的 P50、P95、姿态 FPS 和检出率；当前 Lite 黄金回归未通过产品门，因此自动档保持 Full，只有用户显式选择时才启用 Lite。Worker/WASM/模型加载失败或持续过载时会明确回退服务器兼容姿态；手指跟踪默认关闭。
- **摄像头诊断与本地优先收口**：网页请求 640×480@60、最低 30 FPS，并报告实际轨道设置、呈现 FPS、帧间隔、低光和重复帧；桌面 `auto` 使用设备基准缓存，不再永久优先 DSHOW。浏览器只发送原始 landmarks 给 Python，常速度预测仅用于 Canvas；协议拒绝预测字段进入 HYROX 和正式报告。神经网络、时序模型及训练流程均未实现。
- **显示/分析双姿态流**：Python 正式分析流保持独立 `responsive` One Euro 且禁止预测；浏览器显示流使用 `ultra_responsive`（`min_cutoff=2.2`、`beta=0.12`），按速度和可见度最多混入 `0.45` 原始坐标。腕踝、脚跟和脚尖响应最高，肩髋降低混合，面部不做 raw blend；显示结果不会进入状态机、计数或正式报告姿态字段。
- **3D Assist 与三维角度显示**：MediaPipe world landmarks 并行提供膝、髋、肘、肩三维角度和可靠性审计。网页角度叠加只显示通过质量门的三维关节角并标注 `3D`，不可靠时隐藏该角度，不再用二维显示值冒充三维；可靠且与二维一致的证据只增强相关规则置信度，严重冲突会降为 `UNSURE`，不可用时动作判定完整回退二维。动作阶段、二维阈值、触地、地板、起落、腕部时序、补步和距离规则没有被三维替换。
- **工程与验收**：桌面入口已拆分为 `src.realtime` 下的 CLI、捕获、后端、显示、录制、会话和 HYROX 分析组件；旧 `src.realtime_pose` 只保留兼容转发。新增 8 视频黄金回归、版本化报告和 30/60 分钟耐久工具。

旧服务器兼容姿态的本机 30 帧协议探针（640 px、JPEG 0.65）测得往返延迟 P50 `18.7 ms`、P95 `35.5 ms`，此前轮询链路约为 `58 ms`；这组数字不代表新的浏览器本机姿态链路。新链路需使用完整延迟审计与高速录像重新建立设备基线。固定箭步蹲示例 133 帧的无播放节流计算基准约 `1.1 s`；页面始终按源视频 FPS 播放。

Full/Lite 的 8 视频对比工具会审计关键点、关节角、动作计数、触地、脚部事件和 3D Assist；本轮实测 Lite 仅 `1/8` 通过严格产品门，因此 `lite_auto_approved=false`。详细结论见 [Lite 与 Full 模型档位回归](Lite与Full模型档位回归.md)。

### 实时延迟优化前九阶段状态

| 阶段 | 已完成内容 | 当前边界 |
|---|---|---|
| 1. 完整延迟审计 | 网页和桌面逐段记录采集、复制、编码、推理、结果、绘制与预计显示时间；输出姿态—视频年龄差，并提供高速录像验收工具 | 摄像头曝光、内部缓冲和显示器扫描延迟仍需在目标硬件用 120/240 FPS 外部录像测量 |
| 2. 视频帧驱动显示 | 网页使用 `requestVideoFrameCallback` 建立单调帧身份，采集提交和骨架 Canvas 共用同一视频帧时钟；旧浏览器明确回退 `requestAnimationFrame` | 单 Canvas 仍不是默认方案，是否启用需用目标浏览器实测拷贝成本 |
| 3. 浏览器本机姿态 | MediaPipe 在模块 Worker 中运行，主线程转移 `VideoFrame`/`ImageBitmap`，忙时只保留最新槽位；本机结果立即绘制，只向 Python 发送原始关键点 | Worker/WASM 失败或设备持续不可用时仅按 `server_pose_fallback` 明确回退；禁止回退时直接报错 |
| 4. Lite/Full 档位 | 自动档约 3 秒比较两档 P50/P95、姿态 FPS、检出率和主线程长任务；完整双模型回归报告已生成 | Lite 原黄金区间仅 `3/8`、严格等价门仅 `1/8`，所以自动档保持 Full，Lite 仅显式实验 |
| 5. 显示/分析双流 | 分析流保持 responsive 且强制无预测；显示流采用 ultra-responsive、速度相关 raw blend、节点分组和独立 image/world 状态；切换动作/模型会重置 | 显示 landmarks 不发送规则链，也不写入正式报告姿态字段 |
| 6. 显示层短时预测 | 独立 `DisplayPosePredictor` 根据姿态采集/呈现时间到当前预计显示时间动态外推 0–45 ms；包含速度平滑、低置信度/断流禁用、人体尺度位移上限、反向衰减及支撑脚约束 | 预测只作用于 Canvas 绘制副本；Python 规则、计数、触地和报告继续使用真实观测 |
| 7. 主线程渲染优化 | 骨架继续跟随视频帧；指标限制为 5 FPS、报告统计 3 FPS、3D 角度文字 12 FPS，反馈 DOM 仅内容变化时重建；关键点坐标使用固定缓冲区，并缓存连接、字体和视频变换矩阵 | 延迟审计保存渲染循环、Canvas、DOM P95 以及 Long Task 总量和阶段归因；硬件数值需真实会话采样 |
| 8. 摄像头与硬件诊断 | 网页显式请求 640×480@60、最低 30 FPS，并保存实际轨道设置、实际 FPS、帧间隔、亮度和重复帧诊断；桌面提供 default/DSHOW/MSMF、MJPG/YUY2 设备基准与精确配置缓存 | 未自动打开真实摄像头；sensor-to-photon 与目标设备结果必须现场实测，报告不得用软件时间伪造 |
| 9. 本地优先架构收口 | 浏览器本地 MediaPipe 负责即时视觉，Python 只用原始 landmarks 执行 HYROX；桌面全部在本机运行；服务器兼容回退可严格配置 | 常速度预测仅传给 Canvas，协议白名单禁止预测结果进入规则、计数与正式报告；神经网络、时序模型和训练流程未实现 |

第八阶段与本地优先收口后的自动化基线为 Python `530 passed`、Node `16 passed`，Full 黄金视频回归 `8/8`；真实摄像头后端与 sensor-to-photon 仍需在目标设备主动测试。

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
pose-golden --list
pose-endurance --help
pose-camera-benchmark --help
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
  → 原始人体关键点
      ├─ 分析流：responsive One Euro → Python 规则
      └─ 显示流：ultra-responsive + 速度 raw blend → Canvas
  → 计算关节角度、相对位置、可见度、触地和脚步事件
  → 动作专属状态机确认当前阶段
  → 当前阶段技术反馈 + 完整周期人体规则验证
  → 可观测性门控
  → 骨架颜色、关节角度、文字/语音反馈和计数结果
```

完整处理过程如下：

1. **识别并拆分关键点流**：姿态后端输出肩、肘、腕、髋、膝、踝、脚跟和脚尖等原始关键点及置信度。原始点进入 Python 的独立 `responsive` 分析滤波后用于规则；浏览器显示副本进入 `ultra_responsive` 和速度相关 raw blend 后只用于 Canvas。两流都使用姿态对应帧的真实采集时间，不使用显示刷新时间或固定 30 FPS；超过 250 ms 的观测间隔会各自重置。
2. **计算二维运动学特征**：系统计算左右膝/髋/肘/肩角度、躯干角、肩髋高差、髋膝相对高度、手腕与肩髋位置、身体中心移动、人体尺度归一化距离和各身体区域可见度。需要触地或起落判断的动作还会使用局部地板、膝/胸虚拟表面、左右脚支撑、起跳、落地和碎步事件。
3. **识别动作阶段**：每个动作有独立状态机，例如 Lunge 使用 `stand → descent → bottom → ascent → stand`，Rowing 使用 `catch → drive → finish → recovery`。规则只在适用阶段运行，阶段切换还会经过时序、端点顺序和冷却门控；离线分析和“低”灵敏度通常要求连续多帧，实时“中/高”灵敏度允许一个已处理帧确认短暂关键端点。
4. **生成实时技术反馈**：动作分析器将当前特征与动作配置阈值比较，输出带问题码、严重级别和置信度的中文提示。例如 Lunge 最低点不够深会输出 `NOT_DEEP_ENOUGH`，躯干倾斜过大会输出 `LEAN_TOO_MUCH`。默认每帧最多显示两项；可见度不足时只显示取景提示，不继续给出不可靠的技术结论。
5. **验证完整动作候选**：Lunge、Wall Ball 和 Burpee Broad Jump 在关键端点顺序完成后，分别检查该动作的必需人体规则。每条规则先得到 `PASS`、`FAIL`、`UNSURE` 或 `NOT_APPLICABLE`；任一必需规则明确 `FAIL` 得到 `NO_REP`，没有失败但存在无法判断的必需规则得到 `UNSURE`，全部通过才得到 `VALID`。技术质量提示与计数必需规则相互独立，训练建议不一定取消计数。
6. **执行可观测性门控**：规则聚合后还会检查整次可见度、必需关键点、决定性规则置信度、地板、拍摄视角和失败持续帧数。证据不足时，即使初步结果为 `VALID` 或 `NO_REP`，也会降级为 `UNSURE`。
7. **呈现问题位置**：网页把通过 world-landmark 质量门的三维关节角标在对应关节附近，并明确标注 `3D`；缺失或不可靠时不显示该角度，也不会用二维角替代。明显超出当前阶段参考范围时显示红色，通过时显示绿色，接近边界时显示中性。某些没有明确姿态标准的过渡阶段不会计入报告的可评价帧。非角度问题（例如补步、触地、左右不同步）主要通过“动作反馈”和报告说明。

正式分析 One Euro 默认使用 `responsive`；`stable` 更强调稳定，`balanced` 在稳定与响应之间折中。分析参数位于 `configs/product_pose.yaml` 的 `analysis_smoothing`，并强制 `prediction_enabled: false`。显示参数位于 `display_smoothing`，固定 `ultra_responsive` 起点及 `max_raw_weight: 0.45`；短时显示预测参数位于 `display_prediction`，默认最大预测时间 45 ms、最大位移为人体尺度的 0.06、100 ms 无新姿态即停止外推。`rendering` 配置控制角度文字、指标和统计刷新频率，以及 P95 固定采样窗口容量。显示滤波、预测和分析滤波状态相互独立。桌面可用 `--smoothing-profile stable|balanced|responsive` 临时覆盖分析档位，但不会改变网页显示滤波。旧的 `realtime_smoothing` 配置键仍可单独兼容读取，不能与 `analysis_smoothing` 同时定义。

MediaPipe world landmarks 当前处于严格的 `assist` 模式：系统并行计算膝、髋、肘、肩的 3D 角度并执行 visibility/presence、骨段、z 跳变、角速度、身份交换、姿态年龄和观测间隔门控。网页画面显示通过质量门的 3D 角度并明确标注 `3D`；缺失或不可靠的角度不显示。所有用于动作状态机的 `selected_angle` 仍取二维值，动作阶段和规则 PASS/FAIL 仍使用原有二维阈值；可靠且与二维一致的 3D 只提高映射角度规则的置信度，原本 `UNSURE` 的规则不会被越级改为 `VALID`，严重 2D/3D 冲突会将相关候选降为 `UNSURE`，3D 缺失或不可靠时则保持完全二维回退。触地、地板、髋膝图像高度、腕部位置与同步、起跳、落地、补步和距离规则不使用 3D。网页和桌面报告继续保存 2D/3D 差值、Assist 状态、冲突比例与失败原因；这些统计依赖实际视频，程序不会伪造硬件或动作结论。

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
- `yolo11n-pose.pt`：实验性 YOLO11n Pose；
- `models/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx`：实验性 RTMW-X/L WholeBody 133 点，约 229 MB，不提交到 Git；
- `yolo11n.pt`：实验性人体检测器。

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

实时摄像头和上传视频只使用 `MediaPipe Pose`；`auto` 仅作为旧请求兼容值并直接映射到 MediaPipe。8 个固定示例读取随版本生成的 v2 姿态、world landmarks 与手部关键点缓存，模型栏显示“预计算示例结果”，播放时不再加载姿态或手部模型。负重箭步蹲缓存由离线 YOLO 人物锁定与 MediaPipe image/world 关键点身份匹配生成，用于避开样例中的背景人物；该 YOLO 路径不会进入摄像头或上传视频会话。视频、相关模型或缓存 schema 变化时缓存会自动失效并回退真实推理。

## 网页版快速开始

本机使用可双击 `启动网页.bat`，或在命令行运行：

```powershell
.venv\Scripts\python.exe start_web.py
```

程序会打开 `http://127.0.0.1:5000`。选择“本机摄像头”，点击“开启摄像头”并允许浏览器的视频权限，然后点击“开始实时分析”。

### 逐次动作语音提示

“动作反馈”右上角的“语音开/关”默认开启。系统在一次人体规则候选或分析周期完成时，汇总该段动作中持续出现的问题以及清晰关键阶段的角度偏离，立即播报最多两条改进建议；没有持续性问题时不会播放无意义提示。农夫行走不按次数拆分，会在动作问题持续约 1.2 秒后播报，同一问题至少间隔 8 秒才会再次提示。

语音使用浏览器 Web Speech API 在当前设备本机合成，不申请麦克风权限、不向服务器上传音频，也不会写入录制视频。开关选择会保存在当前浏览器；若浏览器不支持语音合成，页面会显示“语音不可用”。

页面的“高级设置”中，“骨架设置”只选择完整骨架、仅上半身或仅下半身；“显示手指节点”“隐藏面部”和“镜像画面”是与其分开的同级开关。按需开启手指节点后，实时摄像头和上传视频按约 10 FPS 运行独立的 MediaPipe Hand Landmarker；固定示例直接读取预计算手部关键点。页面为每只手绘制五根手指的 20 个非手腕关节点，手部短暂漏检时最多保留约 0.35 秒以减少闪烁。为降低实时延迟，手指节点默认关闭；隐藏面部默认关闭，摄像头模式默认开启镜像，示例和上传视频默认关闭镜像。

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
- `--backend mediapipe`：正式支持的桌面姿态后端；`auto` 为映射到 MediaPipe 的兼容值，`yolo-pose` 仅用于显式离线实验；
- `--experimental-backends`：显式启用摄像头实验后端和后端热切换，并显示实验性警告；
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
src/realtime/            # 统一桌面运行时：CLI、捕获、后端、显示、录制、会话与 HYROX 分析
src/validation/          # 8 视频黄金回归与性能/耐久验收
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

第四批成熟度验收另提供两条可安装命令。黄金回归固定覆盖仓库内 8 段视频的帧数、
姿态检出率、候选数、有效/未完成/不确定数和分析周期区间：

```powershell
# 完整 8 视频回归；结果写入版本化 JSON
pose-golden --report outputs\validation\hyrox_golden_report.json

# 秒级管线冒烟（不替代正式耐久验收）
pose-endurance --duration-seconds 10 --report outputs\validation\endurance_smoke.json

# 正式 30/60 分钟验收；分别运行并保存独立报告
pose-endurance --minutes 30 --report outputs\validation\endurance_30m.json
pose-endurance --minutes 60 --report outputs\validation\endurance_60m.json
```

耐久报告包含平均 FPS、P95 端到端帧延迟、起止/峰值进程内存、内存增长、视频循环
次数、异常读帧率、完成状态和帧/延迟记录完整性。默认阈值可通过命令行覆盖，便于不同
CPU/GPU 机器建立各自基线。`src.realtime_pose` 已停止维护独立循环；旧导入会显示弃用
提示并转发到同一个桌面运行时。

Windows/Linux CI 会在 Python 3.10 和 3.12 上执行依赖安装、导入、编译、文本格式、
全量单测、无摄像头冒烟和发布包构建。自动化测试还覆盖通用接触检测、脚部事件、
`VALID/NO_REP/UNSURE` 可观测性门控、三项人体有效计数动作、距离动作违规边界、
输出 schema/保留策略、相同特征流重复运行确定性、旧入口转发、黄金区间与耐久报告
判定。第八阶段与本地优先架构收口后的当前基线为 Python `530 passed`、Node
`16 passed`；Full 黄金视频回归为 `8/8`。

## 限制

- 当前结论来自人体关键点，是视觉运动学代理，不是医疗诊断；
- Wall Ball 不检测药球、目标命中或目标高度；
- Sled Push / Pull 不检测器械或真实负载；
- Rowing / SkiErg 不读取器械阻力、功率或距离；
- 距离动作的 `cycle_count` 不是官方有效次数，系统不确认雪橇过线、Rowing/SkiErg 完成里程或训练区间外的规则状态；
- Farmers Carry 不检测壶铃重量、真实携带距离或完成线；
- 拍摄视角、遮挡、光照和多人干扰会影响识别质量；
- 自动化验证没有打开真实摄像头；后端表现和 sensor-to-photon 必须在目标设备现场测量；
- 网页版虽已完成会话隔离和容量限制，但尚未完成实体设备兼容验收、50 路压力测试和正式公网部署。
