# HYROX 动作配置

本目录为 8 个 HYROX 动作分别保存独立阈值。实时入口 `main.py` 与离线回放工具
`tools/replay_hyrox_video.py` 共用同一配置和同一分析器；不指定 `--hyrox-config`
时，会按 `--hyrox-action` 自动加载对应文件。自定义文件只需写需要覆盖的字段，缺失
字段会使用代码内安全默认值。配置采用严格校验：未知或重复字段、错误类型、越界值、
不匹配的 `action_name`、超过一层的嵌套及其他未支持 YAML 结构都会以 `CFG001`
拒绝启动，不再静默忽略。修改后可运行 `python -m src.doctor` 验证全部配置。

## 文件与建议视角

| 文件 | 用途 | 建议视角 |
|---|---|---|
| `lunge.yaml` | 负重箭步蹲阶段、深度与伸展提示 | 侧面或斜侧面，全身入镜 |
| `wall_ball.yaml` | 深蹲、起身投掷阶段与计数 | 正面或斜前方，手腕与脚踝均入镜 |
| `farmers_carry.yaml` | 连续搬运监控与手臂位置违规代理 | 正面或斜前方，全身入镜 |
| `rowing.yaml` | 划船分析周期与训练区间站起违规代理 | 优先侧面 |
| `skierg.yaml` | 上拉、下拉分析周期与技术提示 | 正前方或斜前方 |
| `burpee_broad_jump.yaml` | 波比与向前跳的组合计数 | 侧面约 45°，保留落地区域 |
| `sled_push.yaml` | 推动姿态、步态与检测步数 | 侧面或斜侧面 |
| `sled_pull.yaml` | 拉动分析周期、跪姿/坐姿违规代理与左右同步 | 侧面或斜侧面 |
| `contact.yaml` | 通用膝盖/胸部代理触地阈值与分割地板带 | 侧面或斜侧面，先建立可靠地板参考 |
| `foot_events.yaml` | 左右脚支撑、同步起落、错位代理与碎步阈值 | 侧面或斜侧面，脚跟和脚尖完整入镜 |
| `observability.yaml` | 统一 `VALID/NO_REP/UNSURE` 证据质量门控 | 所有需要人体规则计数的动作 |

## 配置在判断链路中的位置

姿态模型只输出人体关键点；本目录的配置与动作分析器共同把关键点转换成阶段、技术问题和计数结论。执行顺序为：

```text
关键点
  → 通用运动学特征
  → 动作 YAML 阶段阈值
  → 动作专属技术反馈与完整端点顺序
  → 接触/脚部事件规则
  → 必需人体规则聚合
  → observability.yaml 证据质量门控
```

各层职责如下：

| 层级 | 主要来源 | 作用 |
|---|---|---|
| 关键点与运动学特征 | 姿态后端、`hyrox/features.py` | 计算角度、相对位置、可见度和人体尺度归一化值，不直接输出动作结论 |
| 阶段与实时反馈 | 各动作 YAML、`hyrox/actions/*.py` | 确认当前阶段并生成技术问题；`feedback_limits` 控制单帧最多显示多少项 |
| 触地与脚部事件 | `contact.yaml`、`foot_events.yaml` 及动作专属覆盖值 | 生成膝/胸触地、左右脚支撑、同步起落、错位和碎步证据 |
| 完整候选规则 | 各动作分析器中的必需规则列表 | 将逐规则 `PASS/FAIL/UNSURE` 聚合成初步 `VALID/NO_REP/UNSURE` |
| 可观测性 | `observability.yaml` | 检查整次可见度、必需关键点、决定性规则、地板、视角和单帧失败，必要时降级为 `UNSURE` |
| 网页当前姿态评价 | `webui/analysis.py` 中的 `ACTION_STANDARDS` | 控制当前阶段的红/绿/中性骨架和角度标记；它用于训练解释，不替代完整候选规则 |

因此，修改某个页面参考角度不一定会改变计数，修改技术提示阈值也不一定会改变 `VALID/NO_REP`。要调整完整动作判定，必须确认改动属于阶段识别、技术反馈、计数必需规则还是可观测性门控，并使用对应测试验证。主流程和结果解释见 [项目 README](../../README.md#动作问题如何判断)。

## 默认中灵敏度计数与分析周期端点

下列阈值是当前代码和默认 YAML 的计数口径，单位为角度时均为度；归一化距离按画面或人体尺度计算。它们是二维姿态代理，不是正式比赛裁判标准。过渡相位可以因短暂丢帧而跳过，表中的关键端点不可跳过。

| 动作 | 必需端点顺序 | 默认端点阈值与计数位置 |
|---|---|---|
| Lunge | `stand → trailing-knee contact → post-contact full extension` | 综合前进方向、脚位置和双膝离地高度确定后腿，清晰的近地膝可纠正侧视图左右误配。正面证据要求双膝/双髋 ≥165°；推荐侧面视角使用置信度更高的同侧腿链和 `side_extension_tolerance_deg: 3`，连续保持中灵敏度 2 帧。还必须交替触地且无额外调整步。 |
| Wall Ball | `tall start → hip below knee → upward extension → bilateral throw proxy` | 起始双膝、双髋均需 ≥165°，躯干偏离竖直 ≤25°；最低点按局部地板距离要求髋比膝低至少人体高度的 0.01。投掷端点双膝、双髋均需 ≥165°，双腕从胸部附近上升且均高于肩；双腕峰值时间差采用 `120/220 ms` 三档。四项规则全 PASS 后加 1。 |
| Rowing | `catch → finish` | `catch`：至少一侧膝 ≤105；`finish`：双膝 ≥145 且双肘平均 ≤145；`finish` 时增加一个分析周期。`drive` 可选，不产生官方有效次数。 |
| SkiErg | `top → bottom → top` | `top`：双手腕均高于肩 ≥0.03 且躯干绝对角 <15；`bottom`：手腕低于胸部 ≥0.05，并且躯干绝对角 ≥15 或膝角 <155；返回 `top` 时增加一个分析周期。`pull_down/return` 可选，不产生官方有效次数。 |
| Burpee Broad Jump | `chest contact confirmed → simultaneous takeoff → simultaneous landing → next hands-down validation` | 胸部必须由通用接触器确认；双脚起落同步及起落错位代理均需通过。身体中心位移/腿长需 ≥0.20，左右脚位移/腿长均需 ≥0.15 且方向一致。落地后进入 `AWAITING_NEXT_HANDS`，继续排查补步或碎步，到下一次 hands-down/chest-down 才完成八项规则验证并决定是否加 1。 |
| Sled Push | `drive → step` | `drive`：躯干明显前倾且身体中心变化 ≥0.003 或膝伸展变化 ≥3；`step`：脚踝位置或脚间距变化 ≥0.04；`step` 时增加一个推动步分析周期，不产生官方有效次数。 |
| Sled Pull | `reach → pull → recover → reach` | `reach`：双肘平均 ≥145；肘角减小形成 `pull`，随后肘角增大形成 `recover`；身体向前回正并再次到达 `reach` 时增加一个分析周期，使后拉和随后的回正属于同一周期。清晰拉幅建议 ≥25，不足主要触发质量提示，不产生官方有效次数。 |
| Farmers Carry | 连续状态，无重复端点 | 双手在髋部附近或以下、身体站立且检测到水平位移、步态或膝角变化时进入 `carrying`；`cycle_count` 与兼容 `rep_count` 均保持 0。静止约 1200 ms 后进入 `rest`。 |

计数门槛与质量门槛分离：多数技术质量问题只作为反馈保留；动作专属的必需规则仍会决定是否计数，例如 Burpee Broad Jump 的胸部触地与最低前向位移，以及 Wall Ball 的站直起始、髋低于膝、完全伸展和双手投掷代理均必须通过。

## 距离动作的人体违规配置

Rowing、SkiErg、Sled Push 和 Sled Pull 的阶段计数只用于动作分析，统一输出
`cycle_count`、`count_semantics: analysis_cycle` 和
`official_rep_count_supported: false`。为兼容统一结果卡片和既有消费者，通用
`candidate_count`、`pose_valid_rep_count`、`rep_count` 也可能随完整分析周期增加；
这些字段在距离动作中不能解释为官方有效次数。Farmers Carry 使用
`count_semantics: continuous_monitor`，不增加周期或有效次数。

- Rowing 的 `ROWING_EARLY_STAND_PROXY` 仅在用户开始至停止分析的训练区间内工作。默认要求双膝 `≥160°`、双髋 `≥155°`、躯干偏离竖直 `≤30°`，并且髋部相对坐姿基线抬升至少人体高度的 `0.18`，持续 `≥300 ms`。正面等无法可靠观察站起的已知视角输出 `UNSURE`。
- SkiErg 不识别底座，不能从脚离地判断是否违规；只保留分析周期和技术反馈，不新增官方人体违规代码。
- Sled Push 没有固定合法关节角度或躯干姿势；只保留 `drive/step` 分析周期和技术提示，不建立 `pose_valid_rep` 或姿态违规。
- Sled Pull 的跪姿违规只在 `pull` 阶段、通用膝盖接触器确认接触并持续 `≥150 ms` 后输出 `SLED_PULL_KNEELING_VIOLATION`。坐姿代理默认要求髋部下降至少人体高度的 `0.18`、膝角 `≤130°`、躯干前倾 `≤30°`、髋部垂直速度不超过每秒人体高度的 `0.05`、膝盖明确未触地并持续 `≥250 ms`；明确证据输出 `SLED_PULL_SEATED_VIOLATION`，地板或接触证据不足时输出 `UNSURE_POSSIBLE_SEATED_PULL`。
- Farmers Carry 在检测到搬运移动时检查双臂。任一肘角 `<155°` 持续 `≥300 ms` 输出 `ARM_NOT_EXTENDED_VIOLATION`；任一手腕低于同侧髋部不足人体高度的 `0.03`，或手腕相对同侧髋的横向距离超过肩宽的 `0.80`，持续 `≥300 ms` 输出 `ARM_NOT_BY_SIDE_VIOLATION`。

上述时间字段是持续违规门控，不是官方规则给出的容差。单帧异常只进入候选状态；
关键点、地板或视角证据不足时为 `UNSURE`，不会直接输出明确违规。早拉手、划船节奏、
躯干角度和左右不对称等既有技术提示仍是动作质量反馈，不会自动升级为比赛违规。

## 通用字段

- `action_name`：配置所属动作，便于检查和追踪。
- `visibility_min`：最低关键点可见度；提高后判断更保守。
- `stable_frames`：视频阶段切换默认需连续满足 2 帧；实时摄像头的中/高灵敏度使用 1 个已处理帧，由完整端点序列负责防抖，避免推理丢帧时错过快速顶点。
- 计数器会在短暂关键点丢失时保留当前动作进度，并允许跳过下降、回程等过渡相位；最低点、伸展点等关键端点仍不可跳过。
- `*_cooldown_ms` / `cooldown_ms`：两次计数或事件之间的最短间隔。
- `feedback_limits.max_messages`：单帧最多提示数，默认 2。
- `feedback_limits.low_visibility_exclusive`：可见度不足时只显示取景提示，避免输出不可靠的技术结论。

动作专属字段可按需要微调：角度字段通常以度为单位，`*_margin`、`*_delta_x`、
`*_distance_norm` 是画面或人体尺度归一化值，时间字段以毫秒为单位。建议一次只调整一个
阈值，并用 `--save-debug-csv` 对照阶段和特征变化；不要把不同机位的经验阈值直接混用。

`contact.yaml` 记录通用触地检测器的初始阈值。膝盖使用关节中心减去小腿长度
`0.10` 倍得到虚拟表面；胸部使用双肩和双髋构造代理表面。两者都依赖局部地板参考、
动作阶段、速度、持续时间和关键点置信度，并使用进入/退出双阈值防抖。MediaPipe
分割可提高胸部代理结论置信度；YOLO Pose 没有分割时仍会计算，但置信度最高为
`0.74`。胸部结果是二维视觉代理，不代表精确乳头线触地。

Lunge 使用动作专属的膝盖代理配置：`knee_surface_radius_shank_ratio: 0.25`，
进入/退出地板带分别为人体高度的 `0.060/0.090`，且仍需连续底部证据、运动速度、
地板与关键点置信度共同确认；该放宽不会改变其他动作使用的通用接触器阈值。

`foot_events.yaml` 记录左右脚独立状态机的阈值。支撑判断同时检查脚跟和脚尖，避免
脚尖仍着地时因踝或脚跟抬高误报起跳。双脚起落同步的 `100/180 ms` 是视频系统容差，
不是官方规则给出的时间值。`FOOT_STAGGER_PROXY` 使用本人平均脚长归一化前后错位，
不能解释为精确测得官方 5 厘米。独立步事件还要求至少 `80 ms` 腾空、`80 ms`
稳定支撑和腿长 `0.07` 倍的水平位移，以过滤关键点抖动。

`observability.yaml` 在动作规则聚合后统一检查证据质量。
`required_landmark_confidence: 0.60` 是各必需关键点在决定性证据窗口中的中位置信度下限（再取最弱关键点），
`rep_mean_confidence: 0.65` 是整次候选的平均可见度下限，
`decisive_rule_confidence: 0.72` 是输出最终 `VALID` 或 `NO_REP` 所需的决定性规则
置信度。局部地板失效、已知视角不适合或只有一个异常失败帧也会将最终结论降为
`UNSURE`。这层门控不改写逐规则 `PASS/FAIL`，降级详情输出在
`last_rep_observability`。

## 调试输出

桌面版使用 `--hyrox-debug` 时会读取动作状态中的 `debug.floor_reference`、
`debug.contacts`、`debug.foot_events` 和 `last_rep_decision`。画面绘制局部地板线、
虚拟膝盖表面点 `K`、虚拟胸部表面点 `C`，并显示：

- 膝盖/胸部代理的接触状态和归一化离地高度；
- 左右脚支撑、起跳候选、腾空、落地候选或不可观测状态；
- 双脚起跳和落地时间差、`FOOT_STAGGER_PROXY` 状态及比例；
- 本次候选逐规则 `PASS/FAIL/UNSURE`、置信度、值和最终结果。

虚拟表面点只用于解释二维接触代理，不表示真实接触面积或厘米距离。视频回放工具可用
`--save-debug-csv` 保存逐帧阶段和特征；规则对象及完整结果仍以 JSON/动作状态为准。

Burpee Broad Jump 的 `hand_placement_pass_foot_length_ratio: 1.25` 和
`hand_placement_unsure_foot_length_ratio: 1.45` 控制 `LEGAL_HAND_PLACEMENT_PROXY`；
它按双手相对前脚尖的最远前向距离除以本人平均脚长计算，不能解释为精确测得官方
30 厘米。`forward_jump_min_com_displacement_leg_ratio: 0.20` 要求身体中心明显前移，
`forward_jump_min_both_feet_displacement_leg_ratio: 0.15` 要求左右脚都沿同一方向前移。
缺少可靠地板、胸部、手腕或双脚证据时，相关规则会降级为 `UNSURE`，不会静默计入
有效次数。

Wall Ball 的阶段识别阈值和有效计数阈值相互独立。`stand_knee_angle_min`、
`stand_hip_angle_min`、`throw_knee_angle_min` 和 `throw_hip_angle_min` 用于识别宽松阶段；
最终规则使用 `tall_start_knee_angle_min: 165`、`tall_start_hip_angle_min: 165`、
`tall_start_trunk_from_vertical_max_deg: 25` 及投掷端点双髋双膝 `≥165°`。
`hip_below_knee_margin: 0.01` 是基于局部地板的人体高度归一化差值。双腕投掷代理的
`wrist_peak_time_diff_ms_pass: 120` 和 `wrist_peak_time_diff_ms_unsure: 220`
是视频工程容差；`throw_wrist_rise_body_ratio_min: 0.12`、
`throw_wrist_chest_band_body_ratio: 0.25` 与
`throw_wrist_midline_body_ratio_max: 0.60` 分别约束上升幅度、胸前起点和身体中线范围。
`BILATERAL_THROW_PROXY` 不检测球或目标，不能证明命中。

Lunge 的 `full_extension_knee_angle_min`、`full_extension_hip_angle_min` 和
`full_extension_hold_frames_high/medium/low` 只用于已确认后膝触地之后的伸展。
动作开始前的站姿不会通过该规则。侧面视角由 `side_extension_tolerance_deg` 控制
同侧可见腿链的二维角度容差。Lunge 的有效计数还依赖可靠局部地板与脚部事件；
缺少这些证据时完整阶段序列会保留为 `UNSURE`，不会静默计入有效次数。

示例：

```powershell
python main.py --camera 0 --mirror --hyrox-action rowing --hyrox-config configs/hyrox/rowing.yaml --hyrox-debug
python tools/replay_hyrox_video.py --video "HYROX视频\划船机.mp4" --hyrox-action rowing --camera-view side --save-debug-csv outputs/rowing_debug.csv
```

## 单摄像头无法可靠判断的项目

这些分析器只提供二维姿态近似指导，不检测器械，也不冒充比赛裁判系统。当前无法可靠
判断器械重量、阻力档位、绳子或雪橇是否过线、lane 合规性、划船机或 SkiErg 屏幕里程，
也不能确认 Farmers Carry 200 m、Burpee Broad Jump 80 m、雪橇 50 m 等官方距离。
二维画面还不能精确测量离地高度、落点距离、胸部真实触地、双脚厘米级误差或关节受力。
