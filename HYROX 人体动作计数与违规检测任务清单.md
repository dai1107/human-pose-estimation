第1轮：建立人体规则验证层

完成状态（2026-07-17）：✅ 已完成

- 已新增 `RepCandidate`、`BodyRuleResult`、`RepDecision` 和统一聚合器；
- 所有动作分析器不再直接执行 `rep_count += 1`，完整周期先登记候选并执行必需规则；
- 已统一输出 `candidate_count`、`pose_valid_rep_count`、`no_rep_count`、`unsure_count`，其中旧字段 `rep_count` 是 `pose_valid_rep_count` 的兼容别名；
- 必需规则 `FAIL`、`UNSURE/NOT_APPLICABLE`、全部 `PASS` 分别聚合为 `NO_REP`、`UNSURE`、`VALID`，未返回已声明必需规则时按 `UNSURE` 处理；
- `required_for_count=false` 的技术质量结果不会阻止计数；
- 当前第一轮仅把已有完整人体动作序列注册为 `body_sequence_valid`。各动作的触地、同步起落、交替腿等专属必需规则按后续轮次接入，不在本轮伪判。

任务1：候选动作和有效动作分离

不要再在状态机到达终点时直接：

rep_count += 1

改成：

状态机完成一个动作周期
    ↓
生成 RepCandidate
    ↓
执行本动作 required_rules
    ↓
VALID / NO_REP / UNSURE
    ↓
只有 VALID 增加 pose_valid_rep_count

建议数据结构：

@dataclass
class RepCandidate:
    action: str
    start_frame: int
    end_frame: int
    phases_seen: set[str]
    events: dict[str, object]
    frames: list["PoseFrame"]


@dataclass
class BodyRuleResult:
    rule_id: str
    status: Literal["PASS", "FAIL", "UNSURE", "NOT_APPLICABLE"]
    confidence: float
    value: float | bool | None
    reason_code: str | None
    evidence_frames: list[int]


@dataclass
class RepDecision:
    status: Literal["VALID", "NO_REP", "UNSURE"]
    rules: list[BodyRuleResult]
    reason_codes: list[str]
    confidence: float

聚合规则：

任一必需规则 FAIL
    → NO_REP

没有 FAIL，但存在必需规则 UNSURE
    → UNSURE

全部必需规则 PASS
    → VALID
任务2：保留三个计数
candidate_count
pose_valid_rep_count
no_rep_count
unsure_count

界面显示：

动作周期：12
有效动作：9
未完成：2
无法确认：1
任务3：不要把技术提示作为计数门槛

例如：

膝盖轻微内扣
躯干略微前倾
左右不完全对称

这些属于技术质量，不应阻止计数。

只把官方动作端点或明确动作限制作为 required_for_count=true：

膝盖是否触地
是否完全伸展
髋是否低于膝
胸部是否触地
双脚是否同步起落
是否交替腿
是否多走一步
是否坐下或跪地
第2轮：局部地面参考

完成状态（2026-07-17）：✅ 已完成

- 已新增有状态的 `LocalFloorReference`，仅在站直、足部稳定且全身完整入镜时采集最近约 0.5 秒的脚跟/脚尖最低点，并用稳健中位数建立自动水平地板线；
- 已支持网页摄像头预览中的可选两点点击标定，倾斜画面使用两点直线，不进行场地识别、四点透视或真实距离标定；
- 已实现 `signed_distance_to_floor()` 与 `normalized_height_to_floor()`，统一规定人体位于地板上方时距离为正；
- 身体尺度按“稳定站立标定 → 当前人体高度 → 躯干与腿段估计”降级选择，并输出尺度来源；
- 地板线、身体尺度、髋/膝/脚部离地高度、来源、置信度及原因代码均接入动作特征和调试状态；
- 足部长期不可见、全身未完整入镜、相机明显移动或手动线与脚部严重矛盾时返回 `UNSURE`，不会输出确定触地结论；
- 桌面 `--hyrox-debug` 可显示地板状态、置信度和地板线；网页覆盖层会显示自动或手动地板线。

你不需要检测赛道或场地，但触地判断仍然需要一个局部地板参考。

这不是场地识别，只用于判断身体点距离脚下地面多远。

任务1：自动估计局部地板线

在用户站立且双脚稳定时，采集：

left_heel
right_heel
left_foot_index
right_foot_index

取最近约0.5秒最低点的稳健中位数，建立水平地板线：

floor_y = median(stable_foot_y_samples)

如果摄像机存在明显倾斜，提供可选的两点手动地板线：

用户只需点击脚下地板的两个点

不需要四点透视标定，不需要现实距离标定。

任务2：统一离地高度

实现：

signed_distance_to_floor(point)
normalized_height_to_floor(point, body_height)

推荐人体尺度优先级：

用户站立标定身高
→ 当前稳定人体框高度
→ 肩髋长度 + 双腿长度估算

禁止在动作代码里直接写：

if knee_y > 680:
任务3：地板参考置信度

以下情况返回 UNSURE：

脚踝和足部关键点长期不可见
人体未完整入镜
摄像机明显移动
自动地板线与脚部位置严重矛盾
第3轮：通用触地检测器

完成状态（2026-07-17）：✅ 已完成

- 已新增统一 `ContactResult`、`KneeContactDetector`、`ChestContactDetector` 与 `ContactDetectorSuite`，输出 `CONTACT / NO_CONTACT / UNSURE / NOT_OBSERVABLE`、置信度、虚拟表面离地比例、持续时间及证据帧；
- 膝盖检测使用“膝关节中心离地高度 − 0.10 × 小腿长度”的虚拟表面，并融合动作阶段、垂直速度、持续帧数和关键点置信度；
- 膝盖和胸部均采用进入/退出双阈值滞回；单帧跳点不会确认接触，下降—局部最低—回升序列可补偿实时推理跳过最低帧，但最低帧本身仍必须进入接触范围；
- 胸部检测明确命名为 `chest_proxy`，由双肩/双髋构造虚拟胸部下表面，并同时约束肩、髋离地高度和躯干相对地板角度，不声称测得真实乳头线；
- MediaPipe 后端已默认请求可选人体分割，胸部按 40% 表面距离、25% 地板带分割重叠、15% 躯干平行、10% 低垂直速度、10% 持续时间融合；无分割的 YOLO 或缺失分割帧仍可运行，但置信度上限为 `0.74`；
- 分割掩码只存在于当前帧私有特征，不进入候选证据缓冲、网页 JSON 或报告历史；网页/离线状态只保留轻量接触结果；
- 初始阈值已写入 `configs/hyrox/contact.yaml`，专项测试覆盖明确接触、接近未触地、胸部悬空、单帧跳点、遮挡/丢点、滞回、局部最低点以及有/无分割。

任务1：实现统一接触状态
@dataclass
class ContactResult:
    status: Literal[
        "CONTACT",
        "NO_CONTACT",
        "UNSURE",
        "NOT_OBSERVABLE",
    ]
    confidence: float
    surface_height_ratio: float | None
    hold_ms: int
    evidence_frames: list[int]

接触不能只靠单帧距离，需要融合：

虚拟身体表面到地面的距离
当前动作阶段
垂直运动速度
局部最低点
持续时间
关键点置信度
任务2：膝盖虚拟表面

膝关节点是关节中心，不能要求它直接落到地板线上。

shank_length = distance(knee, ankle)
knee_surface_radius = 0.10 * shank_length

knee_surface_height = (
    knee_joint_height - knee_surface_radius
)

初始配置：

knee_contact:
  surface_radius_shank_ratio: 0.10

  enter_height_body_ratio: 0.015
  exit_height_body_ratio: 0.035

  max_vertical_speed_body_per_second: 0.15

  min_hold_frames:
    high: 2
    medium: 3
    low: 4

  min_landmark_confidence: 0.60
  confirm_confidence: 0.72

使用进入、退出两个阈值形成滞回：

低于 enter → 接触候选
高于 exit → 明确离地
处于两者之间 → 保持原状态
任务3：胸部虚拟表面

MediaPipe 没有乳头线或胸骨关键点，需要构造代理点：

shoulder_mid = midpoint(left_shoulder, right_shoulder)
hip_mid = midpoint(left_hip, right_hip)

chest_center = (
    0.65 * shoulder_mid
    + 0.35 * hip_mid
)

torso_length = distance(shoulder_mid, hip_mid)
chest_half_thickness = 0.20 * torso_length

沿地板法向估算胸部下表面。

初始配置：

chest_contact:
  shoulder_weight: 0.65
  hip_weight: 0.35
  surface_offset_torso_ratio: 0.20

  enter_height_body_ratio: 0.020
  exit_height_body_ratio: 0.045

  shoulder_height_body_ratio_max: 0.080
  hip_height_body_ratio_max: 0.160
  torso_to_floor_angle_deg_max: 25

  min_hold_frames:
    high: 2
    medium: 3
    low: 4

官方 Burpee Broad Jump 要求乳头线明确接触地面，所以只使用肩髋角度不足以完成判断，应当把这个检测器称为胸部触地代理。

任务4：接入人体分割蒙版

MediaPipe 后端开启可选人体分割。

在胸部 ROI 内检查人体轮廓是否接触局部地板带：

segmentation_contact:
  enabled: true
  floor_band_body_ratio: 0.012
  minimum_overlap_ratio: 0.08

胸部接触融合建议：

40% 虚拟胸部表面距离
25% 分割区域与地板带重叠
15% 躯干接近平行地面
10% 垂直速度接近零
10% 持续时间

没有分割结果时可以继续运行，但接触置信度上限设为：

0.74

这样临界情况会得到 UNSURE，不会被误判为确定触地。

任务5：局部最低点补偿

实时推理可能跳过真正触地帧。

允许以下序列确认接触候选：

前一帧：向下运动
当前帧：离地距离局部最小
后一帧：向上运动

但当前帧距离仍必须进入接触范围，不能仅靠速度方向变化判断。

第4轮：脚部事件检测器

它将用于 Burpee Broad Jump 和 Lunge。

完成状态（2026-07-17）：✅ 已完成

- 已新增 `FootEventDetectorSuite`，左右脚分别维护 `GROUNDED → TAKEOFF_CANDIDATE → AIRBORNE → LANDING_CANDIDATE → GROUNDED`，候选帧必须连续满足条件；
- 支撑高度同时使用 heel 与 foot_index；只抬脚跟、脚尖仍着地时保持 `GROUNDED`，仅当脚跟和脚尖都明确离地才进入起跳候选；
- 起跳和落地事件保留候选首次出现的原始时间，统一输出 `left/right_takeoff_ms` 与 `left/right_landing_ms`；
- 双脚同步按 `≤100 ms → PASS`、`100～180 ms → UNSURE`、`>180 ms → FAIL` 评估，并明确标注为视频系统工程容差，而非官方毫秒标准；
- 已实现本人脚长归一化的 `FOOT_STAGGER_PROXY`，沿局部地板方向比较左右脚尖前后差；结果不会声称精确测得官方 5 厘米；
- 已实现独立 `STEP` 事件，只有单脚完成有效腾空、稳定落地且水平位移达到腿长 `0.07` 倍时才记录；短暂腾空、落地未稳定和原地关键点抖动均被过滤；
- 阈值已写入 `configs/hyrox/foot_events.yaml`，分析器调试状态、网页实时 JSON 与报告历史只保存轻量脚部状态和事件，不保存图像数据；
- 专项测试覆盖左右脚完整状态序列、脚尖支撑、连续帧、三档同步窗口、落地同步、三档错位代理、有效步、短腾空、短支撑、抖动及关键点缺失。

任务1：检测脚部支撑状态

左右脚分别维护：

GROUNDED
TAKEOFF_CANDIDATE
AIRBORNE
LANDING_CANDIDATE
GROUNDED

判断依据：

heel 和 foot_index 的离地高度
垂直速度
连续帧

不要只使用 ankle，因为脚尖可能仍在地面而踝关节已经升高。

任务2：双脚同步起跳和落地

记录：

left_takeoff_ms
right_takeoff_ms
left_landing_ms
right_landing_ms

工程初值：

foot_sync:
  pass_ms: 100
  unsure_ms: 180

判断：

时间差 <= 100 ms → PASS
100～180 ms → UNSURE
>180 ms → FAIL

“同时”是官方要求，但规则没有给出毫秒阈值，因此这里属于视频系统容差。官方还要求起跳和落地时两脚前后差不超过5厘米。

任务3：两脚前后错位代理

你不希望做现实地面标定，因此不能精确测量5厘米。

改用本人脚长归一化：

stagger_ratio = (
    abs(left_toe_forward - right_toe_forward)
    / mean_foot_length
)

初始配置：

foot_stagger:
  pass_foot_length_ratio: 0.20
  unsure_foot_length_ratio: 0.30

约等于：

不超过本人脚长20% → PASS
20%～30% → UNSURE
超过30% → FAIL

结果名称必须是：

FOOT_STAGGER_PROXY

不能声称精确测得官方5厘米。

任务4：额外步和碎步

建立独立足部落地事件：

左脚单独离地并在前方落地
右脚单独离地并在前方落地
脚部连续发生多次小幅支撑切换

过滤原地关键点抖动：

step_event:
  min_horizontal_displacement_leg_ratio: 0.07
  min_airborne_ms: 80
  min_grounded_ms: 80
第5轮：Lunge 有效计数

完成状态（2026-07-17）：✅ 已完成

- Lunge 候选现统一执行 `trailing_knee_contact`、`full_knee_extension`、`full_hip_extension`、`alternating_contact_leg`、`no_extra_step_or_shuffle` 五项必需规则；只有全部 `PASS` 才增加 `pose_valid_rep_count`；
- 前后腿优先根据本次人体中心前进方向和沿地板方向的脚尖位置确定，不直接把 left/right 固定解释为前后腿；方向不足时使用虚拟膝表面更接近地面的一侧回退，并把腿识别置信度限制在 `0.65`；
- 通用 `KneeContactDetector` 已支持指定 left/right 单侧调用；后腿 `CONTACT / NO_CONTACT / UNSURE/NOT_OBSERVABLE` 分别映射为 `PASS / FAIL / UNSURE`；
- 完全伸展只累计本次已确认膝触地之后的帧；双膝和双髋分别要求达到配置角度并连续保持 high/medium/low 对应的 `1/2/3` 帧，动作开始前的站姿不会复用；
- `previous_valid_contact_leg` 只在最终决策为 `VALID` 时更新；同一后腿连续完成会触发 `SAME_CONTACT_LEG_REPEATED`，失败或不确定候选不会污染下一次交替判断；
- 额外步规则复用第四轮 `STEP` 事件：允许不调整直接下降，也允许一次由当前前腿完成的直接跨步；错误侧单步、多个支撑切换或碎步会得到 `EXTRA_STEP_OR_SHUFFLE`；
- 调试状态新增 `STANDING → DESCENDING → KNEE_CONTACT_CONFIRMED → ASCENDING → FULL_EXTENSION_CONFIRMED → RULE_VALIDATION` 验证状态、单侧膝接触证据、前后腿来源和五项规则结果；
- 缺少可靠地板、足部/膝部被遮挡或伸展角不可见时不再沿用旧式阶段计数，而是生成 `UNSURE` 候选。

官方人体规则为：

后侧膝盖明确触地；
每次结束时站直，髋和膝完全伸展；
两腿交替；
重复之间不能额外迈步或碎步；
可以连续弓步，也可以双脚平行停顿。
必需规则
lunge:
  required_rules:
    - trailing_knee_contact
    - full_knee_extension
    - full_hip_extension
    - alternating_contact_leg
    - no_extra_step_or_shuffle
任务1：确定前腿和后腿

不要直接根据 left/right 判断。

侧面或斜侧面下，根据动作前进方向判断：

前方脚 = leading_foot
后方脚 = trailing_foot
后方脚对应膝 = trailing_knee

若前进方向不明确：

使用触地更接近地面的膝作为候选 trailing_knee
但降低 confidence
任务2：后膝接触

调用通用 detect_knee_contact()。

CONTACT → PASS
NO_CONTACT → FAIL
UNSURE → UNSURE
任务3：完全伸展

复用已有角度规则，建议加入持续时间：

full_extension:
  knee_angle_min_deg: 165
  hip_angle_min_deg: 165
  min_hold_frames:
    high: 1
    medium: 2
    low: 3

必须在本次触地之后出现完全伸展，不能用动作开始前的站立帧通过。

任务4：交替腿
current_contact_leg != previous_valid_contact_leg

只有 VALID 动作更新上一条腿：

if decision.status == "VALID":
    previous_valid_contact_leg = current_contact_leg
任务5：重复之间不得额外迈步

允许两种情况：

触地 → 站直 → 另一条腿直接下降
触地 → 站直 → 双脚平行短暂停顿 → 另一条腿下降

不允许：

站直后单独迈出调整步
碎步后才开始下一次下降
完整状态
STANDING
→ DESCENDING
→ KNEE_CONTACT_CONFIRMED
→ ASCENDING
→ FULL_EXTENSION_CONFIRMED
→ RULE_VALIDATION
第6轮：Burpee Broad Jump 有效计数

完成状态（2026-07-17）：✅ 已完成

- 已接入八项必需规则：胸部触地、双脚同步起跳、双脚同步落地、起跳错位代理、落地错位代理、无额外步或碎步、手部位置代理和向前跳跃；只有全部 `PASS` 才计入 `VALID`；
- 胸部触地复用通用胸部接触检测器，仅 `CHEST_CONTACT_CONFIRMED` 可通过；候选但未确认、地板或关键点证据不足会按规则输出 `FAIL` 或 `UNSURE`；
- 双脚起落复用左右脚事件，按 `≤100 ms → PASS`、`100～180 ms → UNSURE`、`>180 ms → FAIL` 验证同步性；
- 起跳错位取起跳前最后一个稳定双脚支撑帧，落地错位取落地后第一个稳定双脚支撑帧，均输出脚长归一化的 `FOOT_STAGGER_PROXY`，不声称厘米级测量；
- 双手开始触地时计算脚长归一化位置并固定输出 `LEGAL_HAND_PLACEMENT_PROXY`，阈值为 `1.25/1.45`；
- 落地后先进入 `AWAITING_NEXT_HANDS`，持续检查额外 `STEP`；到下一次 hands-down/chest-down 边界才执行最终规则验证，避免先加有效次数后无法撤销；
- 向前跳要求身体中心位移/腿长 `≥0.20`，左右脚位移/腿长均 `≥0.15`，三者方向一致，后续有效重复还必须沿已建立的前进方向；
- 专项测试覆盖八项规则全通过、胸部未确认、同步三档、错位、手部位置、落地后补步、原地或反向跳、缺失证据降级，以及正常宽跳落地事件不误报碎步。

官方人体规则包括：

胸部乳头线接触地面；
双脚同时起跳和落地；
起跳、落地时两脚错位不得超过5厘米；
不得有额外步或碎步；
起身时允许使用膝盖；
后续波比双手不能放在脚尖前方超过30厘米。
必需规则
burpee_broad_jump:
  required_rules:
    - chest_ground_contact
    - simultaneous_takeoff
    - simultaneous_landing
    - takeoff_stagger_proxy
    - landing_stagger_proxy
    - no_extra_step_or_shuffle
    - legal_hand_placement_proxy
    - forward_jump_detected
任务1：胸部触地

调用通用胸部接触检测器。

状态应从当前 README 的：

胸部接近地面

改为：

CHEST_CONTACT_CANDIDATE
→ CHEST_CONTACT_CONFIRMED

只有 CONFIRMED 才能进入有效候选。

任务2：双脚同步

复用脚部事件检测器：

TAKEOFF_LEFT 与 TAKEOFF_RIGHT
LANDING_LEFT 与 LANDING_RIGHT
任务3：脚部前后差代理

分别在：

起跳前最后一个双脚支撑帧
落地后第一个稳定双脚支撑帧

计算 foot_stagger_ratio。

任务4：手部位置代理

官方30厘米要求无法在无尺度标定下精确测量，改用脚长归一化。

在双手落地时：

hand_to_toe_forward_ratio = (
    max_forward_wrist_distance_from_front_toe
    / mean_foot_length
)

初始阈值：

hand_placement:
  pass_foot_length_ratio: 1.25
  unsure_foot_length_ratio: 1.45

输出：

LEGAL_HAND_PLACEMENT_PROXY
任务5：额外步和碎步

检查范围：

本次宽跳落地稳定
→ 下一次双手开始触地

允许：

落地后直接进入下一次波比

不允许：

落地后单脚补一步
左右脚碎步调整
先走动再趴下
任务6：必须存在向前跳跃

防止原地完成一次波比也被计数：

forward_jump:
  min_com_displacement_leg_ratio: 0.20
  min_both_feet_displacement_leg_ratio: 0.15

这只是确认发生了宽跳，不判断跳了多远。

第7轮：Wall Ball 人体有效计数

完成状态（2026-07-17）：✅ 已完成

- 已将 Wall Ball 从“阶段完成即计数”接入统一人体规则验证层，四项必需规则为 `tall_start`、`hip_below_knee`、`upward_extension` 和 `bilateral_throw_proxy`；只有全部 `PASS` 才增加 `pose_valid_rep`；
- 起始站姿独立使用双膝 `≥165°`、双髋 `≥165°`、躯干偏离竖直 `≤25°` 验证；下降阶段不会因防抖仍显示 `stand` 而覆盖已确认的起始证据；
- 下蹲深度通过统一局部地板距离比较髋中点与膝中点，要求 `(knee_height - hip_height) / body_height ≥0.01`，不再只用膝角或直接比较图像 y 坐标；
- 向上伸展要求投掷端点双膝、双髋均达到 `≥165°`；伸展不足会形成完整候选但记为 `NO_REP`；
- `BILATERAL_THROW_PROXY` 要求双腕从胸部附近同步上升、保持在身体中线合理范围并均到达肩部以上；双腕峰值时间差按 `≤120 ms → PASS`、`120～220 ms → UNSURE`、`>220 ms → FAIL` 处理；
- 调试状态已输出 `TALL_START → DESCENDING → HIP_BELOW_KNEE_CONFIRMED → ASCENDING → BILATERAL_THROW_CONFIRMED → POSE_VALID_REP/RULE_VALIDATION`，逐规则结果自动进入现有实时 JSON 与报告；
- 缺少可靠局部地板、身体尺度、任一手腕或投掷端点证据时降级为 `UNSURE`，不会静默计入有效次数；
- 专项测试覆盖四规则全通过、未站直或缺起始、髋未低于膝、缺地板、伸展不足、单手投掷、双腕峰值三档、单腕缺失、实时单帧投掷端点和短暂关键点丢失。

忽略球和目标后，只判断人体动作：

动作开始时站直，髋膝伸展；
完成完整下蹲；
最低点髋部低于膝部；
使用双手完成投掷动作代理；
球落地后的重置规则由于不检测球，暂不实现。

官方规则明确要求站直起始，最低点髋低于膝；完整官方计数还要求双手投球并命中目标。

输出名称

不要使用：

official_valid_rep

使用：

pose_valid_rep
body_sequence_valid
必需规则
wall_ball:
  required_rules:
    - tall_start
    - hip_below_knee
    - upward_extension
    - bilateral_throw_proxy
任务1：站直开始
tall_start:
  knee_angle_min_deg: 165
  hip_angle_min_deg: 165
  trunk_from_vertical_max_deg: 25
任务2：髋低于膝

不要只根据膝角度推断深度。

使用相对地板高度：

hip_height = floor_distance(hip_mid)
knee_height = floor_distance(knee_mid)

hip_below_knee = (
    hip_height
    < knee_height - 0.01 * body_height
)

图像坐标 y 轴向下，因此应通过统一地板距离函数比较，避免在动作模块里写反。

任务3：双手投掷代理

不检测球时，用左右手腕同步上升：

左右手腕均从胸部附近向上运动
两个手腕到身体中线距离合理
两手达到肩部以上
左右手腕峰值时间接近

初始配置：

bilateral_throw_proxy:
  wrist_peak_time_diff_ms_pass: 120
  wrist_peak_time_diff_ms_unsure: 220
  both_wrists_above_shoulders_required: true

这只能证明做出了双手投掷样式，不能证明球命中。

状态
TALL_START
→ DESCENDING
→ HIP_BELOW_KNEE_CONFIRMED
→ ASCENDING
→ BILATERAL_THROW_CONFIRMED
→ POSE_VALID_REP
第8轮：距离动作的人体违规检测

完成状态（2026-07-17）：✅ 已完成

- 已新增通用 `TemporalViolationTracker`，以 `CLEAR → CANDIDATE → ACTIVE` 持续时间门控过滤单帧异常；证据不足时输出 `UNSURE`，不会直接升级为明确违规；
- Rowing 已实现 `ROWING_EARLY_STAND_PROXY`：训练检测区间内双膝、双髋伸展，躯干接近竖直且髋部相对坐姿基线明显抬升持续 `≥300 ms` 时激活；已知正面等不合适视角只输出不确定状态；
- SkiErg 保留周期计数与技术反馈，不因脚离地判违规；Sled Push 保留 `drive/step` 与角度提示，不建立姿态有效次数或固定姿势违规；
- Sled Pull 已实现仅在 `pull` 阶段、膝盖接触确认后持续 `≥150 ms` 的 `SLED_PULL_KNEELING_VIOLATION`，以及明显稳定坐姿持续 `≥250 ms` 的 `SLED_PULL_SEATED_VIOLATION`；接触证据不足但坐姿几何成立时输出 `UNSURE_POSSIBLE_SEATED_PULL`；
- Farmers Carry 已实现搬运移动期间持续 `≥300 ms` 的 `ARM_NOT_EXTENDED_VIOLATION` 与 `ARM_NOT_BY_SIDE_VIOLATION`；缺少关键点时为 `UNSURE`，正常步态中的短暂波动不会立即判错；
- Rowing、SkiErg、Sled Push、Sled Pull 统一输出 `cycle_count`、`count_semantics: analysis_cycle` 和 `official_rep_count_supported: false`；Farmers Carry 使用 `count_semantics: continuous_monitor` 且 `cycle_count` 保持 0；
- 已新增距离动作专项测试，覆盖持续时间防抖、合适/不合适视角、跪姿接触、明确/不确定坐姿、双项手臂违规、无新增违规边界及分析周期语义。

这些动作不产生官方重复计数，只保留现有周期计数用于分析。README 中目前已经将 Rowing、SkiErg、Sled Push/Pull 作为周期或步态计数，Farmers Carry 不按次数拆分。

8.1 Rowing

在不检测机器和距离的前提下，只检测：

训练区间中是否明显站起

输出：

ROWING_EARLY_STAND_PROXY

判定清晰站起：

standing_violation:
  knee_angle_min_deg: 160
  hip_angle_min_deg: 155
  trunk_from_vertical_max_deg: 30
  hip_vertical_rise_body_ratio_min: 0.18
  min_hold_ms: 300

因为不知道1000米何时完成，检测区间由用户点击“开始/停止分析”确定。

不要把划船技术顺序：

腿—躯干—手臂

当作官方违规规则，它属于动作质量反馈。

8.2 SkiErg

不识别 SkiErg 底座时，无法可靠区分：

脚落在底座
脚落在地板

因此不增加官方人体违规判断。

保留：

cycle_count
technique_feedback

不要用脚离地本身判错，因为官方允许动态或跳跃动作，只要求落回底座。

8.3 Sled Push

官方没有要求某个固定关节角度或身体姿势才能推动。

因此：

保留 drive / step 周期
保留角度技术提示
不建立 pose_valid_rep
不因为躯干角度不同判违规
8.4 Sled Pull

只检测明确的人体违规：

跪地拉
坐着拉

官方明确要求始终保持站立，不允许坐姿或跪姿拉雪橇。

跪地检测
正在 PULL 阶段
+ 任一膝盖 CONTACT
+ 持续超过150 ms

输出：

SLED_PULL_KNEELING_VIOLATION
坐姿检测

只判定明显坐姿，避免把低重心站姿误判：

髋部高度显著降低
膝关节屈曲
躯干保持直立或后倾
髋部垂直速度接近零
没有膝盖触地
持续超过250 ms

置信度不足时输出：

UNSURE_POSSIBLE_SEATED_PULL
8.5 Farmers Carry

只检查人体可见规则：

双臂是否在身体两侧基本伸展

官方要求两个壶铃在移动时由身体两侧伸展的双臂携带。忽略壶铃后，只能检测手臂姿态代理。

farmers_carry_arm_position:
  elbow_angle_min_deg: 155
  wrist_below_hip_margin_body_ratio: 0.03
  wrist_lateral_from_hip_max_shoulder_width_ratio: 0.80
  min_violation_ms: 300

输出：

ARM_NOT_EXTENDED_VIOLATION
ARM_NOT_BY_SIDE_VIOLATION

不要因为正常步态中的短暂肘角波动立即判错。

第9轮：统一“不确定”处理

完成状态（2026-07-17）：✅ 已完成

- 已在统一候选聚合层新增 `ObservabilityPolicy`，所有人体规则候选在原始规则聚合后、更新计数前执行同一套可观测性门控；
- 默认阈值已写入 `configs/hyrox/observability.yaml`：必需关键点置信度 `0.60`、整次动作平均可见度 `0.65`、决定性规则置信度 `0.72`；
- 候选平均可见度不足、决定性证据帧中的必需关键点置信度不足、决定性规则置信度不足时，原始 `VALID/NO_REP` 统一降级为 `UNSURE`；
- Lunge、Burpee Broad Jump 和 Wall Ball 的决定性证据帧会统一检查局部地板状态；地板无效时不得输出最终 `NO_REP`；
- 用户已明确选择但不适合当前动作规则的相机视角会降级为 `UNSURE`；未选择视角仍保留提示，不凭空判定为错误视角；
- 若一个 `NO_REP` 仅由单个异常证据帧产生，最终结论降级为 `UNSURE`；脚步事件等内部已完成连续帧确认的证据会保留其多帧来源，持续、清晰的失败仍可输出 `NO_REP`；
- 原始逐规则结果不会被篡改，例如单帧规则仍可显示 `FAIL`；最终决策通过 `last_rep_observability` 记录降级原因、整次平均置信度、关键点最低置信度、决定性规则置信度、地板和视角状态；
- `candidate_count = pose_valid_rep_count + no_rep_count + unsure_count` 继续保持互斥；降级候选只增加 `unsure_count`，不会增加有效次数或 `no_rep_count`；
- 专项测试覆盖三项默认阈值、低整次置信度、低关键点置信度、低决定性置信度、地板失效、错误视角、单帧异常、多帧明确失败，以及 Lunge/Burpee/Wall Ball 的实际降级路径。

以下情况不能判 NO_REP：

触地部位被遮挡
脚或手腕离开画面
地板参考失效
相机视角不适合
关键点置信度不足
检测只出现一个异常帧
快速动作中关键端点丢失

必须输出：

UNSURE

例如：

无法确认后膝是否触地
无法确认胸部是否接触地面
双脚落地时间不清晰
拍摄角度无法判断髋是否低于膝

建议阈值：

observability:
  required_landmark_confidence: 0.60
  rep_mean_confidence: 0.65
  decisive_rule_confidence: 0.72
第10轮：调试显示

完成状态（2026-07-17）：✅ 已完成

- `--hyrox-debug` 已绘制自动或手动局部地板线，并按 `READY/UNSURE` 使用不同颜色；
- 通用膝盖和胸部接触检测结果现已输出虚拟表面点坐标、离地高度比例、接触状态、置信度、持续时间和证据帧；桌面调试画面分别以 `K`、`C` 标出虚拟膝盖与胸部表面点；
- 调试文本已显示左右脚 `GROUNDED/TAKEOFF_CANDIDATE/AIRBORNE/LANDING_CANDIDATE/NOT_OBSERVABLE` 状态、起跳/落地时间差及脚部错位状态和比例；
- 已接入 `last_rep_decision.rules`，逐项显示 `PASS/FAIL/UNSURE`、规则名称、置信度和值，并显示最终 `RESULT: VALID/NO_REP/UNSURE`；尚未完成候选时明确显示 `PENDING`；
- 调试模式会同时显示当前动作、候选序号、阶段和最多两条反馈，不再与普通动作信息面板重叠；
- 已新增专项测试验证虚拟表面点、接触高度、脚部状态、同步时间差、错位比例、规则列表、最终结果以及地板线/虚拟点实际绘制。

在 --hyrox-debug 中增加：

局部地板线
虚拟膝盖表面点
虚拟胸部表面点
胸部和膝盖离地高度
左右脚 grounded / airborne 状态
起跳和落地时间差
脚部错位比例
本次动作规则列表
最终 VALID / NO_REP / UNSURE

示例：

LUNGE CANDIDATE #8

PASS  REAR_KNEE_CONTACT       0.84
PASS  FULL_KNEE_EXTENSION     171°
PASS  FULL_HIP_EXTENSION      168°
PASS  ALTERNATING_LEG         LEFT
FAIL  NO_EXTRA_STEP           1 extra step

RESULT: NO_REP
第11轮：自动化测试

完成状态（2026-07-17）：✅ 已完成

- 通用接触检测测试已覆盖膝盖接近但未触地、明确膝触地、胸部悬空/触地、分割辅助、单帧跳点、关键触地帧丢失、进入/退出迟滞、地板失效及关键点遮挡；
- 通用脚部事件测试已覆盖左右脚独立支撑状态、同步/不同步起跳、同步/不同步落地、脚部错位比例、落地后有效补步、短腾空/短支撑过滤、抖动过滤及脚部遮挡；
- Lunge 回放级特征序列已覆盖正常交替、后膝未确认触地、未完全伸展、同腿连续触地、相反腿交替以及额外迈步；
- Burpee Broad Jump 已覆盖胸部未确认触地、单脚提前起跳、单脚提前落地、落地后补步、手部明显过远、原地波比无前向跳跃和完整八规则序列；
- Wall Ball 已覆盖未先站直、下蹲未达到髋低于膝、单手上举、双腕时序三档、伸展不足、地板/手腕不可观测和完整四规则序列；
- Sled Pull 已覆盖正常站立拉、单膝跪地拉、明确/不确定坐姿拉，以及低重心但仍保持站立的非违规边界；
- 新增 Lunge、Burpee Broad Jump、Wall Ball 相同特征流独立重复运行测试，逐项比较 `candidate_count`、`pose_valid_rep_count`、`no_rep_count`、`unsure_count` 和最终决策；
- 自动化测试持续验证 `candidate_count = pose_valid_rep_count + no_rep_count + unsure_count`，确保三种结果互斥且重复运行确定。

通用检测器测试
膝盖接近地板但没有接触
膝盖明确触地
胸部悬空
胸部触地
单帧关键点跳到地板
触地关键帧丢失
左右脚同步起跳
单脚提前起跳
落地后补一步
关键点被遮挡
动作回放测试
Lunge
正常交替
后膝未触地
未完全站直
连续使用同一条腿
重复之间额外迈步
Burpee Broad Jump
胸部未触地
单脚先起跳
单脚先落地
落地后补步
双手放得明显过远
只有波比没有向前跳
Wall Ball
未先站直
下蹲未达到髋低于膝
只有一只手上举
完整人体动作序列
Sled Pull
站立拉
单膝跪地拉
坐地拉
低重心但仍站立

同一视频重复运行时：

candidate_count 必须一致
VALID / NO_REP / UNSURE 数量必须一致
