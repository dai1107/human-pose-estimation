from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from statistics import median
from typing import Any


PHASE_LABELS = {
    "stand": "站立",
    "standing": "站立",
    "ready": "准备",
    "descent": "下降",
    "bottom": "最低点",
    "ascent": "起身",
    "squat_down": "下蹲",
    "drive": "发力",
    "throw_extension": "投球伸展",
    "reset": "复位",
    "finish": "划船终点",
    "catch": "起始",
    "recovery": "恢复",
    "carrying": "负重行走",
    "rest": "停步休息",
    "pull": "拉动",
    "pull_down": "下拉",
    "return": "回位",
    "step": "蹬地迈步",
    "setup": "准备",
    "top": "顶部",
    "down": "下拉",
    "hands_down": "双手撑地",
    "landing": "落地",
    "chest_down": "俯卧最低点",
    "step_or_jump_in": "收腿",
    "broad_jump_takeoff": "跳远起跳",
    "flight_or_move": "腾空或移动",
    "reach": "前伸取绳",
    "recover": "向前移动回位",
}


# This policy only controls the browser's frame colouring and report denominator.
# Rep counting, state transitions and coaching feedback remain owned by each
# action analyzer and are deliberately not changed here.
ACTION_PHASE_POLICY: dict[str, dict[str, frozenset[str]]] = {
    "lunge": {
        "standard": frozenset({"stand", "bottom"}),
        "not_applicable": frozenset({"descent", "ascent"}),
    },
    "wall_ball": {
        "standard": frozenset({"bottom", "throw_extension"}),
        "not_applicable": frozenset({"stand", "squat_down", "drive", "reset"}),
    },
    "farmers_carry": {
        "standard": frozenset({"carrying"}),
        "not_applicable": frozenset({"ready", "rest"}),
    },
    "rowing": {
        "standard": frozenset({"catch", "finish"}),
        "not_applicable": frozenset({"drive", "recovery"}),
    },
    "skierg": {
        "standard": frozenset({"pull_down", "bottom"}),
        "not_applicable": frozenset({"top", "return"}),
    },
    "burpee_broad_jump": {
        "standard": frozenset({"chest_down", "broad_jump_takeoff", "landing"}),
        "not_applicable": frozenset({"stand", "hands_down", "step_or_jump_in", "flight_or_move", "reset"}),
    },
    "sled_push": {
        "standard": frozenset({"setup", "drive", "step"}),
        "not_applicable": frozenset({"reset"}),
    },
    "sled_pull": {
        "standard": frozenset({"ready", "pull"}),
        "not_applicable": frozenset({"reach", "recover"}),
    },
}


OFFICIAL_RULEBOOK_URL = "https://maintain.hyrox.com/rulebooks/HYROX_RulebookSingles_EN.pdf"
OFFICIAL_RULEBOOK_NAME = "HYROX Singles Rulebook 26/27"

ACTION_OFFICIAL_RULES: dict[str, list[dict[str, Any]]] = {
    "lunge": [
        {"text": "每次弓步后侧膝盖必须清楚触地。", "pose_observable": True},
        {"text": "每次结束时身体站直，膝关节与髋关节完全伸展。", "pose_observable": True},
        {"text": "左右腿交替完成，次数之间不得额外走步或碎步。", "pose_observable": True},
    ],
    "wall_ball": [
        {"text": "起始时站直，髋关节与膝关节伸展，双手持球。", "pose_observable": True},
        {"text": "深蹲最低点髋部必须低于膝盖。", "pose_observable": True},
        {"text": "双手投球并击中对应目标区域；仅凭人体姿态无法确认是否击中。", "pose_observable": False},
    ],
    "farmers_carry": [
        {"text": "移动过程中必须同时携带两只壶铃。", "pose_observable": False},
        {"text": "双臂伸展，壶铃保持在身体两侧。", "pose_observable": True},
        {"text": "允许放下休息，但放下时壶铃不能向前移动。", "pose_observable": False},
    ],
    "rowing": [
        {"text": "完成 1000 米前必须保持坐在划船机座椅上。", "pose_observable": True},
        {"text": "达到规定距离并经裁判确认前不得站起或离开划船机。", "pose_observable": False},
    ],
    "skierg": [
        {"text": "训练过程中双脚必须留在 SkiErg 底板上；跳起后也必须落回底板。", "pose_observable": False},
        {"text": "完成距离后仍需留在底板上，举手示意并等待裁判确认。", "pose_observable": False},
    ],
    "burpee_broad_jump": [
        {"text": "波比跳最低点胸部必须清楚接触地面。", "pose_observable": True},
        {"text": "双脚必须同时起跳、同时落地，前后差不得超过 5 厘米。", "pose_observable": True},
        {"text": "不得出现额外走步或碎步；后续波比跳双手距脚尖不得超过 30 厘米。", "pose_observable": True},
    ],
    "sled_push": [
        {"text": "起始时运动员和雪橇必须完全位于指定起始区域。", "pose_observable": False},
        {"text": "全程保持在指定赛道内，雪橇需完全越过每段终点线后才能换向。", "pose_observable": False},
    ],
    "sled_pull": [
        {"text": "拉动雪橇时必须始终保持站立，不得坐姿或跪姿拉动。", "pose_observable": True},
        {"text": "持绳时不得踩到工作区前后实线，绳索不得影响相邻赛道。", "pose_observable": False},
        {"text": "雪橇需完全越过每段终点线后，才可跑向另一端继续拉动。", "pose_observable": False},
    ],
}

ACTION_REP_LABELS = {
    "lunge": "弓步",
    "wall_ball": "投球",
    "rowing": "划船",
    "skierg": "下拉",
    "burpee_broad_jump": "波比跳远",
    "sled_push": "蹬步",
    "sled_pull": "拉动",
    "farmers_carry": "负重行走",
}

REP_DECISION_LABELS = {
    "VALID": "有效动作",
    "NO_REP": "未完成（NO_REP）",
    "UNSURE": "无法确认（UNSURE）",
}

BODY_RULE_LABELS = {
    "body_sequence_valid": "完整动作端点顺序",
    "trailing_knee_contact": "后膝触地",
    "full_knee_extension": "触地后的膝关节完全伸展",
    "full_hip_extension": "触地后的髋关节完全伸展",
    "alternating_contact_leg": "左右触地腿交替",
    "no_extra_step_or_shuffle": "无额外走步或碎步",
    "tall_start": "起始姿势站直",
    "hip_below_knee": "深蹲最低点髋部低于膝盖",
    "upward_extension": "投掷前髋膝完全伸展",
    "bilateral_throw_proxy": "双手完成投掷动作",
    "chest_ground_contact": "胸部触地",
    "simultaneous_takeoff": "双脚同步起跳",
    "simultaneous_landing": "双脚同步落地",
    "takeoff_stagger_proxy": "起跳时双脚前后差",
    "landing_stagger_proxy": "落地时双脚前后差",
    "legal_hand_placement_proxy": "双手落地点位置",
    "forward_jump_detected": "向前跳跃距离",
}

REP_REASON_TEXT = {
    "NO_REQUIRED_RULES": "没有可用于计数判定的必需人体规则。",
    "RULE_NOT_EVALUATED": "必需规则没有得到评价。",
    "REP_MEAN_CONFIDENCE_LOW": "整次动作的平均关键点清晰度不足。",
    "REQUIRED_LANDMARK_CONFIDENCE_LOW": "判定所需的关键身体点置信度不足。",
    "DECISIVE_RULE_CONFIDENCE_LOW": "决定计数结果的规则证据置信度不足。",
    "CAMERA_VIEW_UNSUITABLE": "当前拍摄视角不适合可靠判断这项动作规则。",
    "FLOOR_REFERENCE_UNSURE": "地板参考线不可靠，无法确认触地或相对高度。",
    "SINGLE_FRAME_RULE_FAILURE": "异常只出现在单个画面，证据不足以判为未完成。",
    "TRAILING_LEG_UNRESOLVED": "无法可靠分辨后侧腿。",
    "TRAILING_KNEE_NO_CONTACT": "没有检测到后膝明确触地。",
    "TRAILING_KNEE_CONTACT_UNSURE": "后膝触地证据不足。",
    "TRAILING_KNEE_CONTACT_NOT_OBSERVABLE": "后膝区域不可可靠观察。",
    "EXTENSION_NOT_AFTER_CONFIRMED_CONTACT": "没有在本次已确认触地之后观察到伸展。",
    "FULL_KNEE_EXTENSION_NOT_OBSERVABLE": "触地后的膝关节伸展不可可靠观察。",
    "FULL_HIP_EXTENSION_NOT_OBSERVABLE": "触地后的髋关节伸展不可可靠观察。",
    "FULL_KNEE_EXTENSION_NOT_HELD": "膝关节完全伸展没有保持足够画面。",
    "FULL_HIP_EXTENSION_NOT_HELD": "髋关节完全伸展没有保持足够画面。",
    "CONTACT_LEG_UNRESOLVED": "无法可靠确定本次触地腿。",
    "SAME_CONTACT_LEG_REPEATED": "连续两次使用了同一条触地腿。",
    "FOOT_EVENTS_NOT_OBSERVABLE": "脚部事件不可可靠观察。",
    "LEADING_LEG_UNRESOLVED": "无法可靠确定前侧腿。",
    "EXTRA_STEP_OR_SHUFFLE": "动作之间检测到额外走步或碎步。",
    "TALL_START_NOT_OBSERVABLE": "起始站姿不可可靠观察。",
    "TALL_START_REQUIREMENTS_NOT_MET": "起始时髋、膝或躯干没有达到站直要求。",
    "HIP_KNEE_FLOOR_HEIGHT_NOT_OBSERVABLE": "最低点的髋膝相对高度不可可靠观察。",
    "HIP_NOT_BELOW_KNEE": "深蹲最低点髋部没有低于膝盖。",
    "UPWARD_EXTENSION_NOT_OBSERVABLE": "投掷前的髋膝伸展不可可靠观察。",
    "UPWARD_EXTENSION_INCOMPLETE": "投掷前髋膝没有完全伸展。",
    "WRIST_START_NOT_OBSERVABLE": "双手起始位置不可可靠观察。",
    "WRISTS_DID_NOT_START_NEAR_CHEST": "双手没有从胸前附近开始投掷。",
    "BILATERAL_THROW_ENDPOINT_NOT_OBSERVABLE": "双手投掷终点不可可靠观察。",
    "BILATERAL_THROW_NOT_OBSERVABLE": "双手投掷过程不可可靠观察。",
    "BILATERAL_THROW_BODY_SCALE_NOT_OBSERVABLE": "身体尺度不足，无法可靠换算双手投掷幅度。",
    "BOTH_WRISTS_NOT_ABOVE_SHOULDERS": "投掷终点双手腕没有都高于肩部。",
    "BILATERAL_WRIST_RISE_TOO_SMALL": "双手腕从胸前向上的移动幅度不足。",
    "WRISTS_TOO_FAR_FROM_BODY_MIDLINE": "投掷时双手离身体中线过远。",
    "WRIST_PEAK_TIMING_UNSURE": "双手最高点时序证据不足。",
    "WRIST_PEAKS_NOT_SYNCHRONIZED": "双手没有同步到达投掷最高点。",
    "CHEST_GROUND_CONTACT_NOT_CONFIRMED": "没有确认胸部触地。",
    "CHEST_GROUND_CONTACT_UNSURE": "胸部触地证据不足。",
    "CHEST_GROUND_CONTACT_NOT_OBSERVABLE": "胸部触地区域不可可靠观察。",
    "FORWARD_JUMP_NOT_OBSERVABLE": "向前跳跃位移不可可靠观察。",
    "FORWARD_JUMP_DISPLACEMENT_TOO_SMALL": "向前跳跃位移不足。",
    "POST_LANDING_FEET_NOT_OBSERVABLE": "落地后的脚部动作不可可靠观察。",
    "LEGAL_HAND_PLACEMENT_PROXY_NOT_OBSERVABLE": "双手与脚尖的相对落点不可可靠观察。",
    "LEGAL_HAND_PLACEMENT_PROXY_FLOOR_UNSURE": "地板参考不足，无法确认双手落点。",
    "LEGAL_HAND_PLACEMENT_PROXY_FOOT_LENGTH_UNSURE": "脚长尺度不足，无法确认双手落点距离。",
    "LEGAL_HAND_PLACEMENT_PROXY_BORDERLINE": "双手落点接近允许距离边界。",
    "LEGAL_HAND_PLACEMENT_PROXY_TOO_FAR": "双手落地点离脚尖过远。",
}

CAPTURE_ISSUE_CODES = {
    "LOW_VISIBILITY",
    "POSE_MISSING",
    "CAMERA_VIEW_REQUIRED",
    "CAMERA_VIEW_LIMITED",
    "NOT_SEATED_OR_BAD_VIEW",
}


class RepVoiceFeedbackTracker:
    """Build one local speech event from the same segment used by the text report."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._action = ""
        self._previous_count = 0
        self._frames: deque[dict[str, Any]] = deque(maxlen=9000)
        self._continuous_first_seen: dict[tuple[str, str], int] = {}
        self._continuous_last_spoken: dict[tuple[str, str], int] = {}
        self._last_event: dict[str, Any] | None = None

    def update(
        self,
        *,
        action: str,
        reps: int,
        assessment: Mapping[str, Any] | None,
        detected_issues: Sequence[Mapping[str, Any]],
        timestamp_ms: int,
    ) -> dict[str, Any] | None:
        action = str(action)
        reps = max(0, int(reps))
        if action != self._action or reps < self._previous_count:
            self.reset()
            self._action = action

        if action not in {"none", "farmers_carry"}:
            self._frames.append(
                {
                    "timestamp_unix_ms": int(timestamp_ms),
                    "reps": reps,
                    "assessment": dict(assessment or {}),
                    "detected_issues": [dict(item) for item in detected_issues],
                }
            )

        if reps > self._previous_count:
            detail = _summarize_rep_segment(reps, "", list(self._frames), None)
            improvements = list(detail.get("improvements") or [])[:2]
            speech_items = [_speech_friendly_improvement(text) for text in improvements]
            speech_items = [text for text in speech_items if text]
            self._last_event = {
                "id": f"{action}:rep:{reps}",
                "rep": reps,
                "mode": "rep",
                "improvements": improvements,
                "speech": f"第 {reps} 次，需要改进：{'；'.join(speech_items)}" if speech_items else "",
            }
            self._frames.clear()
            self._previous_count = reps
            self._continuous_first_seen.clear()
            return self._last_event

        self._previous_count = max(self._previous_count, reps)
        if action == "farmers_carry":
            event = self._continuous_event(detected_issues, int(timestamp_ms))
            if event is not None:
                self._last_event = event
        return self._last_event

    def _continuous_event(
        self,
        detected_issues: Sequence[Mapping[str, Any]],
        timestamp_ms: int,
    ) -> dict[str, Any] | None:
        current: dict[tuple[str, str], Mapping[str, Any]] = {}
        for item in detected_issues:
            if str(item.get("level", "info")) not in {"warn", "error"}:
                continue
            code = str(item.get("code", ""))
            text = str(item.get("text", "")).strip()
            if not text or code.upper() in CAPTURE_ISSUE_CODES:
                continue
            current[(code, text)] = item

        for key in tuple(self._continuous_first_seen):
            if key not in current:
                self._continuous_first_seen.pop(key, None)
        for key in current:
            self._continuous_first_seen.setdefault(key, timestamp_ms)

        eligible = [
            key
            for key in current
            if timestamp_ms - self._continuous_first_seen[key] >= 1200
            and timestamp_ms - self._continuous_last_spoken.get(key, -10_000_000) >= 8000
        ]
        if not eligible:
            return None
        eligible.sort(
            key=lambda key: (
                0 if str(current[key].get("level")) == "error" else 1,
                -float(current[key].get("confidence", 0.0) or 0.0),
                key,
            )
        )
        key = eligible[0]
        self._continuous_last_spoken[key] = timestamp_ms
        return {
            "id": f"{self._action}:continuous:{key[0]}:{timestamp_ms}",
            "rep": None,
            "mode": "continuous",
            "improvements": [key[1]],
            "speech": f"动作提示：{key[1]}",
        }


def _speech_friendly_improvement(text: str) -> str:
    value = str(text).strip().rstrip("。")
    if "（出现" in value:
        value = value.split("（出现", 1)[0].rstrip("。")
    return value


def _criterion(
    metric: str,
    label: str,
    minimum: float | None,
    maximum: float | None,
    *,
    phases: Sequence[str] = (),
    unit: str = "°",
    note: str = "",
    category: str = "training",
    tolerance: float = 0.0,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "label": label,
        "min": minimum,
        "max": maximum,
        "phases": list(phases),
        "unit": unit,
        "note": note,
        "category": category,
        "category_text": "官方规则的视觉判断" if category == "official_proxy" else "训练技术参考",
        "source": OFFICIAL_RULEBOOK_NAME if category == "official_proxy" else "姿态训练参考",
        "tolerance": max(0.0, float(tolerance)),
    }


ACTION_STANDARDS: dict[str, list[dict[str, Any]]] = {
    "lunge": [
        _criterion("min_knee_angle", "后膝接近地面的角度参考", 75, 125, phases=("bottom",), category="official_proxy", tolerance=8),
        _criterion("torso_angle_abs", "躯干稳定", 0, 30, phases=("bottom", "stand"), note="避免过度前倾", tolerance=8),
        _criterion("min_knee_angle", "站直时膝关节伸展", 155, 180, phases=("stand",), category="official_proxy", tolerance=6),
        _criterion("min_hip_angle", "站直时髋关节伸展", 150, 180, phases=("stand",), category="official_proxy", tolerance=6),
    ],
    "wall_ball": [
        _criterion("hip_knee_depth", "最低点髋部低于膝盖", 0, None, phases=("bottom",), unit="", category="official_proxy", tolerance=0.015),
        _criterion("min_knee_angle", "投球时下肢伸展", 145, 180, phases=("throw_extension",), tolerance=8),
        _criterion("min_elbow_angle", "投球时手臂伸展", 120, 180, phases=("throw_extension",), tolerance=10),
    ],
    "farmers_carry": [
        _criterion("mean_elbow_angle", "双臂伸展在身体两侧", 150, 180, phases=("carrying",), category="official_proxy", tolerance=8),
        _criterion("torso_angle_abs", "躯干前后倾", 0, 30, phases=("carrying",), tolerance=8),
        _criterion("shoulder_tilt_abs", "肩线左右高差", 0, 0.10, phases=("carrying",), unit="", note="左右肩尽量保持水平", tolerance=0.025),
        _criterion("hip_tilt_abs", "髋线左右高差", 0, 0.10, phases=("carrying",), unit="", note="左右髋尽量保持水平", tolerance=0.025),
    ],
    "rowing": [
        _criterion("min_knee_angle", "起始位屈膝", 55, 120, phases=("catch",), tolerance=10),
        _criterion("min_knee_angle", "结束位腿部伸展", 140, 180, phases=("finish",), tolerance=8),
        _criterion("torso_angle_abs", "结束位躯干控制", 0, 40, phases=("finish",), tolerance=8),
        _criterion("mean_elbow_angle", "结束位拉手位置", 50, 130, phases=("finish",), tolerance=12),
    ],
    "skierg": [
        _criterion("min_knee_angle", "下拉时避免过度深蹲", 95, 180, phases=("pull_down", "bottom"), tolerance=10),
        _criterion("torso_angle_abs", "下拉时髋部折叠", 10, 70, phases=("pull_down", "bottom"), tolerance=10),
        _criterion("wrist_asymmetry", "双手同步下拉", 0, 0.12, phases=("pull_down", "bottom"), unit="", tolerance=0.03),
    ],
    "burpee_broad_jump": [
        _criterion("torso_angle_abs", "胸部接近地面的姿态", 55, 100, phases=("chest_down",), category="official_proxy", tolerance=10),
        _criterion("min_knee_angle", "起跳准备屈膝", 60, 165, phases=("step_or_jump_in", "broad_jump_takeoff"), tolerance=12),
    ],
    "sled_push": [
        _criterion("torso_angle_abs", "发力时躯干前倾", 20, 70, phases=("setup", "drive", "step"), tolerance=10),
        _criterion("min_knee_angle", "蹬地时腿部伸展", 95, 180, phases=("drive", "step"), tolerance=10),
    ],
    "sled_pull": [
        _criterion("mean_knee_angle", "拉动时保持站立", 110, 180, phases=("ready", "pull"), category="official_proxy", tolerance=10),
        _criterion("torso_angle_abs", "拉动时控制后仰", 0, 45, phases=("ready", "pull"), tolerance=10),
    ],
}


ANGLE_SPECS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "lunge": (("left_knee_angle", "左膝", "left_knee"), ("right_knee_angle", "右膝", "right_knee"), ("torso_angle", "躯干", "left_hip")),
    "wall_ball": (("left_knee_angle", "左膝", "left_knee"), ("right_knee_angle", "右膝", "right_knee"), ("left_elbow_angle", "左肘", "left_elbow"), ("right_elbow_angle", "右肘", "right_elbow")),
    "rowing": (("left_knee_angle", "左膝", "left_knee"), ("right_knee_angle", "右膝", "right_knee"), ("left_elbow_angle", "左肘", "left_elbow"), ("right_elbow_angle", "右肘", "right_elbow"), ("torso_angle", "躯干", "left_hip")),
    "skierg": (("left_knee_angle", "左膝", "left_knee"), ("right_knee_angle", "右膝", "right_knee"), ("torso_angle", "躯干", "left_hip")),
    "burpee_broad_jump": (("left_knee_angle", "左膝", "left_knee"), ("right_knee_angle", "右膝", "right_knee"), ("torso_angle", "躯干", "left_hip")),
    "sled_push": (("left_knee_angle", "左膝", "left_knee"), ("right_knee_angle", "右膝", "right_knee"), ("torso_angle", "躯干", "left_hip")),
    "sled_pull": (("left_elbow_angle", "左肘", "left_elbow"), ("right_elbow_angle", "右肘", "right_elbow"), ("left_knee_angle", "左膝", "left_knee"), ("right_knee_angle", "右膝", "right_knee"), ("torso_angle", "躯干", "left_hip")),
    "farmers_carry": (("torso_angle", "躯干", "left_hip"),),
}


RECOVERY_PHASES = {"recover", "recovery", "reset", "return", "reach", "walking"}
EFFORT_ONLY_CODES = {
    "ARMS_ONLY_PULL",
    "NO_CLEAR_PULL",
    "NO_LEG_DRIVE",
    "EARLY_ARM_PULL",
    "INCOMPLETE_PULL",
    "ARMS_NOT_HIGH_ENOUGH",
}


def standards_for(action: str) -> list[dict[str, Any]]:
    standards = deepcopy(ACTION_STANDARDS.get(action, []))
    for item in standards:
        item["range_text"] = _range_text(item)
        item["phase_text"] = " / ".join(PHASE_LABELS.get(value, value) for value in item["phases"]) or "全程"
    return standards


def official_rules_for(action: str) -> list[dict[str, Any]]:
    return deepcopy(ACTION_OFFICIAL_RULES.get(action, []))


def visible_feedback(items: Sequence[Mapping[str, Any]], phase: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items:
        code = str(item.get("code", ""))
        if phase in RECOVERY_PHASES and code.upper() in EFFORT_ONLY_CODES:
            continue
        resolved = {"level": str(item.get("level", "info")), "code": code, "text": str(item.get("text", ""))}
        confidence = _number(item.get("confidence"))
        if confidence is not None:
            resolved["confidence"] = round(max(0.0, min(1.0, confidence)), 3)
        output.append(resolved)
    return output


def assess_action(
    action: str,
    phase: str,
    features: Mapping[str, Any] | None,
    feedback: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    values = dict(features or {})
    policy = ACTION_PHASE_POLICY.get(action)
    phase_has_standard = policy is None or phase in policy["standard"]
    phase_not_applicable = policy is not None and phase in policy["not_applicable"]
    active: list[dict[str, Any]] = []
    if phase_has_standard:
        for standard in standards_for(action):
            phases = standard["phases"]
            if phases and phase not in phases:
                continue
            value = _metric_value(standard["metric"], values)
            if value is None:
                continue
            passed = (standard["min"] is None or value >= standard["min"]) and (
                standard["max"] is None or value <= standard["max"]
            )
            tolerance = float(standard.get("tolerance", 0.0))
            clear_failure = (standard["min"] is not None and value < standard["min"] - tolerance) or (
                standard["max"] is not None and value > standard["max"] + tolerance
            )
            active.append(
                {
                    "label": standard["label"],
                    "value": round(value, 2),
                    "unit": standard["unit"],
                    "range_text": standard["range_text"],
                    "passed": passed,
                    "clear_failure": clear_failure,
                    "borderline": not passed and not clear_failure,
                    "metric": standard["metric"],
                    "category": standard["category"],
                    "category_text": standard["category_text"],
                }
            )
    problem_feedback = [item for item in feedback if str(item.get("level", "info")) in {"warn", "error"}]
    visible_score = _number(values.get("visible_score"))
    low_quality = visible_score is not None and visible_score < 0.45
    strong_problem_feedback = []
    for item in problem_feedback:
        confidence = _number(item.get("confidence"))
        if str(item.get("code", "")).upper() not in CAPTURE_ISSUE_CODES and (1.0 if confidence is None else confidence) >= 0.55:
            strong_problem_feedback.append(item)
    if low_quality:
        status = "unknown"
        evaluable = False
        evaluation_mode = "low_quality"
        status_reason = "关键点清晰度不足，本帧不作红绿判断"
    elif phase_not_applicable:
        status = "good"
        evaluable = False
        evaluation_mode = "not_applicable"
        status_reason = "当前为无明确姿态标准的过渡阶段"
    elif not phase_has_standard:
        status = "unknown"
        evaluable = False
        evaluation_mode = "unavailable"
        status_reason = "当前阶段无法可靠评价"
    else:
        clear_failure = any(item["clear_failure"] for item in active)
        borderline = any(item["borderline"] for item in active)
        if clear_failure or strong_problem_feedback:
            status = "bad"
            status_reason = "检测到明显偏离参考范围或稳定动作问题"
        elif borderline:
            status = "unknown"
            status_reason = "数值接近边界，暂不判红或判绿"
        else:
            status = "good"
            status_reason = "当前关键阶段未发现明确问题"
        evaluable = status in {"good", "bad"}
        evaluation_mode = "standard"
    angles: list[dict[str, Any]] = []
    for key, label, anchor in ANGLE_SPECS.get(action, ()):
        value = _number(values.get(key))
        if value is None:
            continue
        related = [item for item in active if _angle_matches_metric(key, item["metric"])]
        angle_status = "bad" if any(item["clear_failure"] for item in related) else "good" if related and all(item["passed"] for item in related) else "neutral"
        angles.append({"key": key, "label": label, "anchor": anchor, "value": round(value, 1), "status": angle_status})
    return {
        "status": status,
        "evaluable": evaluable,
        "evaluation_mode": evaluation_mode,
        "status_reason": status_reason,
        "visible_score": None if visible_score is None else round(visible_score, 3),
        "phase": phase,
        "criteria": active,
        "angles": angles,
        "problem_codes": [str(item.get("code", "")) for item in problem_feedback],
    }


def enrich_report(report: Mapping[str, Any]) -> dict[str, Any]:
    output = deepcopy(dict(report))
    frames = list(output.get("frames") or [])
    summary = dict(output.get("summary") or {})
    action = str(summary.get("action", frames[-1].get("action", "none") if frames else "none"))
    evaluable = [frame for frame in frames if (frame.get("assessment") or {}).get("evaluable") is True]
    good = sum(1 for frame in evaluable if frame["assessment"]["status"] == "good")
    bad = len(evaluable) - good
    issue_counts: Counter[tuple[str, str]] = Counter()
    capture_issue_counts: Counter[tuple[str, str]] = Counter()
    for frame in frames:
        issues = frame.get("detected_issues") or frame.get("feedback") or []
        for item in issues:
            if isinstance(item, Mapping) and str(item.get("level", "info")) in {"warn", "error"}:
                key = (str(item.get("code", "")), str(item.get("text", "")))
                target = capture_issue_counts if key[0].upper() in CAPTURE_ISSUE_CODES else issue_counts
                target[key] += 1
    compliance = None if not evaluable else round(good * 100.0 / len(evaluable), 1)
    overall = _overall_status(compliance)
    rep_details = _rep_details(action, frames)
    strengths = _overall_strengths(evaluable)
    next_focus = [text for (_, text), _ in issue_counts.most_common(3) if text]
    analysis = {
        "overall_status": overall,
        "compliance_rate": compliance,
        "compliance_explanation": "合规率只统计关键点清晰、处于可评价阶段的画面；过渡阶段和边界不确定画面不计入。",
        "evaluable_frames": len(evaluable),
        "compliant_frames": good,
        "nonstandard_frames": bad,
        "common_issues": [
            {"code": code, "text": text, "count": count}
            for (code, text), count in issue_counts.most_common(5)
        ],
        "capture_quality_issues": [
            {"code": code, "text": text, "count": count}
            for (code, text), count in capture_issue_counts.most_common(3)
        ],
        "strengths": strengths,
        "next_focus": next_focus,
        "rep_details": rep_details,
        "standards": standards_for(action),
        "official_rules": official_rules_for(action),
        "reference": {"name": OFFICIAL_RULEBOOK_NAME, "url": OFFICIAL_RULEBOOK_URL},
    }
    summary["overall_status"] = overall
    summary["compliance_rate"] = compliance
    output["summary"] = summary
    output["analysis"] = analysis
    return output


def render_text_report(report: Mapping[str, Any]) -> str:
    enriched = enrich_report(report)
    summary = enriched.get("summary") or {}
    analysis = enriched.get("analysis") or {}
    generated_ms = _number(enriched.get("generated_at_unix_ms"))
    generated_at = (
        datetime.fromtimestamp(generated_ms / 1000.0).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        if generated_ms is not None
        else datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    )
    compliance = analysis.get("compliance_rate")
    lines = [
        "HYROX 动作分析文字报告",
        "=" * 28,
        f"动作：{summary.get('action_label') or summary.get('action') or '未知'}",
        f"生成时间：{generated_at}",
        f"识别完整周期：{int(summary.get('candidate_count', summary.get('reps')) or 0)} 次",
        f"人体规则有效动作：{int(summary.get('pose_valid_rep_count', summary.get('reps')) or 0)} 次",
        f"人体规则未完成：{int(summary.get('no_rep_count') or 0)} 次",
        f"人体规则无法确认：{int(summary.get('unsure_count') or 0)} 次",
        f"总体评价：{analysis.get('overall_status', '暂无评价')}",
        f"关键阶段合规率：{'暂无' if compliance is None else f'{compliance}%'}",
        "说明：合规率不包含过渡阶段、低清晰度和接近判定边界的画面。姿态系统无法确认器械重量、距离、是否击中目标或是否越过场地线。",
        "",
        "一、做得好的地方",
    ]
    strengths = list(analysis.get("strengths") or [])
    lines.extend(f"- {item}" for item in strengths) if strengths else lines.append("- 暂无足够清晰的关键阶段用于总结。")
    lines.extend(["", "二、下一次优先改进"])
    next_focus = list(analysis.get("next_focus") or [])
    lines.extend(f"- {item}" for item in next_focus) if next_focus else lines.append("- 未发现持续性动作问题，继续保持完整动作幅度和稳定节奏。")
    lines.extend(["", "三、逐次动作表现"])
    rep_details = list(analysis.get("rep_details") or [])
    if not rep_details:
        lines.append("- 未识别到一套完整动作，无法按次数拆分。")
    for detail in rep_details:
        count_status = str(detail.get("count_status", ""))
        count_status_text = str(detail.get("count_status_text", ""))
        header_status = (
            f"{count_status_text}；动作质量：{detail['status']}"
            if count_status_text
            else str(detail["status"])
        )
        lines.extend(
            [
                "",
                f"{detail['title']}（{header_status}，{detail['time_text']}）",
            ]
        )
        if count_status in {"NO_REP", "UNSURE"}:
            lines.extend(
                [
                    "  未计为有效动作的原因：",
                    *[
                        f"  - {item}"
                        for item in detail.get("invalid_reasons")
                        or ["本次计数判定的必需证据不足。"]
                    ],
                ]
            )
        lines.extend(
            [
                "  做得好：",
                *[f"  - {item}" for item in detail.get("positives") or ["本次没有足够清晰的达标阶段可总结。"]],
                "  需要改进：",
                *[f"  - {item}" for item in detail.get("improvements") or ["未发现持续性问题。"]],
            ]
        )
    lines.extend(["", "四、HYROX 官方动作要求"])
    lines.extend(f"- {item['text']}" for item in analysis.get("official_rules") or [])
    reference = analysis.get("reference") or {}
    lines.extend(
        [
            "",
            "五、参考来源与限制",
            f"- {reference.get('name', OFFICIAL_RULEBOOK_NAME)}",
            f"- {reference.get('url', OFFICIAL_RULEBOOK_URL)}",
            "- 角度范围属于视频姿态训练参考，并非 HYROX 裁判手册规定的固定角度。最终比赛判罚以现场裁判和官方规则为准。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _overall_status(compliance: float | None) -> str:
    if compliance is None:
        return "暂无足够清晰的关键阶段"
    if compliance >= 90.0:
        return "优秀"
    if compliance >= 80.0:
        return "良好"
    if compliance >= 60.0:
        return "有亮点，仍可改进"
    return "建议重点调整"


def _overall_strengths(frames: Sequence[Mapping[str, Any]]) -> list[str]:
    stats: dict[str, Counter[str]] = {}
    for frame in frames:
        assessment = frame.get("assessment") or {}
        for item in assessment.get("criteria") or []:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label", ""))
            if not label:
                continue
            counter = stats.setdefault(label, Counter())
            counter["observed"] += 1
            if item.get("passed") is True:
                counter["passed"] += 1
    ranked = sorted(
        (
            (counter["passed"] / max(1, counter["observed"]), counter["passed"], label)
            for label, counter in stats.items()
            if counter["passed"] > 0
        ),
        reverse=True,
    )
    return [f"{label}在大多数清晰关键画面中达到参考范围。" for ratio, _, label in ranked[:3] if ratio >= 0.6]


def _rep_details(action: str, frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not frames:
        return []
    label = ACTION_REP_LABELS.get(action, "动作")
    base_timestamp = _number(frames[0].get("timestamp_unix_ms"))
    has_candidate_counts = any("candidate_count" in frame for frame in frames)
    count_key = "candidate_count" if has_candidate_counts else "reps"
    previous_count = 0
    segment_start = 0
    segments: list[tuple[int, Sequence[Mapping[str, Any]], Mapping[str, Any] | None]] = []
    for index, frame in enumerate(frames):
        count = _safe_int(frame.get(count_key))
        if count <= previous_count:
            continue
        decision = frame.get("last_rep_decision")
        segments.append(
            (
                count,
                frames[segment_start : index + 1],
                decision if isinstance(decision, Mapping) else None,
            )
        )
        segment_start = index + 1
        previous_count = count
    if not segments and action == "farmers_carry":
        return [_summarize_rep_segment(0, "整段负重行走", frames, base_timestamp)]
    return [
        _summarize_rep_segment(
            rep_index,
            f"第 {rep_index} 次{label}",
            segment,
            base_timestamp,
            decision,
        )
        for rep_index, segment, decision in segments
    ]


def _summarize_rep_segment(
    rep_index: int,
    title: str,
    frames: Sequence[Mapping[str, Any]],
    base_timestamp: float | None,
    decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evaluable = [frame for frame in frames if (frame.get("assessment") or {}).get("evaluable") is True]
    good = sum(1 for frame in evaluable if (frame.get("assessment") or {}).get("status") == "good")
    compliance = None if not evaluable else round(good * 100.0 / len(evaluable), 1)
    criterion_stats: dict[str, dict[str, Any]] = {}
    issue_counts: Counter[tuple[str, str]] = Counter()
    for frame in frames:
        assessment = frame.get("assessment") or {}
        for item in assessment.get("criteria") or []:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label", ""))
            if not label:
                continue
            current = criterion_stats.setdefault(
                label,
                {"passed": 0, "failed": 0, "values": [], "range_text": str(item.get("range_text", "")), "unit": str(item.get("unit", ""))},
            )
            if item.get("passed") is True:
                current["passed"] += 1
            if item.get("clear_failure") is True:
                current["failed"] += 1
            value = _number(item.get("value"))
            if value is not None:
                current["values"].append(value)
        issues = frame.get("detected_issues") or frame.get("feedback") or []
        for item in issues:
            if not isinstance(item, Mapping) or str(item.get("level", "info")) not in {"warn", "error"}:
                continue
            code = str(item.get("code", ""))
            text = str(item.get("text", ""))
            if code.upper() not in CAPTURE_ISSUE_CODES and text:
                issue_counts[(code, text)] += 1
    positives: list[str] = []
    improvements: list[str] = []
    for label, item in criterion_stats.items():
        values = item["values"]
        value_text = ""
        if values:
            value_text = f"，代表值约 {median(values):g}{item['unit']}"
        if item["passed"] > 0 and item["passed"] >= item["failed"]:
            positives.append(f"{label}达到参考范围 {item['range_text']}{value_text}。")
        if item["failed"] > 0:
            improvements.append(f"{label}有 {item['failed']} 个清晰关键画面明显超出 {item['range_text']}。")
    for (_, text), count in issue_counts.most_common(3):
        improvements.append(f"{text}（出现 {count} 个画面）。")
    if not positives and compliance is not None and compliance >= 80.0:
        positives.append("关键阶段衔接完整，未发现明显偏离。")
    start_timestamp = _number(frames[0].get("timestamp_unix_ms")) if frames else None
    end_timestamp = _number(frames[-1].get("timestamp_unix_ms")) if frames else None
    if base_timestamp is not None and start_timestamp is not None and end_timestamp is not None:
        time_text = f"视频 {max(0.0, (start_timestamp - base_timestamp) / 1000.0):.1f}-{max(0.0, (end_timestamp - base_timestamp) / 1000.0):.1f} 秒"
    else:
        time_text = f"{len(frames)} 个画面"
    count_status = str(decision.get("status", "")) if decision is not None else ""
    if not count_status and rep_index > 0:
        count_status = "VALID"
    return {
        "rep_index": rep_index,
        "title": title,
        "status": _overall_status(compliance),
        "count_status": count_status,
        "count_status_text": REP_DECISION_LABELS.get(count_status, ""),
        "invalid_reasons": _invalid_rep_reasons(decision),
        "compliance_rate": compliance,
        "evaluable_frames": len(evaluable),
        "time_text": time_text,
        "positives": positives[:4],
        "improvements": list(dict.fromkeys(improvements))[:4],
    }


def _invalid_rep_reasons(decision: Mapping[str, Any] | None) -> list[str]:
    if decision is None:
        return []
    status = str(decision.get("status", ""))
    if status not in {"NO_REP", "UNSURE"}:
        return []

    reasons: list[str] = []
    rule_reason_codes: set[str] = set()
    decisive_statuses = {"FAIL"} if status == "NO_REP" else {"UNSURE", "NOT_APPLICABLE"}
    for rule in decision.get("rules") or []:
        if not isinstance(rule, Mapping) or rule.get("required_for_count") is False:
            continue
        rule_status = str(rule.get("status", ""))
        if rule_status not in decisive_statuses:
            continue
        rule_id = str(rule.get("rule_id", ""))
        label = BODY_RULE_LABELS.get(rule_id, _humanize_code(rule_id))
        reason_code = str(rule.get("reason_code", "")).strip().upper()
        if reason_code:
            rule_reason_codes.add(reason_code)
        reason = _rep_reason_text(reason_code)
        if reason:
            reasons.append(f"{label}：{reason}")
        elif status == "NO_REP":
            reasons.append(f"{label}未达到计数要求。")
        else:
            reasons.append(f"{label}的证据不足，无法确认。")

    if status == "UNSURE":
        for code in decision.get("reason_codes") or []:
            resolved_code = str(code).strip().upper()
            if resolved_code in rule_reason_codes:
                continue
            text = _rep_reason_text(resolved_code)
            if text:
                reasons.append(text)

    if not reasons:
        fallback = (
            "一项或多项必需人体规则没有达到计数要求。"
            if status == "NO_REP"
            else "判定所需的人体关键点、视角或动作时序证据不足。"
        )
        reasons.append(fallback)
    return list(dict.fromkeys(reasons))


def _rep_reason_text(code: str) -> str:
    resolved = str(code).strip().upper()
    if not resolved:
        return ""
    if resolved in REP_REASON_TEXT:
        return REP_REASON_TEXT[resolved]
    if resolved.endswith("_NOT_OBSERVABLE"):
        return "相关身体部位或动作阶段不可可靠观察。"
    if resolved.endswith("_UNSURE") or resolved.endswith("_BORDERLINE"):
        return "证据接近判定边界，无法可靠确认。"
    if resolved.endswith("_ASYNCHRONOUS"):
        return "左右动作没有达到同步要求。"
    if resolved.endswith("_NOT_HELD"):
        return "目标姿势没有保持足够画面。"
    return ""


def _humanize_code(code: str) -> str:
    resolved = str(code).strip().replace("_", " ")
    return resolved if resolved else "必需人体规则"


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def _range_text(item: Mapping[str, Any]) -> str:
    minimum, maximum, unit = item.get("min"), item.get("max"), str(item.get("unit", ""))
    if minimum is not None and maximum is not None:
        return f"{minimum:g}–{maximum:g}{unit}"
    if minimum is not None:
        return f"≥ {minimum:g}{unit}"
    if maximum is not None:
        return f"≤ {maximum:g}{unit}"
    return "观察动作顺序"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _metric_value(metric: str, values: Mapping[str, Any]) -> float | None:
    if metric.endswith("_abs"):
        value = _number(values.get(metric[:-4]))
        return None if value is None else abs(value)
    if metric == "mean_elbow_angle":
        return _mean(values.get("left_elbow_angle"), values.get("right_elbow_angle"))
    if metric == "mean_knee_angle":
        return _mean(values.get("left_knee_angle"), values.get("right_knee_angle"))
    if metric == "wrist_asymmetry":
        left, right = _number(values.get("left_wrist_y")), _number(values.get("right_wrist_y"))
        return None if left is None or right is None else abs(left - right)
    return _number(values.get(metric))


def _mean(*values: Any) -> float | None:
    numbers = [number for number in (_number(value) for value in values) if number is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _angle_matches_metric(angle_key: str, metric: str) -> bool:
    if "knee" in angle_key and "knee" in metric:
        return True
    if "elbow" in angle_key and "elbow" in metric:
        return True
    return angle_key == metric or (angle_key == "torso_angle" and metric == "torso_angle_abs")


__all__ = [
    "RepVoiceFeedbackTracker",
    "assess_action",
    "enrich_report",
    "official_rules_for",
    "render_text_report",
    "standards_for",
    "visible_feedback",
]
