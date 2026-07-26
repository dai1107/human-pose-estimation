"""Default-off Round 10 action-gating shadow baseline.

The model is deliberately dependency-light and auditable.  It is never trained
from filename or AI proposals: callers must satisfy the separate data-readiness
gate before passing samples to :meth:`LogisticActionModel.fit`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts import ActionGatingContract


BASE_FEATURE_NAMES = (
    "visible_score",
    "min_knee_angle",
    "min_hip_angle",
    "max_hip_angle",
    "min_elbow_angle",
    "max_elbow_angle",
    "torso_angle",
    "body_center_x",
    "body_center_y",
    "body_height_norm",
    "wrist_distance_norm",
    "ankle_distance_norm",
    "hip_knee_depth",
    "left_wrist_to_hip_y",
    "right_wrist_to_hip_y",
    "equipment_visible",
)
WINDOW_STATISTICS = ("mean", "std", "delta")
ACTION_FEATURE_NAMES = tuple(
    f"{name}_{statistic}"
    for name in BASE_FEATURE_NAMES
    for statistic in WINDOW_STATISTICS
)
ACTION_FEATURE_SCHEMA_VERSION = "body_canonical_window_v1"


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return resolved if math.isfinite(resolved) else default


def _feature_value(features: Mapping[str, object], name: str) -> float:
    if name == "equipment_visible":
        raw = features.get(name, features.get("object_visible", False))
        if isinstance(raw, str):
            return 1.0 if raw.strip().lower() in {"true", "yes", "visible", "1"} else 0.0
        return 1.0 if bool(raw) else 0.0
    return _finite_float(features.get(name), 0.0)


class ActionFeatureWindow:
    """Causal fixed-duration window over body-canonical pose features."""

    def __init__(self, contract: ActionGatingContract) -> None:
        self.contract = contract
        self._frames: deque[tuple[int, dict[str, float]]] = deque()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def ready(self) -> bool:
        return self.frame_count >= self.contract.minimum_window_frames

    def reset(self) -> None:
        self._frames.clear()

    def update(self, features: Mapping[str, object], *, timestamp_ms: int) -> None:
        values = {name: _feature_value(features, name) for name in BASE_FEATURE_NAMES}
        self._frames.append((int(timestamp_ms), values))
        cutoff = int(timestamp_ms) - self.contract.window_ms
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()
        while len(self._frames) > self.contract.maximum_window_frames:
            self._frames.popleft()

    def vector(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(len(ACTION_FEATURE_NAMES), dtype=np.float64)
        matrix = np.asarray(
            [[values[name] for name in BASE_FEATURE_NAMES] for _, values in self._frames],
            dtype=np.float64,
        )
        means = matrix.mean(axis=0)
        stds = matrix.std(axis=0)
        deltas = matrix[-1] - matrix[0]
        output: list[float] = []
        for index in range(len(BASE_FEATURE_NAMES)):
            output.extend((float(means[index]), float(stds[index]), float(deltas[index])))
        return np.asarray(output, dtype=np.float64)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(slots=True)
class LogisticActionModel:
    """Auditable multinomial logistic-regression model with temperature scaling."""

    classes: tuple[str, ...]
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    temperature: float = 1.0
    model_version: str = "unversioned"
    training_metadata: dict[str, object] = field(default_factory=dict)
    model_hash: str = ""

    def __post_init__(self) -> None:
        feature_count = len(self.feature_names)
        class_count = len(self.classes)
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.scale = np.asarray(self.scale, dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.intercept = np.asarray(self.intercept, dtype=np.float64)
        if self.mean.shape != (feature_count,) or self.scale.shape != (feature_count,):
            raise ValueError("normalization arrays do not match feature schema")
        if self.weights.shape != (feature_count, class_count):
            raise ValueError("weight matrix does not match features/classes")
        if self.intercept.shape != (class_count,):
            raise ValueError("intercept does not match classes")
        if self.temperature <= 0 or not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite and > 0")
        self.scale = np.where(self.scale > 1e-9, self.scale, 1.0)

    @classmethod
    def fit(
        cls,
        matrix: np.ndarray,
        labels: Sequence[str],
        *,
        classes: Sequence[str],
        feature_names: Sequence[str] = ACTION_FEATURE_NAMES,
        epochs: int = 600,
        learning_rate: float = 0.08,
        l2: float = 1e-3,
        model_version: str = "action_gate_logreg_v1",
        training_metadata: Mapping[str, object] | None = None,
    ) -> LogisticActionModel:
        values = np.asarray(matrix, dtype=np.float64)
        resolved_classes = tuple(str(value) for value in classes)
        resolved_features = tuple(str(value) for value in feature_names)
        if values.ndim != 2 or values.shape[1] != len(resolved_features):
            raise ValueError("matrix shape does not match feature schema")
        if values.shape[0] != len(labels) or values.shape[0] == 0:
            raise ValueError("labels must match a non-empty matrix")
        by_class = {name: index for index, name in enumerate(resolved_classes)}
        if any(label not in by_class for label in labels):
            raise ValueError("labels contain a class outside the frozen class list")
        present = {str(label) for label in labels}
        missing = [name for name in resolved_classes if name not in present]
        if missing:
            raise ValueError(f"training data missing classes: {', '.join(missing)}")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale > 1e-9, scale, 1.0)
        normalized = (values - mean) / scale
        targets = np.asarray([by_class[str(label)] for label in labels], dtype=np.int64)
        one_hot = np.eye(len(resolved_classes), dtype=np.float64)[targets]
        counts = np.bincount(targets, minlength=len(resolved_classes)).astype(np.float64)
        sample_weights = values.shape[0] / np.maximum(counts[targets] * len(resolved_classes), 1.0)
        weights = np.zeros((values.shape[1], len(resolved_classes)), dtype=np.float64)
        intercept = np.zeros(len(resolved_classes), dtype=np.float64)
        for _ in range(max(1, int(epochs))):
            probabilities = _softmax(normalized @ weights + intercept)
            error = (probabilities - one_hot) * sample_weights[:, None]
            gradient_w = normalized.T @ error / values.shape[0] + float(l2) * weights
            gradient_b = error.mean(axis=0)
            weights -= float(learning_rate) * gradient_w
            intercept -= float(learning_rate) * gradient_b

        logits = normalized @ weights + intercept
        temperature_candidates = (0.75, 1.0, 1.25, 1.5, 2.0)
        losses: list[tuple[float, float]] = []
        for temperature in temperature_candidates:
            probabilities = _softmax(logits / temperature)
            selected = probabilities[np.arange(values.shape[0]), targets]
            losses.append((float(-np.log(np.maximum(selected, 1e-12)).mean()), temperature))
        temperature = min(losses)[1]
        metadata = dict(training_metadata or {})
        metadata.update(
            {
                "sample_count": int(values.shape[0]),
                "class_counts": {name: int(counts[index]) for index, name in enumerate(resolved_classes)},
                "probability_calibration": "temperature_scaling",
                "calibration_scope": "training_only_requires_grouped_validation_before_release",
            }
        )
        model = cls(
            classes=resolved_classes,
            feature_names=resolved_features,
            mean=mean,
            scale=scale,
            weights=weights,
            intercept=intercept,
            temperature=temperature,
            model_version=model_version,
            training_metadata=metadata,
        )
        model.model_hash = model.compute_hash()
        return model

    def predict_proba(self, vector: Sequence[float] | np.ndarray) -> dict[str, float]:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (len(self.feature_names),):
            raise ValueError("feature vector does not match model schema")
        normalized = (values - self.mean) / self.scale
        probabilities = _softmax((normalized @ self.weights + self.intercept) / self.temperature)
        return {name: float(probabilities[index]) for index, name in enumerate(self.classes)}

    def _payload(self, *, include_hash: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "model_family": "multinomial_logistic_regression",
            "model_version": self.model_version,
            "feature_schema_version": ACTION_FEATURE_SCHEMA_VERSION,
            "classes": list(self.classes),
            "feature_names": list(self.feature_names),
            "normalization": {"mean": self.mean.tolist(), "scale": self.scale.tolist()},
            "weights": self.weights.tolist(),
            "intercept": self.intercept.tolist(),
            "temperature": self.temperature,
            "training_metadata": self.training_metadata,
        }
        if include_hash:
            payload["model_hash"] = self.model_hash or self.compute_hash()
        return payload

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload(include_hash=False)).encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.model_hash = self.compute_hash()
        output.write_text(json.dumps(self._payload(include_hash=True), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> LogisticActionModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        normalization = payload.get("normalization") or {}
        model = cls(
            classes=tuple(payload["classes"]),
            feature_names=tuple(payload["feature_names"]),
            mean=np.asarray(normalization["mean"], dtype=np.float64),
            scale=np.asarray(normalization["scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            intercept=np.asarray(payload["intercept"], dtype=np.float64),
            temperature=float(payload.get("temperature", 1.0)),
            model_version=str(payload.get("model_version", "unversioned")),
            training_metadata=dict(payload.get("training_metadata") or {}),
            model_hash=str(payload.get("model_hash", "")),
        )
        if not model.model_hash or model.model_hash != model.compute_hash():
            raise ValueError("action model hash mismatch")
        if tuple(model.feature_names) != ACTION_FEATURE_NAMES:
            raise ValueError("action model feature schema is incompatible")
        return model


def grouped_cross_validate_logistic(
    matrix: np.ndarray,
    labels: Sequence[str],
    groups: Sequence[str],
    *,
    classes: Sequence[str],
    feature_names: Sequence[str] = ACTION_FEATURE_NAMES,
    maximum_folds: int = 5,
    epochs: int = 300,
) -> dict[str, object]:
    """Deterministic group-exclusive CV; no record/subject may cross a fold."""
    values = np.asarray(matrix, dtype=np.float64)
    resolved_labels = np.asarray([str(value) for value in labels], dtype=object)
    resolved_groups = np.asarray([str(value) for value in groups], dtype=object)
    resolved_classes = tuple(str(value) for value in classes)
    if values.ndim != 2 or values.shape[0] != len(resolved_labels) or len(resolved_labels) != len(resolved_groups):
        raise ValueError("matrix, labels, and groups must contain the same samples")
    unique_groups = sorted(set(resolved_groups.tolist()))
    if len(unique_groups) < 2:
        raise ValueError("grouped cross-validation requires at least two independent groups")
    fold_count = min(max(2, int(maximum_folds)), len(unique_groups))
    fold_groups = [set(unique_groups[index::fold_count]) for index in range(fold_count)]
    predictions: list[str | None] = [None] * len(resolved_labels)
    probability_rows: list[dict[str, float] | None] = [None] * len(resolved_labels)
    folds: list[dict[str, object]] = []
    for fold_index, held_out in enumerate(fold_groups):
        test_mask = np.asarray([group in held_out for group in resolved_groups], dtype=bool)
        train_mask = ~test_mask
        train_classes = set(resolved_labels[train_mask].tolist())
        missing = sorted(set(resolved_classes) - train_classes)
        if missing:
            raise ValueError(
                f"fold {fold_index} training split missing classes: {', '.join(missing)}"
            )
        model = LogisticActionModel.fit(
            values[train_mask],
            resolved_labels[train_mask].tolist(),
            classes=resolved_classes,
            feature_names=feature_names,
            epochs=epochs,
            model_version=f"cv_fold_{fold_index}",
            training_metadata={
                "held_out_groups": sorted(held_out),
                "group_exclusive": True,
            },
        )
        test_indices = np.flatnonzero(test_mask)
        for sample_index in test_indices:
            probabilities = model.predict_proba(values[sample_index])
            predicted = max(probabilities, key=probabilities.get)
            predictions[int(sample_index)] = predicted
            probability_rows[int(sample_index)] = probabilities
        folds.append(
            {
                "fold": fold_index,
                "train_group_count": len(unique_groups) - len(held_out),
                "test_group_count": len(held_out),
                "held_out_groups": sorted(held_out),
                "train_sample_count": int(train_mask.sum()),
                "test_sample_count": int(test_mask.sum()),
                "group_overlap": [],
            }
        )
    if any(value is None for value in predictions) or any(value is None for value in probability_rows):
        raise RuntimeError("grouped cross-validation did not produce every held-out prediction")

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    confusion = {truth: {predicted: 0 for predicted in resolved_classes} for truth in resolved_classes}
    for truth, predicted in zip(resolved_labels.tolist(), predictions):
        confusion[str(truth)][str(predicted)] += 1
    for class_name in resolved_classes:
        tp = confusion[class_name][class_name]
        fp = sum(confusion[truth][class_name] for truth in resolved_classes if truth != class_name)
        fn = sum(confusion[class_name][predicted] for predicted in resolved_classes if predicted != class_name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[class_name].values())
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)

    class_indices = {name: index for index, name in enumerate(resolved_classes)}
    brier_values: list[float] = []
    confidences: list[float] = []
    correctness: list[float] = []
    for truth, predicted, probabilities in zip(resolved_labels.tolist(), predictions, probability_rows):
        assert probabilities is not None
        target = np.zeros(len(resolved_classes), dtype=np.float64)
        target[class_indices[str(truth)]] = 1.0
        vector = np.asarray([probabilities[name] for name in resolved_classes], dtype=np.float64)
        brier_values.append(float(np.mean((vector - target) ** 2)))
        confidences.append(float(max(vector)))
        correctness.append(1.0 if predicted == truth else 0.0)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        members = [index for index, value in enumerate(confidences) if lower <= value < upper or (upper >= 1.0 and value == 1.0)]
        if not members:
            continue
        average_confidence = sum(confidences[index] for index in members) / len(members)
        average_accuracy = sum(correctness[index] for index in members) / len(members)
        ece += len(members) / len(confidences) * abs(average_confidence - average_accuracy)
    return {
        "group_exclusive": True,
        "group_count": len(unique_groups),
        "fold_count": fold_count,
        "sample_count": len(resolved_labels),
        "macro_f1": sum(f1_values) / len(f1_values),
        "unknown_recall": per_class.get("unknown", {}).get("recall"),
        "brier_score": sum(brier_values) / len(brier_values),
        "expected_calibration_error": ece,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "folds": folds,
    }

@dataclass(frozen=True, slots=True)
class ActionGatePrediction:
    action_probabilities: dict[str, float]
    predicted_action: str
    action_confidence: float
    action_state: str
    supported_view: bool | None
    equipment_context: str
    switch_candidate_since_ms: int | None
    action_model_version: str
    action_model_hash: str
    action_source: str
    stale: bool
    switch_committed: bool
    switch_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "action_probabilities": dict(self.action_probabilities),
            "predicted_action": self.predicted_action,
            "action_confidence": self.action_confidence,
            "action_state": self.action_state,
            "supported_view": self.supported_view,
            "equipment_context": self.equipment_context,
            "switch_candidate_since_ms": self.switch_candidate_since_ms,
            "action_model_version": self.action_model_version,
            "action_model_hash": self.action_model_hash,
            "action_source": self.action_source,
            "stale": self.stale,
            "switch_committed": self.switch_committed,
            "switch_reason": self.switch_reason,
        }


class SwitchProtectedActionGate:
    """Hysteresis, minimum-duration and cooldown around model probabilities."""

    def __init__(self, contract: ActionGatingContract, *, model_version: str, model_hash: str) -> None:
        self.contract = contract
        self.model_version = model_version
        self.model_hash = model_hash
        self.current_action = "unknown"
        self._candidate: str | None = None
        self._candidate_since_ms: int | None = None
        self._low_since_ms: int | None = None
        self._last_switch_ms: int | None = None

    def reset(self) -> None:
        self.current_action = "unknown"
        self._candidate = None
        self._candidate_since_ms = None
        self._low_since_ms = None
        self._last_switch_ms = None

    def _prediction(
        self,
        probabilities: Mapping[str, float],
        *,
        timestamp_ms: int,
        supported_view: bool | None,
        equipment_context: str,
        stale: bool,
        action_source: str,
        switch_committed: bool,
        switch_reason: str,
    ) -> ActionGatePrediction:
        confidence = float(probabilities.get(self.current_action, 0.0))
        action_state = (
            "setup"
            if self.current_action in {"unknown", "idle"}
            and self._candidate not in {None, "unknown", "idle", "transition"}
            else "idle" if self.current_action == "idle"
            else "transition" if self.current_action == "transition"
            else "unknown" if self.current_action == "unknown"
            else "active"
        )
        return ActionGatePrediction(
            action_probabilities={name: float(probabilities.get(name, 0.0)) for name in self.contract.classes},
            predicted_action=self.current_action,
            action_confidence=confidence,
            action_state=action_state,
            supported_view=supported_view,
            equipment_context=equipment_context,
            switch_candidate_since_ms=self._candidate_since_ms,
            action_model_version=self.model_version,
            action_model_hash=self.model_hash,
            action_source=action_source,
            stale=stale,
            switch_committed=switch_committed,
            switch_reason=switch_reason,
        )

    def update(
        self,
        probabilities: Mapping[str, float],
        *,
        timestamp_ms: int,
        supported_view: bool | None = None,
        equipment_context: str = "unknown",
        stale: bool = False,
        manual_override: str | None = None,
    ) -> ActionGatePrediction:
        resolved = {name: max(0.0, _finite_float(probabilities.get(name))) for name in self.contract.classes}
        total = sum(resolved.values())
        if total > 0:
            resolved = {name: value / total for name, value in resolved.items()}
        else:
            resolved["unknown"] = 1.0
        if manual_override is not None:
            if manual_override not in self.contract.classes:
                raise ValueError(f"unsupported manual action override: {manual_override}")
            self.current_action = manual_override
            self._candidate = None
            self._candidate_since_ms = None
            self._low_since_ms = None
            return self._prediction(
                {name: 1.0 if name == manual_override else 0.0 for name in self.contract.classes},
                timestamp_ms=timestamp_ms,
                supported_view=supported_view,
                equipment_context=equipment_context,
                stale=False,
                action_source="manual",
                switch_committed=True,
                switch_reason="manual_override",
            )
        if stale:
            self.current_action = "unknown"
            self._candidate = None
            self._candidate_since_ms = None
            self._low_since_ms = None
            return self._prediction(
                resolved,
                timestamp_ms=timestamp_ms,
                supported_view=supported_view,
                equipment_context=equipment_context,
                stale=True,
                action_source="auto_shadow",
                switch_committed=False,
                switch_reason="stale_pose",
            )

        ranked = sorted(resolved.items(), key=lambda item: (-item[1], item[0]))
        top_action, top_confidence = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_confidence - runner_up
        raw_action = top_action
        if top_confidence < self.contract.unknown_confidence or margin < self.contract.minimum_margin:
            raw_action = "unknown"

        now = int(timestamp_ms)
        if raw_action == self.current_action:
            self._candidate = None
            self._candidate_since_ms = None
            self._low_since_ms = None
            return self._prediction(
                resolved,
                timestamp_ms=now,
                supported_view=supported_view,
                equipment_context=equipment_context,
                stale=False,
                action_source="auto_shadow",
                switch_committed=False,
                switch_reason="stable",
            )

        current_confidence = resolved.get(self.current_action, 0.0)
        if raw_action == "unknown" and self.current_action != "unknown":
            if current_confidence >= self.contract.exit_confidence:
                self._low_since_ms = None
                return self._prediction(
                    resolved,
                    timestamp_ms=now,
                    supported_view=supported_view,
                    equipment_context=equipment_context,
                    stale=False,
                    action_source="auto_shadow",
                    switch_committed=False,
                    switch_reason="exit_hysteresis",
                )
            if self._low_since_ms is None:
                self._low_since_ms = now
            if now - self._low_since_ms < self.contract.exit_duration_ms:
                return self._prediction(
                    resolved,
                    timestamp_ms=now,
                    supported_view=supported_view,
                    equipment_context=equipment_context,
                    stale=False,
                    action_source="auto_shadow",
                    switch_committed=False,
                    switch_reason="exit_duration",
                )
            if self._last_switch_ms is not None and now - self._last_switch_ms < self.contract.switch_cooldown_ms:
                return self._prediction(
                    resolved,
                    timestamp_ms=now,
                    supported_view=supported_view,
                    equipment_context=equipment_context,
                    stale=False,
                    action_source="auto_shadow",
                    switch_committed=False,
                    switch_reason="cooldown",
                )
            self.current_action = "unknown"
            self._last_switch_ms = now
            self._candidate = None
            self._candidate_since_ms = None
            self._low_since_ms = None
            return self._prediction(
                resolved,
                timestamp_ms=now,
                supported_view=supported_view,
                equipment_context=equipment_context,
                stale=False,
                action_source="auto_shadow",
                switch_committed=True,
                switch_reason="exit_committed",
            )
        elif top_confidence < self.contract.enter_confidence:
            return self._prediction(
                resolved,
                timestamp_ms=now,
                supported_view=supported_view,
                equipment_context=equipment_context,
                stale=False,
                action_source="auto_shadow",
                switch_committed=False,
                switch_reason="enter_confidence",
            )

        if self._candidate != raw_action:
            self._candidate = raw_action
            self._candidate_since_ms = now
            return self._prediction(
                resolved,
                timestamp_ms=now,
                supported_view=supported_view,
                equipment_context=equipment_context,
                stale=False,
                action_source="auto_shadow",
                switch_committed=False,
                switch_reason="candidate_started",
            )
        if self._candidate_since_ms is None or now - self._candidate_since_ms < self.contract.enter_duration_ms:
            return self._prediction(
                resolved,
                timestamp_ms=now,
                supported_view=supported_view,
                equipment_context=equipment_context,
                stale=False,
                action_source="auto_shadow",
                switch_committed=False,
                switch_reason="minimum_duration",
            )
        if self._last_switch_ms is not None and now - self._last_switch_ms < self.contract.switch_cooldown_ms:
            return self._prediction(
                resolved,
                timestamp_ms=now,
                supported_view=supported_view,
                equipment_context=equipment_context,
                stale=False,
                action_source="auto_shadow",
                switch_committed=False,
                switch_reason="cooldown",
            )

        self.current_action = raw_action
        self._last_switch_ms = now
        self._candidate = None
        self._candidate_since_ms = None
        self._low_since_ms = None
        return self._prediction(
            resolved,
            timestamp_ms=now,
            supported_view=supported_view,
            equipment_context=equipment_context,
            stale=False,
            action_source="auto_shadow",
            switch_committed=True,
            switch_reason="committed",
        )


class AutoActionShadowRuntime:
    """Offline/realtime sidecar that never changes the formal action analyzer."""

    def __init__(self, model: LogisticActionModel, contract: ActionGatingContract) -> None:
        if tuple(model.classes) != tuple(contract.classes):
            raise ValueError("action model classes do not match the contract")
        self.model = model
        self.contract = contract
        self.window = ActionFeatureWindow(contract)
        self.guard = SwitchProtectedActionGate(
            contract,
            model_version=model.model_version,
            model_hash=model.model_hash,
        )

    def reset(self) -> None:
        self.window.reset()
        self.guard.reset()

    def update(
        self,
        features: Mapping[str, object] | None,
        *,
        timestamp_ms: int,
        stale: bool = False,
        supported_view: bool | None = None,
        equipment_context: str = "unknown",
    ) -> dict[str, object]:
        if features is None or stale:
            probabilities = {name: 1.0 if name == "unknown" else 0.0 for name in self.contract.classes}
            return self.guard.update(
                probabilities,
                timestamp_ms=timestamp_ms,
                supported_view=supported_view,
                equipment_context=equipment_context,
                stale=stale,
            ).as_dict()
        self.window.update(features, timestamp_ms=timestamp_ms)
        if not self.window.ready:
            probabilities = {name: 1.0 if name == "unknown" else 0.0 for name in self.contract.classes}
            output = self.guard.update(
                probabilities,
                timestamp_ms=timestamp_ms,
                supported_view=supported_view,
                equipment_context=equipment_context,
            ).as_dict()
            output["switch_reason"] = "window_warming"
            output["window_frame_count"] = self.window.frame_count
            return output
        output = self.guard.update(
            self.model.predict_proba(self.window.vector()),
            timestamp_ms=timestamp_ms,
            supported_view=supported_view,
            equipment_context=equipment_context,
        ).as_dict()
        output["window_frame_count"] = self.window.frame_count
        return output


__all__ = [
    "ACTION_FEATURE_NAMES",
    "ACTION_FEATURE_SCHEMA_VERSION",
    "ActionFeatureWindow",
    "ActionGatePrediction",
    "AutoActionShadowRuntime",
    "BASE_FEATURE_NAMES",
    "LogisticActionModel",
    "SwitchProtectedActionGate",
    "grouped_cross_validate_logistic",
]
