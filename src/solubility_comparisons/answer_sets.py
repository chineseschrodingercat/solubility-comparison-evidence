"""Finite numerical interpretation sets, distinct from physical completeness."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any, Literal, Mapping, Sequence

from .baselines import PHASE_FAMILIES, predict
from .contracts import ContractError, Prediction

Direction = Literal["lower", "higher"]
GapScale = Literal["relative_positive_quantity", "absolute_log_contrast"]


@dataclass(frozen=True)
class Interpretation:
    identifier: str
    values: Mapping[str, float]


def finite_answer_set(
    states: Sequence[Interpretation],
    candidates: Sequence[str],
    *,
    direction: Direction = "lower",
    complete: bool = True,
    gap_scale: GapScale = "relative_positive_quantity",
) -> dict[str, Any]:
    """Union answers without treating interpretation frequencies as probabilities.

    ``complete`` is a caller declaration about this finite list. It is never a
    claim about all physical explanations, missing phases or covariance.
    """
    if direction not in {"lower", "higher"}:
        raise ContractError("Direction must be lower or higher")
    if gap_scale not in {"relative_positive_quantity", "absolute_log_contrast"}:
        raise ContractError("Unknown representation gap scale")
    if len({state.identifier for state in states}) != len(states):
        raise ContractError("Interpretation identifiers must be unique")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ContractError("Candidates must be nonempty and unique")
    union: set[str] = set()
    margins: list[float] = []
    outputs: list[dict[str, Any]] = []
    for state in states:
        values = state.values
        if set(values) != set(candidates) or not all(
            not isinstance(value, bool) and math.isfinite(value)
            for value in values.values()
        ):
            raise ContractError(
                "Every interpretation must contain finite values for exactly the candidates"
            )
        extreme = min(values.values()) if direction == "lower" else max(values.values())
        winners = sorted(key for key, value in values.items() if value == extreme)
        ordered = sorted(values.values(), reverse=direction == "higher")
        if gap_scale == "absolute_log_contrast":
            if set(candidates) != {"A", "B"}:
                raise ContractError("Absolute log-contrast gap requires A and B")
            margin = abs(values["A"] - values["B"])
        else:
            if min(values.values()) <= 0:
                raise ContractError(
                    "Relative-quantity gaps require positive quantities"
                )
            margin = (
                abs(ordered[1] - ordered[0]) / abs(ordered[0])
                if len(ordered) > 1
                else 0.0
            )
        union.update(winners)
        margins.append(float(margin))
        outputs.append(
            dict(
                interpretation_id=state.identifier,
                values=dict(values),
                numerical_minimum_or_maximum=winners,
            )
        )
    edges = []
    for a, b in combinations(candidates, 2):
        if states and all(state.values[a] < state.values[b] for state in states):
            edges.append([a, "<", b])
        elif states and all(state.values[a] > state.values[b] for state in states):
            edges.append([b, "<", a])
    singleton = bool(complete and states and len(union) == 1)
    return dict(
        n_enumerated_interpretations=len(states),
        enumeration_complete=bool(complete),
        supported_numerical_answer_set=sorted(union),
        invariant_numeric_answer=next(iter(union)) if singleton else None,
        strict_partial_order_edges=edges if complete else [],
        minimum_representation_gap=min(margins) if singleton else None,
        representation_gap_scale=gap_scale,
        interpretations=outputs,
        scope="finite declared numerical interpretations only; not all physical interpretations",
    )


def observed_gate(observations: Mapping[str, Any], candidates: Sequence[str]) -> bool:
    """Recompute from safe fields; a cached 'satisfied' field cannot bypass masks."""
    if not observations["applicable"]:
        return True
    records = observations.get("candidates", {})
    return all(
        key in records
        and records[key].get("clear_aliquot_before_filtration") is True
        and records[key].get("separation_documented") is True
        for key in candidates
    )


def _unknown_observations(task: Mapping[str, Any]) -> dict[str, Any]:
    if task["question_family"] not in PHASE_FAMILIES:
        return dict(
            requirement="phase-specific observed gate not applicable to reported-native order",
            applicable=False,
            satisfied=True,
            candidates={},
            source_locator=None,
        )
    return dict(
        applicable=True,
        satisfied=False,
        candidates={
            key: dict(clear_aliquot_before_filtration=None, separation_documented=None)
            for key in task["context"]["candidate_ids"]
        },
        requirement="all candidates explicitly clear before sampling and separated",
        source_locator=None,
        reason="O requirement not documented; dynamic endpoint is already rejected by D",
    )


def _renamed(prediction: Prediction, name: str) -> Prediction:
    result = deepcopy(prediction)
    result["policy"], result["implementation_version"] = (
        name,
        "development_answer_sets_v1",
    )
    return result


def _abstain(prediction: Prediction, reason: str) -> Prediction:
    result = deepcopy(prediction)
    result.update(
        abstain=True, answer_probabilities=None, confidence=None, reason=reason
    )
    return result


def predict_finite_policies(
    task: Mapping[str, Any], answer_set: Mapping[str, Any]
) -> list[Prediction]:
    numerical = _renamed(predict(task, "numerical_only"), "numerical")
    declared = _renamed(predict(task, "metadata_gate"), "declared_D")
    observations = deepcopy(task.get("observed_assay_O", _unknown_observations(task)))
    # Applicability is a target requirement, not a caller-controlled shortcut.
    observations["applicable"] = task["question_family"] in PHASE_FAMILIES
    observations["satisfied"] = observed_gate(
        observations, task["context"]["candidate_ids"]
    )
    observed = _renamed(declared, "observed_D_plus_O")
    if not declared["abstain"] and not observations["satisfied"]:
        observed = _abstain(
            observed,
            "concentration-assay observed O is incomplete/adverse: "
            + observations.get("reason", "unknown"),
        )
    selected = answer_set["invariant_numeric_answer"]
    stable = Prediction(
        task_id=task["task_id"],
        policy="finite_S_numerical_only",
        abstain=selected is None,
        answer_probabilities={selected: 1.0} if selected is not None else None,
        confidence=(
            answer_set["minimum_representation_gap"] if selected is not None else None
        ),
        confidence_meaning="minimum gap across finite representations; neither calibrated uncertainty nor CI",
        reason=(
            "finite numerical singleton"
            if selected is not None
            else "non-singleton or incomplete finite numerical set"
        ),
        intervention_type="numerical_representation_invariance_development",
        implementation_version="development_answer_sets_v1",
    )
    stable_declared, stable_observed = _renamed(stable, "finite_S_plus_D"), _renamed(
        stable, "finite_S_plus_D_plus_O"
    )
    if declared["abstain"]:
        stable_declared = _abstain(
            stable_declared, "D blocks physical use: " + declared["reason"]
        )
    if observed["abstain"]:
        stable_observed = _abstain(
            stable_observed, "D+O blocks physical use: " + observed["reason"]
        )
    return [numerical, declared, observed, stable, stable_declared, stable_observed]


def predict_safe_view(
    task: Mapping[str, Any], *, profile: str = "declared"
) -> tuple[dict[str, Any], list[Prediction]]:
    """Replay provided interpretations, without opening any referee or raw table.

    Generic use requires ``finite_set_spec``. The named compatibility profile
    describes the archived development enumeration; it introduces no data.
    """
    states = [
        Interpretation(row["interpretation_id"], row["values"])
        for row in task.get("finite_numerical_interpretations", [])
    ]
    if profile == "declared":
        spec = task.get("finite_set_spec")
        if not isinstance(spec, Mapping):
            raise ContractError(
                "Provide finite_set_spec, or explicitly request frozen-development-v1"
            )
        expected, direction, gap = (
            spec["expected_count"],
            spec["direction"],
            spec["gap_scale"],
        )
        candidates = spec.get("candidates", task["context"]["candidate_ids"])
    elif profile == "frozen-development-v1":
        if task.get("method_view") in {"IT", "PT"}:
            expected = 2 if task["method_view"] == "IT" else 1
        else:
            expected = (
                8
                if task["resource_id"] == "SMZ_EXPOSED"
                else 2 if task["resource_id"] == "SOLPROP_EXPOSED" else 0
            )
        order = task["resource_id"] == "SOLPROP_EXPOSED"
        direction, gap = (
            ("higher", "absolute_log_contrast")
            if order
            else ("lower", "relative_positive_quantity")
        )
        candidates = ["A", "B"] if order else task["context"]["candidate_ids"]
    else:
        raise ContractError(f"Unknown finite-set profile {profile!r}")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        raise ContractError(
            "Expected interpretation count must be a nonnegative integer"
        )
    answer_set = finite_answer_set(
        states,
        candidates,
        direction=direction,
        gap_scale=gap,
        complete=expected > 0 and len(states) == expected,
    )
    if task.get("method_view") in {"IT", "PT"} and any(
        value is None for value in task["features"]["values"].values()
    ):
        # Preserve archived abstention payloads, without replacing the input itself.
        dummy = deepcopy(task)
        dummy["features"]["values"] = {
            key: 1.0 for key in task["context"]["candidate_ids"]
        }
        predictions = [
            _abstain(
                row,
                "no supported common candidate values at the target temperature; no extrapolation",
            )
            for row in predict_finite_policies(dummy, answer_set)
        ]
    else:
        predictions = predict_finite_policies(task, answer_set)
    if task["resource_id"] == "SOLPROP_UNRESOLVED_EXPOSED":
        predictions = [
            _abstain(
                row,
                "specified phase row mapping unresolved; finite state-free row choices do not close this question",
            )
            for row in predictions
        ]
    return answer_set, predictions
