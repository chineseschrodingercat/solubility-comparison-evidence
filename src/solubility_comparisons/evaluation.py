"""Reference joins and coverage comparisons, grouped by the original question."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import ContractError, unique_index, validate_judge, validate_task

SCOPE_COLUMNS = ["resource_id", "question_family", "stage", "exposure_status"]
RESOLVED_STATUSES = frozenset(
    {
        "resolved_observation",
        "resolved_point_reference",
        "resolved_primary_reported_point_order",
    }
)


def is_accepted(prediction: Mapping[str, Any] | None) -> bool:
    return (
        prediction is not None
        and not prediction.get("abstain", False)
        and prediction.get("answer_probabilities") is not None
    )


def coverage_budget_selection(
    accepted_a: pd.DataFrame,
    accepted_b: pd.DataFrame,
    n_scoreable: int,
    denominator: str,
    fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Each full-pool selector may choose different IDs at the same budget.

    The shared-pool comparison is a separate identical-object control, not a
    measurement of selector benefit over the full scoreable scope.
    """
    if not 0 < fraction <= 1 or n_scoreable < 0:
        raise ContractError("Coverage fraction must be in (0,1] and scope nonnegative")
    common = sorted(set(accepted_a.task_id) & set(accepted_b.task_id))
    if denominator == "shared_accepted_pool":
        pool_a = accepted_a[accepted_a.task_id.isin(common)]
        pool_b = accepted_b[accepted_b.task_id.isin(common)]
        budget_n = len(common)
    elif denominator == "full_scoreable_scope":
        pool_a, pool_b, budget_n = accepted_a, accepted_b, n_scoreable
    else:
        raise ContractError(f"Unknown coverage denominator {denominator!r}")
    k = math.floor(fraction * budget_n)
    valid = k > 0 and k <= len(pool_a) and k <= len(pool_b)
    chosen_a = (
        pool_a.sort_values(["confidence", "task_id"], ascending=[False, True]).iloc[:k]
        if valid
        else pool_a.iloc[:0]
    )
    chosen_b = (
        pool_b.sort_values(["confidence", "task_id"], ascending=[False, True]).iloc[:k]
        if valid
        else pool_b.iloc[:0]
    )
    shared_selected = sorted(set(chosen_a.task_id) & set(chosen_b.task_id))
    info = dict(
        k=k,
        available=valid,
        pool_n=len(common),
        n_scoreable_scope=n_scoreable,
        policy_A_available_n=len(accepted_a),
        policy_B_available_n=len(accepted_b),
        budget_denominator=denominator,
        budget_fraction=fraction,
        n_shared_selected_ids=len(shared_selected),
        shared_selected_task_ids="|".join(shared_selected),
        answered_fraction_of_shared_pool=(
            k / len(common)
            if valid and denominator == "shared_accepted_pool"
            else np.nan
        ),
        answered_fraction_of_full_scope=(
            k / n_scoreable if valid and n_scoreable else np.nan
        ),
    )
    return chosen_a, chosen_b, info


def _check_prediction(
    prediction: Mapping[str, Any], task: Mapping[str, Any], policy: str
) -> None:
    if prediction.get("policy") != policy:
        raise ContractError("Prediction policy disagrees with its collection name")
    if not is_accepted(prediction):
        return
    probabilities = prediction["answer_probabilities"]
    if not probabilities or not all(
        isinstance(value, (int, float)) and 0 <= value <= 1
        for value in probabilities.values()
    ):
        raise ContractError("Invalid answer probabilities")
    if abs(sum(probabilities.values()) - 1) >= 1e-10:
        raise ContractError("Accepted probabilities must sum to one")
    if not set(probabilities) <= set(task["context"]["candidate_ids"]):
        raise ContractError("Prediction outside the task candidate universe")
    confidence = prediction.get("confidence")
    if confidence is None or not math.isfinite(confidence):
        raise ContractError("Accepted predictions require a finite ordering score")


@dataclass(frozen=True)
class EvaluationReport:
    tables: Mapping[str, pd.DataFrame]
    metadata: Mapping[str, Any]

    def write(self, directory: Path) -> None:
        import json

        directory.mkdir(parents=True, exist_ok=True)
        for name, table in self.tables.items():
            table.to_csv(directory / f"{name}.csv", index=False)
        (directory / "evaluation_contract_summary.json").write_text(
            json.dumps(dict(self.metadata), indent=2, allow_nan=False), encoding="utf-8"
        )


def evaluate(
    inputs: Sequence[Mapping[str, Any]],
    judges: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    mode: str = "point",
    budget_fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> EvaluationReport:
    """No fitting or threshold selection; unresolved references remain visible."""
    if mode not in {"point", "uncertainty_resolved"} or not predictions:
        raise ContractError("Provide policies and a valid evaluation mode")
    task_index = unique_index([validate_task(row) for row in inputs])
    judge_index = unique_index([validate_judge(row) for row in judges])
    if set(task_index) != set(judge_index):
        raise ContractError(
            "Task and judge registries must contain exactly the same IDs"
        )
    policies = {
        name: unique_index(list(rows)) for name, rows in sorted(predictions.items())
    }
    for policy, collection in policies.items():
        if not set(collection) <= set(task_index):
            raise ContractError("Prediction contains an unknown task ID")
        for task_id, prediction in collection.items():
            _check_prediction(prediction, task_index[task_id], policy)
    rows = []
    for task_id, task in task_index.items():
        judge = judge_index[task_id]
        resolved = (
            judge["answer_status"] in RESOLVED_STATUSES
            and judge.get("accepted_answers") is not None
        )
        if resolved and not set(judge["accepted_answers"]) <= set(
            task["context"]["candidate_ids"]
        ):
            raise ContractError("Reference outside the task candidate universe")
        role = judge.get(
            "evaluation_role",
            (
                "provenance_check"
                if judge["answer_status"] == "resolved_primary_reported_point_order"
                else "reference"
            ),
        )
        scoreable = (
            resolved
            and role == "reference"
            and (mode == "point" or judge.get("uncertainty_resolved_scoreable") is True)
        )
        for policy, collection in policies.items():
            prediction = collection.get(task_id)
            accepted = is_accepted(prediction)
            error = (
                1
                - sum(
                    prediction["answer_probabilities"].get(key, 0)
                    for key in judge["accepted_answers"]
                )
                if accepted and scoreable
                else np.nan
            )
            rows.append(
                dict(
                    task_id=task_id,
                    **{key: task[key] for key in SCOPE_COLUMNS},
                    entity_id=task["entity_id"],
                    source_group_id=task["source_group_id"],
                    policy=policy,
                    judge_status=judge["answer_status"],
                    scoreable=scoreable,
                    unresolved_answer=not resolved,
                    accepted=accepted,
                    confidence=prediction["confidence"] if accepted else np.nan,
                    error=error,
                    numerical_u_status=task["features"].get("numerical_u_status"),
                    judge_uncertainty_status=judge["uncertainty_status"],
                    intervention_type=(
                        prediction.get("intervention_type") if prediction else None
                    ),
                    reason=(
                        prediction.get("reason")
                        if prediction
                        else "MISSING_PREDICTION_ABSTAIN"
                    ),
                )
            )
    all_rows = pd.DataFrame(rows)
    summaries, prefixes, shared, budgets = [], [], [], []
    for key, group in all_rows.groupby(SCOPE_COLUMNS, dropna=False):
        scope = dict(zip(SCOPE_COLUMNS, key))
        n = len(group.loc[group.scoreable, "task_id"].unique())
        for policy, policy_rows in group.groupby("policy"):
            answered = policy_rows[
                policy_rows.scoreable & policy_rows.accepted
            ].sort_values(["confidence", "task_id"], ascending=[False, True])
            m = len(answered)
            summaries.append(
                dict(
                    **scope,
                    policy=policy,
                    n_tasks=len(policy_rows),
                    n_scoreable=n,
                    n_unresolved_answers=int(policy_rows.unresolved_answer.sum()),
                    n_predictions_on_unresolved=int(
                        (policy_rows.accepted & policy_rows.unresolved_answer).sum()
                    ),
                    n_accepted_scoreable=m,
                    coverage=m / n if n else np.nan,
                    selective_expected_error=(
                        float(answered.error.mean()) if m else np.nan
                    ),
                    n_errors=float(answered.error.sum()) if m else 0,
                    n_unique_entities=int(policy_rows.entity_id.nunique()),
                    n_source_groups=int(policy_rows.source_group_id.nunique()),
                    mode=mode,
                )
            )
            prefixes.append(
                dict(
                    **scope,
                    policy=policy,
                    k=0,
                    n_scoreable=n,
                    coverage=0 if n else np.nan,
                    selective_expected_error=np.nan,
                    selected_task_ids="",
                )
            )
            for k in range(1, m + 1):
                selected = answered.iloc[:k]
                prefixes.append(
                    dict(
                        **scope,
                        policy=policy,
                        k=k,
                        n_scoreable=n,
                        coverage=k / n,
                        selective_expected_error=float(selected.error.mean()),
                        selected_task_ids="|".join(selected.task_id),
                    )
                )
        for policy_a, policy_b in combinations(sorted(policies), 2):
            accepted_a = group[
                group.policy.eq(policy_a) & group.scoreable & group.accepted
            ]
            accepted_b = group[
                group.policy.eq(policy_b) & group.scoreable & group.accepted
            ]
            common = sorted(set(accepted_a.task_id) & set(accepted_b.task_id))
            identical_a, identical_b = (
                accepted_a[accepted_a.task_id.isin(common)],
                accepted_b[accepted_b.task_id.isin(common)],
            )
            shared.append(
                dict(
                    **scope,
                    policy_A=policy_a,
                    policy_B=policy_b,
                    n_shared_accepted_question_ids=len(common),
                    A_expected_error=(
                        float(identical_a.error.mean()) if common else np.nan
                    ),
                    B_expected_error=(
                        float(identical_b.error.mean()) if common else np.nan
                    ),
                    shared_task_ids="|".join(common),
                    mode=mode,
                )
            )
            for denominator in ("shared_accepted_pool", "full_scoreable_scope"):
                for fraction in budget_fractions:
                    selected_a, selected_b, info = coverage_budget_selection(
                        accepted_a, accepted_b, n, denominator, fraction
                    )
                    description = (
                        "shared accepted-ID pool only; representation/identical-object control, not full-pool selector benefit"
                        if denominator == "shared_accepted_pool"
                        else "each policy ranks its own full scoreable accepted set; same k, different selected IDs allowed"
                    )
                    budgets.append(
                        dict(
                            **scope,
                            policy_A=policy_a,
                            policy_B=policy_b,
                            comparison=description,
                            **info,
                            A_expected_error=(
                                float(selected_a.error.mean())
                                if info["available"]
                                else np.nan
                            ),
                            B_expected_error=(
                                float(selected_b.error.mean())
                                if info["available"]
                                else np.nan
                            ),
                            A_selected_task_ids="|".join(selected_a.task_id),
                            B_selected_task_ids="|".join(selected_b.task_id),
                            mode=mode,
                        )
                    )
    tables = dict(
        all_task_prediction_judge_rows=all_rows,
        policy_scope_summary=pd.DataFrame(summaries),
        risk_coverage_prefixes=pd.DataFrame(prefixes),
        same_question_id_comparisons=pd.DataFrame(shared),
        same_coverage_comparisons=pd.DataFrame(budgets),
    )
    for unit in ("entity_id", "source_group_id"):
        tables[f"{unit}_descriptive_aggregates"] = (
            all_rows[all_rows.scoreable & all_rows.accepted]
            .groupby(SCOPE_COLUMNS + ["policy", unit])
            .agg(n=("task_id", "size"), expected_error=("error", "mean"))
            .reset_index()
        )
    metadata = dict(
        implementation_version="1.1",
        mode=mode,
        total_distinct_tasks=len(task_index),
        resource_families=len(
            {(task["resource_id"], task["question_family"]) for task in inputs}
        ),
        no_cross_family_pooled_rate=True,
        unresolved_task_ids=[
            row["task_id"] for row in judges if row["answer_status"] == "unresolved"
        ],
        risk_not_calibrated=True,
        no_heldout_threshold_selection=True,
    )
    return EvaluationReport(tables, metadata)
