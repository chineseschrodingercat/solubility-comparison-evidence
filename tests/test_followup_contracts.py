from copy import deepcopy

import pytest

from solubility_comparisons.answer_sets import Interpretation, finite_answer_set
from solubility_comparisons.baselines import predict
from solubility_comparisons.contracts import ContractError
from solubility_comparisons.evaluation import evaluate


def test_duplicate_member_cannot_supply_finite_completeness():
    duplicated = Interpretation("same", {"A": 1.0, "B": 2.0})
    with pytest.raises(ContractError, match="unique"):
        finite_answer_set([duplicated, duplicated], ["A", "B"], complete=True)


def test_primary_provenance_order_is_resolved_without_efficacy_score(phase_task, judge):
    judge["answer_status"] = "resolved_primary_reported_point_order"
    report = evaluate(
        [phase_task],
        [judge],
        {"numerical_only": [predict(phase_task, "numerical_only")]},
    )
    row = report.tables["policy_scope_summary"].iloc[0]
    assert row.n_unresolved_answers == 0
    assert row.n_scoreable == 0


def test_explicit_independent_primary_reference_can_be_point_scored(phase_task, judge):
    judge.update(
        answer_status="resolved_primary_reported_point_order",
        evaluation_role="reference",
    )
    report = evaluate(
        [phase_task],
        [judge],
        {"numerical_only": [predict(phase_task, "numerical_only")]},
    )
    assert report.tables["policy_scope_summary"].iloc[0].n_scoreable == 1
