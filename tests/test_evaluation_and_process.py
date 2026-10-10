import math

import pandas as pd
import pytest

from solubility_comparisons.baselines import predict
from solubility_comparisons.contracts import ContractError
from solubility_comparisons.evaluation import coverage_budget_selection, evaluate
from solubility_comparisons.process_endpoints import marginal_wilson_interval
from solubility_comparisons.process_screen import (
    pillong_cells,
    screen_policy_comparison,
)


def test_full_pool_equal_budget_can_select_disjoint_questions():
    a = pd.DataFrame(dict(task_id=["a", "b"], confidence=[2.0, 1.0], error=[0.0, 1.0]))
    b = pd.DataFrame(dict(task_id=["c", "d"], confidence=[2.0, 1.0], error=[1.0, 0.0]))
    chosen_a, chosen_b, info = coverage_budget_selection(
        a, b, 4, "full_scoreable_scope", 0.5
    )
    assert info["available"] and info["k"] == 2 and info["n_shared_selected_ids"] == 0
    assert set(chosen_a.task_id) == {"a", "b"} and set(chosen_b.task_id) == {"c", "d"}
    assert not coverage_budget_selection(a, b, 4, "shared_accepted_pool", 0.5)[2][
        "available"
    ]


def test_coverage_budget_cannot_exceed_either_selector_availability():
    a = pd.DataFrame(dict(task_id=["a"], confidence=[1.0]))
    b = pd.DataFrame(dict(task_id=["a", "b"], confidence=[2.0, 1.0]))
    assert not coverage_budget_selection(a, b, 2, "full_scoreable_scope", 1.0)[2][
        "available"
    ]


def test_all_abstention_has_undefined_risk(phase_task, judge):
    phase_task["features"]["measurement_protocol"] = None
    report = evaluate(
        [phase_task], [judge], {"metadata_gate": [predict(phase_task, "metadata_gate")]}
    )
    row = report.tables["policy_scope_summary"].iloc[0]
    assert row.coverage == 0 and math.isnan(row.selective_expected_error)


def test_unknown_reference_uncertainty_cannot_be_scored_as_resolved(phase_task, judge):
    report = evaluate(
        [phase_task],
        [judge],
        {"numerical_only": [predict(phase_task, "numerical_only")]},
        mode="uncertainty_resolved",
    )
    assert report.tables["policy_scope_summary"].iloc[0].n_scoreable == 0


def test_duplicate_task_ids_are_not_silently_overwritten(phase_task, judge):
    with pytest.raises(ContractError, match="Duplicate"):
        evaluate(
            [phase_task, phase_task],
            [judge],
            {"numerical_only": [predict(phase_task, "numerical_only")]},
        )


def test_missing_screen_cells_are_not_crystallization_failures():
    wide = pd.DataFrame(
        [
            dict(
                PID="toy",
                SMILES="C",
                Cluster=1,
                Purity=100,
                **{
                    "S-a": "RS",
                    "C-a": "DR",
                    "S-b": "KS",
                    "C-b": "XX",
                    "S-c": None,
                    "C-c": None,
                },
            )
        ]
    )
    cells = pillong_cells(wide)
    policies, _, summary = screen_policy_comparison(cells)
    assert summary["observed_cells"] == 2
    assert policies.iloc[0].top_uniform_success == 0
    assert policies.iloc[0].all_uniform_success == 0.5


@pytest.mark.parametrize("successes", [0, 10])
def test_marginal_wilson_handles_boundary_counts(successes):
    lower, upper = marginal_wilson_interval(successes, 10)
    assert -1e-15 <= lower <= upper <= 1 + 1e-15


def test_impossible_process_count_is_rejected():
    with pytest.raises(ContractError):
        marginal_wilson_interval(11, 10)


@pytest.mark.parametrize("successes", [True, 1.5])
def test_noninteger_process_count_is_rejected(successes):
    with pytest.raises(ContractError):
        marginal_wilson_interval(successes, 10)
