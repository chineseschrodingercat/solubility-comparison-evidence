from copy import deepcopy
import math

import pytest

from solubility_comparisons.baselines import predict
from solubility_comparisons.contracts import ContractError, validate_task
from solubility_comparisons.answer_sets import (
    Interpretation,
    finite_answer_set,
    predict_finite_policies,
)


@pytest.mark.parametrize(
    "field", ["accepted_answers", "observed_terminal_phase", "reference_delta_logC"]
)
def test_nested_referee_fields_are_rejected(phase_task, field):
    phase_task["features"]["nested"] = [{field: "hidden"}]
    with pytest.raises(ContractError, match="Referee field"):
        validate_task(phase_task)


def test_developed_record_cannot_be_relabelled_holdout(phase_task):
    phase_task.update(
        stage="EXTERNAL_HOLDOUT",
        exposure_status="external_holdout",
        exposure_history=dict(used_to_develop_metadata_rule=True),
    )
    with pytest.raises(ContractError, match="exposed/developed"):
        validate_task(phase_task)


@pytest.mark.parametrize("protocol", [None, "unknown", "dynamic_clear_point"])
@pytest.mark.parametrize(
    "family", ["equilibrium_phase", "terminal_phase_after_competitive_slurry"]
)
def test_unknown_or_dynamic_protocol_never_creates_phase_answer(
    phase_task, protocol, family
):
    phase_task["question_family"] = family
    phase_task["features"]["measurement_protocol"] = protocol
    assert predict(phase_task, "metadata_gate")["abstain"]


def test_distinct_phases_are_not_verified_numerical_repeats(phase_task):
    phase_task["features"].update(
        numerical_agreement_object_verified=False, numerical_agreement_dispersion=0.0
    )
    assert predict(phase_task, "numerical_agreement")["abstain"]


def test_same_source_selector_does_not_change_pointwise_representation(phase_task):
    phase_task["question_family"] = "reported_native_concentration_order"
    phase_task["context"]["candidate_ids"] = ["A", "B", "TIE"]
    phase_task["features"].update(
        pointwise_delta_logC=0.2, common_source_delta_logC=-0.1, n_common_groups=1
    )
    selector = predict(phase_task, "same_temperature_source")
    changed = predict(phase_task, "common_source_representation")
    assert selector["answer_probabilities"] == {"A": 1.0}
    assert changed["answer_probabilities"] == {"B": 1.0}


def test_unknown_concentrations_abstain_instead_of_sorting_null(phase_task):
    phase_task["features"]["values"]["II"] = None
    assert predict(phase_task, "numerical_only")["abstain"]


def test_observed_gate_uses_assay_fields_not_cached_status(phase_task):
    states = finite_answer_set(
        [Interpretation("one", {"I": 2.0, "II": 3.0})], ["I", "II"]
    )
    phase_task["observed_assay_O"] = dict(
        applicable=False,
        satisfied=True,
        candidates={
            "I": dict(clear_aliquot_before_filtration=True, separation_documented=True),
            "II": dict(
                clear_aliquot_before_filtration=None, separation_documented=True
            ),
        },
    )
    result = {row["policy"]: row for row in predict_finite_policies(phase_task, states)}
    assert not result["declared_D"]["abstain"]
    assert result["observed_D_plus_O"]["abstain"]
    assert result["finite_S_plus_D_plus_O"]["abstain"]
