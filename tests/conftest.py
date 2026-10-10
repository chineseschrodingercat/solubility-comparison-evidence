from copy import deepcopy

import pytest


@pytest.fixture
def phase_task():
    return dict(
        schema_version=1,
        task_id="synthetic-phase",
        resource_id="synthetic",
        question_family="equilibrium_phase",
        stage="DEV",
        exposure_status="developed",
        entity_id="synthetic-entity",
        source_group_id="synthetic-source",
        context=dict(candidate_ids=["I", "II"]),
        features=dict(
            values={"I": 2.0, "II": 3.0},
            identity_matched=True,
            quantity_basis_verified=True,
            same_query_temperature=True,
            within_native_shared_range=True,
            source_support_common=True,
            measurement_protocol="equilibrium_static_solubility",
            equilibration_claim_status="claimed_equilibrium",
            numerical_u=None,
            numerical_u_status="UNKNOWN",
        ),
    )


@pytest.fixture
def judge():
    return dict(
        task_id="synthetic-phase",
        answer_status="resolved_observation",
        accepted_answers=["II"],
        uncertainty_status="UNKNOWN",
        uncertainty_resolved_scoreable=False,
    )
