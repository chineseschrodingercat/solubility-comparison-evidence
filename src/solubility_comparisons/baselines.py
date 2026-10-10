"""Frozen v1.1 policy behavior with explicit numerical and selector roles."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import ContractError, Prediction

PHASE_FAMILIES = frozenset(
    {"terminal_phase_after_competitive_slurry", "equilibrium_phase"}
)
PHASE_PROTOCOLS = frozenset(
    {
        "equilibrium_static_solubility",
        "equilibrium_slurry_solubility",
        "equilibrium_solid_liquid",
        "static_equilibrium",
        "competitive_slurry",
    }
)
EQUILIBRATION_CLAIMS = frozenset({"claimed_equilibrium", "documented_equilibrium"})
BASELINE_POLICIES = (
    "numerical_only",
    "numerical_agreement",
    "same_temperature_source",
    "metadata_gate",
    "common_source_representation",
    "restricted_I_II",
)


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ContractError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ContractError(f"{label} must be positive")
    return float(value)


def numerical_answer(
    task: Mapping[str, Any], restricted: bool = False
) -> tuple[dict[str, float] | None, float | None, str]:
    features, family = task["features"], task["question_family"]
    if family in PHASE_FAMILIES:
        values = features.get("values", {})
        if restricted and not all(key in values for key in ("I", "II")):
            return (
                None,
                None,
                "I/II restricted universe absent; DEV-only baseline unavailable",
            )
        keys = ["I", "II"] if restricted else list(task["context"]["candidate_ids"])
        if any(values.get(key) is None for key in keys):
            return None, None, "no supported native concentration for every candidate"
        for key in keys:
            _number(values[key], f"concentration {key}", positive=True)
        ordered = sorted(keys, key=lambda key: (values[key], key))
        low = values[ordered[0]]
        ties = [key for key in ordered if values[key] == low]
        gap = values[ordered[1]] / low - 1 if len(ordered) > 1 else 0
        return (
            {key: 1 / len(ties) for key in ties},
            float(gap),
            "relative minimum gap; not calibrated uncertainty",
        )
    if family == "reported_native_concentration_order":
        delta = _number(features["pointwise_delta_logC"], "pointwise contrast")
        selected = "A" if delta > 0 else "B" if delta < 0 else "TIE"
        return (
            {selected: 1.0},
            abs(delta),
            "absolute fixed-fit contrast; not calibrated uncertainty",
        )
    return None, None, "no admissible numeric feature for this question"


def predict(task: Mapping[str, Any], policy: str) -> Prediction:
    if policy not in BASELINE_POLICIES:
        raise ContractError(f"Unknown policy {policy!r}")
    features, family = task["features"], task["question_family"]
    probabilities, confidence, meaning = numerical_answer(
        task, policy == "restricted_I_II"
    )
    reason = "accepted"
    intervention = "selector_with_fixed_numerical_answer"
    if (
        policy == "restricted_I_II"
        and family != "terminal_phase_after_competitive_slurry"
    ):
        probabilities, confidence, reason = (
            None,
            None,
            "restricted candidate baseline applies only to SMZ DEV",
        )
    elif policy == "numerical_agreement":
        dispersion = features.get("numerical_agreement_dispersion")
        if (
            not features.get("numerical_agreement_object_verified", False)
            or dispersion is None
        ):
            probabilities, confidence = None, None
            reason = "agreement undefined without verified repeats of the stated numerical object"
        else:
            dispersion = _number(dispersion, "same-object dispersion")
            if dispersion < 0:
                raise ContractError("Dispersion cannot be negative")
            confidence, meaning = (
                -dispersion,
                "negative same-object numerical dispersion; no fitted threshold or floor",
            )
    elif policy == "same_temperature_source":
        required = (
            "identity_matched",
            "quantity_basis_verified",
            "same_query_temperature",
            "within_native_shared_range",
        )
        if not all(features.get(key, False) for key in required):
            probabilities, confidence, reason = (
                None,
                None,
                "no matched identity/quantity/common native temperature/range",
            )
        elif family == "reported_native_concentration_order":
            if (
                features.get("common_source_delta_logC") is None
                or features.get("n_common_groups", 0) < 1
            ):
                probabilities, confidence, reason = None, None, "no common source group"
        elif family in PHASE_FAMILIES and not features.get(
            "source_support_common", False
        ):
            probabilities, confidence, reason = (
                None,
                None,
                "candidate concentrations lack common source support",
            )
    elif policy == "common_source_representation":
        intervention = "numerical_representation_change"
        if family == "reported_native_concentration_order":
            delta = features.get("common_source_delta_logC")
            if delta is None or features.get("n_common_groups", 0) < 1:
                probabilities, confidence, reason = None, None, "no common source group"
            else:
                delta = _number(delta, "common-source contrast")
                probabilities = {"A" if delta > 0 else "B" if delta < 0 else "TIE": 1.0}
                confidence, meaning = (
                    abs(delta),
                    "absolute common-source contrast; representation intervention, not selector alone",
                )
    elif policy == "metadata_gate":
        protocol = features.get("measurement_protocol")
        protocol = protocol.strip().lower() if isinstance(protocol, str) else None
        if family in PHASE_FAMILIES and protocol not in PHASE_PROTOCOLS:
            probabilities, confidence = None, None
            reason = "phase inference requires an explicitly appropriate protocol; missing/unknown/dynamic assay is unresolved"
        elif (
            family in PHASE_FAMILIES
            and features.get("equilibration_claim_status") not in EQUILIBRATION_CLAIMS
        ):
            probabilities, confidence = None, None
            reason = "phase inference requires claimed/documented equilibration; missing/unknown claim is unresolved"
        elif not all(
            features.get(key, False)
            for key in (
                "identity_matched",
                "quantity_basis_verified",
                "same_query_temperature",
                "within_native_shared_range",
            )
        ):
            probabilities, confidence, reason = (
                None,
                None,
                "identity/quantity/T/range gate unresolved",
            )
    if probabilities is None and reason == "accepted":
        reason = "unsupported question or unresolved numeric feature"
    if policy == "restricted_I_II":
        intervention = "candidate_universe_change_DEV_sanity"
    return Prediction(
        task_id=task["task_id"],
        policy=policy,
        abstain=probabilities is None,
        answer_probabilities=probabilities,
        confidence=confidence,
        confidence_meaning=meaning,
        reason=reason,
        intervention_type=intervention,
        implementation_version="1.1",
    )
