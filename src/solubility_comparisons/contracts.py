"""Runtime contracts for physically separate predictor and referee records."""

from __future__ import annotations

from importlib.resources import files
import json
from typing import Any, Mapping, TypedDict, cast

from jsonschema import Draft202012Validator


class ContractError(ValueError):
    """An input violates the declared record or provenance contract."""


class TaskRecord(TypedDict):
    schema_version: int
    task_id: str
    resource_id: str
    question_family: str
    stage: str
    exposure_status: str
    entity_id: str
    source_group_id: str
    context: dict[str, Any]
    features: dict[str, Any]


class Prediction(TypedDict):
    task_id: str
    policy: str
    abstain: bool
    answer_probabilities: dict[str, float] | None
    confidence: float | None
    confidence_meaning: str
    reason: str
    intervention_type: str
    implementation_version: str


FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "answer",
        "accepted_answers",
        "observed_terminal_phase",
        "reference_delta",
        "reference_delta_logC",
        "C_code",
        "crystal_success",
        "legacy_agrees_with_observed",
        "same_universe_agrees_with_observed",
        "judge_locator",
        "raw_reference_variants",
    }
)


def guard_predictor_fields(value: Any, location: str = "$", /) -> None:
    """Reject referee fields recursively, including metadata nested in lists.

    This is an interface check; it cannot certify semantic independence of a
    residual-phase field renamed by a caller. Its assay provenance still needs
    source review.
    """
    if isinstance(value, Mapping):
        forbidden = set(value) & FORBIDDEN_INPUT_FIELDS
        if forbidden:
            raise ContractError(
                f"Referee field in predictor input at {location}: {sorted(forbidden)}"
            )
        for key, item in value.items():
            guard_predictor_fields(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            guard_predictor_fields(item, f"{location}[{index}]")


def _validator(name: str) -> Draft202012Validator:
    source = files("solubility_comparisons").joinpath("schemas", f"{name}.json")
    return Draft202012Validator(json.loads(source.read_text(encoding="utf-8")))


def _validate(row: Mapping[str, Any], name: str) -> None:
    errors = sorted(
        _validator(name).iter_errors(row), key=lambda error: str(error.path)
    )
    if errors:
        first = errors[0]
        path = ".".join(map(str, first.path)) or "$"
        raise ContractError(f"{name} schema at {path}: {first.message}")


def validate_task(row: Mapping[str, Any]) -> TaskRecord:
    guard_predictor_fields(row)
    _validate(row, "task")
    if row["stage"] == "EXTERNAL_HOLDOUT":
        history = row.get("exposure_history", {})
        if row["exposure_status"] != "external_holdout" or any(
            history.get(key, False)
            for key in ("previously_exposed", "used_to_develop_metadata_rule")
        ):
            raise ContractError(
                "An exposed/developed task cannot be labelled external holdout"
            )
    return cast(TaskRecord, dict(row))


def validate_judge(row: Mapping[str, Any]) -> dict[str, Any]:
    _validate(row, "judge")
    if row["answer_status"] == "unresolved" and row.get("accepted_answers") is not None:
        raise ContractError("An unresolved reference must not provide accepted answers")
    return dict(row)


def unique_index(
    rows: list[Mapping[str, Any]], key: str = "task_id"
) -> dict[str, Mapping[str, Any]]:
    index = {str(row[key]): row for row in rows}
    if len(index) != len(rows):
        raise ContractError(f"Duplicate {key} values")
    return index
