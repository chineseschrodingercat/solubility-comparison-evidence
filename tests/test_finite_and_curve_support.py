import pytest

from solubility_comparisons.answer_sets import Interpretation, finite_answer_set
from solubility_comparisons.contracts import ContractError
from solubility_comparisons.curves import NativeCurve
from solubility_comparisons.quantities import (
    documented_mass_basis_alternatives,
    mole_fraction_to_pure_solvent_molar,
)


def test_incomplete_enumeration_keeps_union_but_withholds_invariance():
    result = finite_answer_set(
        [Interpretation("available", {"A": 1.0, "B": 2.0})], ["A", "B"], complete=False
    )
    assert result["supported_numerical_answer_set"] == ["A"]
    assert result["invariant_numeric_answer"] is None
    assert result["strict_partial_order_edges"] == []


def test_partial_order_can_survive_ambiguous_minimum():
    states = [
        Interpretation("first", {"A": 1.0, "B": 2.0, "C": 4.0}),
        Interpretation("second", {"A": 2.0, "B": 1.0, "C": 4.0}),
    ]
    result = finite_answer_set(states, ["A", "B", "C"])
    assert result["supported_numerical_answer_set"] == ["A", "B"]
    assert result["strict_partial_order_edges"] == [["A", "<", "C"], ["B", "<", "C"]]


def test_centered_log_contrast_uses_absolute_gap():
    result = finite_answer_set(
        [Interpretation("one", {"A": -0.2, "B": 0.0})],
        ["A", "B"],
        direction="higher",
        gap_scale="absolute_log_contrast",
    )
    assert result["invariant_numeric_answer"] == "B"
    assert result["minimum_representation_gap"] == 0.2


def test_bool_is_not_a_concentration():
    with pytest.raises(ContractError):
        finite_answer_set([Interpretation("one", {"A": True, "B": 2.0})], ["A", "B"])


@pytest.mark.parametrize(
    "kwargs", [dict(direction="sideways"), dict(gap_scale="unknown")]
)
def test_invalid_finite_set_modes_do_not_fall_through(kwargs):
    with pytest.raises(ContractError):
        finite_answer_set(
            [Interpretation("one", {"A": 1.0, "B": 2.0})], ["A", "B"], **kwargs
        )


def test_mass_basis_alternatives_preserve_order_not_absolute_value():
    choices = documented_mass_basis_alternatives({"A": 100.0, "B": 200.0})
    assert all(values["A"] < values["B"] for values in choices.values())
    assert (
        choices["solute_per_solvent_mass"]["A"]
        != choices["solution_mass_fraction_to_solvent_ratio"]["A"]
    )


def test_molar_conversion_uses_x_over_one_minus_x():
    actual = mole_fraction_to_pure_solvent_molar(0.2, 800.0, 80.0)
    assert actual == pytest.approx(2.5)
    assert actual != 0.2 * 800.0 / 80.0


def test_single_native_point_cannot_be_extrapolated():
    curve = NativeCurve.from_samples([300.0], [2.0])
    assert curve.evaluate(300.0) == pytest.approx(2.0)
    with pytest.raises(ContractError, match="no extrapolation"):
        curve.evaluate(301.0)


def test_duplicate_temperatures_require_explicit_record_rule():
    with pytest.raises(ContractError, match="Duplicate"):
        NativeCurve.from_samples([300.0, 300.0], [2.0, 3.0])


def test_common_monotone_basis_transform_need_not_commute_with_interpolation():
    q = NativeCurve.from_samples([300.0, 310.0], [0.1, 0.5]).evaluate(
        305.0, "linear_piecewise"
    )
    transformed = NativeCurve.from_samples(
        [300.0, 310.0], [0.1 / 0.9, 0.5 / 0.5]
    ).evaluate(305.0, "linear_piecewise")
    assert q / (1 - q) != pytest.approx(transformed)
