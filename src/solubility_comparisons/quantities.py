"""Named concentration conversions; solution and solvent volumes are distinct."""

from __future__ import annotations

import math
from typing import Mapping

from .contracts import ContractError


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ContractError(f"{name} must be finite and positive")


def mole_fraction_to_pure_solvent_molar(
    x: float, density_g_per_l: float, solvent_molar_mass: float
) -> float:
    """Return mol/L pure solvent, not mol/L saturated solution."""
    _positive(density_g_per_l, "Pure-solvent density")
    _positive(solvent_molar_mass, "Solvent molar mass")
    if not math.isfinite(x) or not 0 <= x < 1:
        raise ContractError("Solute mole fraction must lie in [0,1)")
    return x / (1 - x) * density_g_per_l / solvent_molar_mass


def g100g_to_pure_solvent_molar(
    q: float, density_g_per_l: float, solute_molar_mass: float
) -> float:
    _positive(density_g_per_l, "Pure-solvent density")
    _positive(solute_molar_mass, "Solute molar mass")
    if not math.isfinite(q) or q < 0:
        raise ContractError("Reported solute mass must be nonnegative")
    return q * density_g_per_l / (100 * solute_molar_mass)


def documented_mass_basis_alternatives(
    values_g_per_kg: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    """Two specified denominator interpretations; not all possible unit errors.

    Apply to a reported query value. Nonlinear transformation need not commute
    with interpolation, so these functions do not choose an interpolation rule.
    """
    ratio = {key: value / 1000 for key, value in values_g_per_kg.items()}
    if not ratio or any(
        not math.isfinite(value) or not 0 <= value < 1 for value in ratio.values()
    ):
        raise ContractError(
            "The solution-mass alternative requires fractions below one"
        )
    return {
        "solute_per_solvent_mass": ratio,
        "solution_mass_fraction_to_solvent_ratio": {
            key: value / (1 - value) for key, value in ratio.items()
        },
    }
