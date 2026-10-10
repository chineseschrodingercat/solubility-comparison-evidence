"""Measured process targets remain distinct from concentration or stable phase."""

from __future__ import annotations

from statistics import NormalDist
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ContractError


def marginal_wilson_interval(
    successes: int, trials: int, level: float = 0.95
) -> tuple[float, float]:
    """Source-style binomial marginal; not independent-study uncertainty.

    Shared stock, bath and reused-vial dependence is not estimated by this
    calculation and must be recorded separately by the caller.
    """
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, Integral)
        or not isinstance(trials, Integral)
    ):
        raise ContractError("Wilson counts must be integers, not fractions or booleans")
    if not 0 < level < 1 or trials <= 0 or not 0 <= successes <= trials:
        raise ContractError("Invalid Wilson count or level")
    z = NormalDist().inv_cdf((1 + level) / 2)
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = (
        z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    )
    return float(center - radius), float(center + radius)


def mixture_assignment_sensitivity(counts: pd.DataFrame) -> pd.DataFrame:
    """Retain pure and mixed counts under all three declared conventions."""
    rows = []
    for row in counts.to_dict("records"):
        pure_i, pure_ii, mixed = row["pure_I"], row["pure_II"], row["mixed"]
        total = pure_i + pure_ii + mixed
        for rule, numerator, denominator in (
            ("mixed_to_II", pure_ii + mixed, total),
            ("mixed_to_I", pure_ii, total),
            ("mixed_excluded", pure_ii, pure_i + pure_ii),
        ):
            rows.append(
                dict(
                    series_id=row["series_id"],
                    rule=rule,
                    pure_I=pure_i,
                    pure_II=pure_ii,
                    mixed=mixed,
                    n_classifiable=denominator,
                    form_II_fraction=numerator / denominator if denominator else np.nan,
                )
            )
    return pd.DataFrame(rows)


def compare_native_objectives(
    capacities: pd.DataFrame, conditions: pd.DataFrame
) -> pd.DataFrame:
    """Compare all native shared drive levels without selecting favorable ones.

    Requires normalized columns solvent,T_K,capacity and, for process records,
    drive,target_phase_count,n_trials,target_conditional_median_s. The driver
    coordinate may itself use a concentration/activity approximation. Phase
    counts and conditional visual times remain separately measured endpoints.
    """
    rows = []
    for temperature, capacity in capacities.groupby("T_K"):
        solvents = sorted(capacity.solvent.unique())
        if len(solvents) < 2 or capacity.solvent.duplicated().any():
            raise ContractError(
                "Need one specified-phase capacity per candidate solvent"
            )
        process = conditions[
            conditions.T_K.eq(temperature) & conditions.solvent.isin(solvents)
        ]
        shared = set.intersection(
            *(set(process[process.solvent.eq(solvent)].drive) for solvent in solvents)
        )
        chosen = set(capacity[capacity.capacity.eq(capacity.capacity.max())].solvent)
        for drive in sorted(shared):
            panel = process[process.drive.eq(drive)]
            if panel.solvent.duplicated().any() or set(panel.solvent) != set(solvents):
                raise ContractError("Ambiguous native process panel")
            fractions = panel.target_phase_count / panel.n_trials
            highest_fraction = set(panel.loc[fractions.eq(fractions.max()), "solvent"])
            fastest = set(
                panel[
                    panel.target_conditional_median_s.eq(
                        panel.target_conditional_median_s.min()
                    )
                ].solvent
            )
            rows.append(
                dict(
                    T_K=temperature,
                    drive=drive,
                    capacity_choice="|".join(sorted(chosen)),
                    fastest_conditional_choice="|".join(sorted(fastest)),
                    highest_phase_fraction_choice="|".join(sorted(highest_fraction)),
                    capacity_agrees_faster_conditional=bool(chosen & fastest),
                    capacity_agrees_phase_fraction=bool(chosen & highest_fraction),
                    role="native point comparison; conditional endpoint and dependency limits retained",
                )
            )
    return pd.DataFrame(rows)
