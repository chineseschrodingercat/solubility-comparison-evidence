"""Released-fit membership decomposition, not a physical temperature referee."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ContractError

PAIR_COLUMNS = ["Solute_Canon", "Solvent_Canon"]


@dataclass(frozen=True)
class SourceFit:
    group_id: int
    model: str
    a: float
    b: float
    c: float
    lower: float
    upper: float

    @classmethod
    def from_record(cls, row: Any) -> SourceFit:
        fit = cls(
            int(row.gid),
            str(row.model),
            float(row.A),
            float(row.B),
            float(row.C) if row.model == "apelblat" else 0.0,
            float(row.T_min),
            float(row.T_max),
        )
        if fit.model not in {"apelblat", "vanthoff"} or not all(
            math.isfinite(value)
            for value in (fit.a, fit.b, fit.c, fit.lower, fit.upper)
        ):
            raise ContractError("Invalid released source fit")
        if not 0 < fit.lower <= fit.upper:
            raise ContractError("Invalid source-fit temperature range")
        return fit

    def supports(self, temperature: float) -> bool:
        return self.lower <= temperature <= self.upper

    def evaluate(self, temperature: float) -> float:
        if not self.supports(temperature):
            raise ContractError("Cannot extrapolate a released source fit")
        return float(
            self.a
            + self.b / temperature
            + (self.c * np.log(temperature) if self.model == "apelblat" else 0)
        )


def select_contributing_fits(fits: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Caller supplies released/audited group eligibility; no new curation rule."""
    keys = keys.rename(columns={"group_id": "gid"})[
        PAIR_COLUMNS + ["gid"]
    ].drop_duplicates()
    return fits.merge(keys, on=PAIR_COLUMNS + ["gid"], validate="one_to_one")


def membership_audit(
    fits: pd.DataFrame,
    bronze: pd.DataFrame,
    gold: pd.DataFrame,
    silver: pd.DataFrame,
    *,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    gold_keys = set(gold[PAIR_COLUMNS].itertuples(index=False, name=None))
    silver_keys = set(silver[PAIR_COLUMNS].itertuples(index=False, name=None))
    fit_pairs = {
        key: [SourceFit.from_record(row) for row in group.itertuples()]
        for key, group in fits.groupby(PAIR_COLUMNS)
    }
    rows, intervals = [], []
    for key, group in bronze.groupby(PAIR_COLUMNS):
        evaluated = []
        if group.Temperature_K.duplicated().any():
            raise ContractError("Released pair has duplicate label temperatures")
        for row in group.sort_values("Temperature_K").itertuples():
            contributions = {
                fit.group_id: fit.evaluate(row.Temperature_K)
                for fit in fit_pairs[key]
                if fit.supports(row.Temperature_K)
            }
            if not contributions:
                raise ContractError("Released label has no supported contributing fit")
            mean = float(np.mean(list(contributions.values())))
            rows.append(
                dict(zip(PAIR_COLUMNS, key))
                | dict(
                    T=float(row.Temperature_K),
                    n=len(contributions),
                    n_reported=int(row.n_contributing_groups),
                    abs_diff=abs(mean - row.LogS_consensus),
                )
            )
            evaluated.append((row, contributions))
        for (left, at_left), (right, at_right) in zip(evaluated, evaluated[1:]):
            common = set(at_left) & set(at_right)
            observed = float(right.LogS_consensus - left.LogS_consensus)
            continuing = (
                float(np.mean([at_right[group] - at_left[group] for group in common]))
                if common
                else np.nan
            )
            composition = observed - continuing
            inverse_gap = 1 / right.Temperature_K - 1 / left.Temperature_K
            # Concentration-label slope scale only. Activity and fixed solid state are not established.
            scale = -8.314462618 * np.log(10) / inverse_gap / 1000
            intervals.append(
                dict(zip(PAIR_COLUMNS, key))
                | dict(
                    T0=left.Temperature_K,
                    T1=right.Temperature_K,
                    dT=right.Temperature_K - left.Temperature_K,
                    gids0="|".join(map(str, sorted(at_left))),
                    gids1="|".join(map(str, sorted(at_right))),
                    n0=len(at_left),
                    n1=len(at_right),
                    n_common=len(common),
                    membership_switch=set(at_left) != set(at_right),
                    Gold=key in gold_keys,
                    Silver=key in silver_keys,
                    consensus_delta_logS=observed,
                    continuing_groups_delta_logS=continuing,
                    membership_component_logS=composition,
                    consensus_slope_scale_kJ=scale * observed,
                    continuing_slope_scale_kJ=scale * continuing,
                    membership_slope_scale_kJ=scale * composition,
                    sigma0=left.sigma,
                    sigma1=right.sigma,
                    sigma_defined_both=bool(
                        np.isfinite(left.sigma) and np.isfinite(right.sigma)
                    ),
                    numeric_increase=bool(observed > 0),
                    continuing_increase=bool(continuing > 0),
                )
            )
    row_frame, interval_frame = pd.DataFrame(rows), pd.DataFrame(intervals)
    if (
        not row_frame.n.eq(row_frame.n_reported).all()
        or row_frame.abs_diff.max() >= tolerance
    ):
        raise ContractError(
            "The supplied fits/keys do not reproduce the released labels"
        )
    stable = interval_frame.loc[
        ~interval_frame.membership_switch, "membership_component_logS"
    ]
    if len(stable) and stable.abs().max() >= tolerance:
        raise ContractError("Fixed-membership decomposition failed")
    summary = dict(
        role="Processing decomposition in actual SC3 concentration labels; not phase or thermodynamic error",
        pairs=len(fit_pairs),
        rows=len(row_frame),
        max_abs_consensus_diff=float(row_frame.abs_diff.max()),
        intervals=len(interval_frame),
        tiers={},
    )
    for name, mask in (
        ("Gold", interval_frame.Gold),
        ("Silver", interval_frame.Silver),
        ("Bronze", pd.Series(True, index=interval_frame.index)),
    ):
        selected = interval_frame[mask]
        switched = selected[selected.membership_switch]
        opposite = lambda frame: (
            frame.consensus_delta_logS * frame.continuing_groups_delta_logS
        ).lt(0)
        summary["tiers"][name] = dict(
            pairs=len(selected[PAIR_COLUMNS].drop_duplicates()),
            intervals=len(selected),
            switch_intervals=len(switched),
            pairs_with_switch=len(switched[PAIR_COLUMNS].drop_duplicates()),
            switch_intervals_lt1K=int(switched.dT.lt(1).sum()),
            switch_intervals_no_common_group=int(switched.n_common.eq(0).sum()),
            switch_sigma_defined_both=int(switched.sigma_defined_both.sum()),
            switch_sign_disagreements=int(opposite(switched).sum()),
            switch_sign_disagreement_by_gap={
                label: int(opposite(frame).sum())
                for label, frame in (
                    ("below_1K", switched[switched.dT.lt(1)]),
                    ("1_to_below_5K", switched[switched.dT.ge(1) & switched.dT.lt(5)]),
                    ("at_least_5K", switched[switched.dT.ge(5)]),
                )
            },
            sign_disagreements_with_sigma_defined_both=int(
                (switched.sigma_defined_both & opposite(switched)).sum()
            ),
            max_abs_membership_component_logS=float(
                switched.membership_component_logS.abs().max()
            ),
            median_abs_membership_component_logS=float(
                switched.membership_component_logS.abs().median()
            ),
        )
    return row_frame, interval_frame, summary
