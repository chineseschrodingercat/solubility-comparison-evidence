"""Dissolution-category policies and distinct crystal-extractability outcomes."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ContractError

PILLONG_DISSOLUTION = {"RS": 4, "KS": 3, "TS": 2, "PS": 1}
PILLONG_EXTRACTABLE = {"XX": 1, "YX": 1, "CT": 0, "DR": 0, "FI": 0, "AM": 0}


def pillong_cells(wide: pd.DataFrame) -> pd.DataFrame:
    """Preserve invalid/missing slots instead of counting them as failures."""
    if wide.PID.isna().any() or wide.PID.duplicated().any():
        raise ContractError("Screen compound IDs must be present and unique")
    solvents = [column[2:] for column in wide if column.startswith("S-")]
    if set(solvents) != {column[2:] for column in wide if column.startswith("C-")}:
        raise ContractError("Dissolution and outcome candidate sets differ")
    rows = []
    for record in wide.to_dict("records"):
        for solvent in solvents:
            dissolution, outcome = record["S-" + solvent], record["C-" + solvent]
            rows.append(
                dict(
                    PID=record["PID"],
                    SMILES=record["SMILES"],
                    Cluster=record.get("Cluster"),
                    Purity=record.get("Purity"),
                    Solvent=solvent,
                    S_code=dissolution,
                    C_code=outcome,
                    valid=dissolution in PILLONG_DISSOLUTION
                    and outcome in PILLONG_EXTRACTABLE,
                    S_ordinal=PILLONG_DISSOLUTION.get(dissolution),
                    crystal_success=PILLONG_EXTRACTABLE.get(outcome),
                )
            )
    return pd.DataFrame(rows)


def screen_policy_comparison(
    cells: pd.DataFrame, *, complete_only: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected = cells[cells.valid].copy()
    if complete_only:
        selected = selected[selected.S_code.isin(["RS", "KS", "TS"])]
    policies, discordant = [], []
    for identifier, group in selected.groupby("PID"):
        top = group[group.S_ordinal.eq(group.S_ordinal.max())]
        lower = group[group.S_ordinal.lt(group.S_ordinal.max())]
        policies.append(
            dict(
                PID=identifier,
                SMILES=group.SMILES.iloc[0],
                Purity=group.Purity.iloc[0],
                Cluster=group.Cluster.iloc[0],
                n_observed=len(group),
                n_S_levels=group.S_ordinal.nunique(),
                top_S=top.S_code.iloc[0],
                n_top=len(top),
                n_low=len(lower),
                top_uniform_success=top.crystal_success.mean(),
                all_uniform_success=group.crystal_success.mean(),
                low_uniform_success=(
                    lower.crystal_success.mean() if len(lower) else np.nan
                ),
                top_minus_all=top.crystal_success.mean() - group.crystal_success.mean(),
                top_minus_low=(
                    top.crystal_success.mean() - lower.crystal_success.mean()
                    if len(lower)
                    else np.nan
                ),
                any_observed_crystal=bool(group.crystal_success.max() > 0),
                top_all_fail_low_has_success=bool(
                    len(lower)
                    and top.crystal_success.sum() == 0
                    and lower.crystal_success.sum() > 0
                ),
                top_all_success_low_all_fail=bool(
                    len(lower)
                    and top.crystal_success.min() == 1
                    and lower.crystal_success.max() == 0
                ),
            )
        )
        for grade, same_grade in group.groupby("S_code"):
            for a, b in combinations(
                same_grade.sort_values("Solvent").to_dict("records"), 2
            ):
                if a["crystal_success"] != b["crystal_success"]:
                    discordant.append(
                        dict(
                            PID=identifier,
                            SMILES=group.SMILES.iloc[0],
                            S_code=grade,
                            solvent_A=a["Solvent"],
                            C_A=a["C_code"],
                            solvent_B=b["Solvent"],
                            C_B=b["C_code"],
                            top_S_class=grade == top.S_code.iloc[0],
                        )
                    )
    policy_frame, pair_frame = pd.DataFrame(policies), pd.DataFrame(discordant)
    summary = dict(
        role="Previously exposed descriptive screen policy; no phase, yield or causal benefit",
        complete_only=complete_only,
        observed_compounds=len(policy_frame),
        observed_cells=len(selected),
        top_uniform_success=(
            float(policy_frame.top_uniform_success.mean())
            if len(policy_frame)
            else None
        ),
        all_uniform_success=(
            float(policy_frame.all_uniform_success.mean())
            if len(policy_frame)
            else None
        ),
        no_independent_cycle_or_source_inflation=True,
    )
    return policy_frame, pair_frame, summary
