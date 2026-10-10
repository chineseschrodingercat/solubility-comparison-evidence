"""Fixed source fits against unit-compatible, source-native solvent panels."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .source_support import PAIR_COLUMNS, SourceFit


def compare_reference_panels(
    cells: pd.DataFrame, fits: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Do not average ambiguous state/repeat cells or extrapolate input fits.

    A caller must first audit full-graph identity, mol/L pure-solvent conversion,
    source reuse and tier-pair eligibility. DOI absence alone is not laboratory
    independence. This function evaluates reported point comparisons only.
    """
    usable = fits[
        fits.model.isin(["apelblat", "vanthoff"]) & fits.R2.ge(0.8) & fits.RMSE.le(0.3)
    ]
    source_groups = {
        key: [SourceFit.from_record(row) for row in group.itertuples()]
        for key, group in usable.groupby(PAIR_COLUMNS)
    }
    eligible = []
    for record in cells[
        cells.status.eq("UNIQUE_REFERENCE_CELL") & cells.reference_LogC.notna()
    ].to_dict("records"):
        key = (record["Solute_Canon"], record["Solvent_Canon"])
        supported = [
            fit for fit in source_groups.get(key, []) if fit.supports(record["T_K"])
        ]
        if not supported:
            continue
        record = dict(record)
        record["P_LogC"] = float(
            np.mean([fit.evaluate(record["T_K"]) for fit in supported])
        )
        record["old_group_ids"] = "|".join(str(fit.group_id) for fit in supported)
        eligible.append(record)
    panels, comparisons = [], []
    if not eligible:
        return pd.DataFrame(), pd.DataFrame()
    for identifier, ((source, solute, temperature), group) in enumerate(
        pd.DataFrame(eligible).groupby(["doi", "strict_solute", "T_K"]), 1
    ):
        if group.strict_solvent.nunique() < 2:
            continue
        group = group.sort_values("strict_solvent")
        target_max, prediction_max = group.reference_LogC.max(), group.P_LogC.max()
        reference_top = group[
            group.reference_LogC.eq(target_max)
        ].strict_solvent.tolist()
        prediction_top = group[group.P_LogC.eq(prediction_max)].strict_solvent.tolist()
        panels.append(
            dict(
                panel_id=identifier,
                doi=source,
                strict_solute=solute,
                T_K=temperature,
                n_solvents=len(group),
                reference_top="|".join(reference_top),
                P_top="|".join(prediction_top),
                P_top_matches_reference=bool(set(reference_top) & set(prediction_top)),
                all_native_released_T=bool(group.released_native_tier_T.all()),
                physical_status="METADATA_ONLY_TERMINAL_STATE_UNKNOWN",
            )
        )
        for a, b in combinations(group.itertuples(), 2):
            at_a = {
                fit.group_id: fit
                for fit in source_groups[(a.Solute_Canon, a.Solvent_Canon)]
                if fit.supports(temperature)
            }
            at_b = {
                fit.group_id: fit
                for fit in source_groups[(b.Solute_Canon, b.Solvent_Canon)]
                if fit.supports(temperature)
            }
            common = sorted(set(at_a) & set(at_b))
            contrast = (
                float(
                    np.mean(
                        [
                            at_a[key].evaluate(temperature)
                            - at_b[key].evaluate(temperature)
                            for key in common
                        ]
                    )
                )
                if common
                else np.nan
            )
            comparisons.append(
                dict(
                    panel_id=identifier,
                    doi=source,
                    strict_solute=solute,
                    T_K=temperature,
                    solvent_A=a.strict_solvent,
                    solvent_B=b.strict_solvent,
                    solvent_name_A=a.solvent_name,
                    solvent_name_B=b.solvent_name,
                    reference_delta=a.reference_LogC - b.reference_LogC,
                    P_delta=a.P_LogC - b.P_LogC,
                    C_delta=contrast,
                    n_common_groups=len(common),
                    common_old_gids="|".join(map(str, common)),
                    A_old_gids=a.old_group_ids,
                    B_old_gids=b.old_group_ids,
                    A_raw_ids=a.raw_ids,
                    B_raw_ids=b.raw_ids,
                    both_native_released_T=a.released_native_tier_T
                    and b.released_native_tier_T,
                    physical_status="METADATA_ONLY_TERMINAL_STATE_UNKNOWN",
                )
            )
    return pd.DataFrame(panels), pd.DataFrame(comparisons)
