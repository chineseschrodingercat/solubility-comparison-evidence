"""Temperature conditioning of released ordinal questions, not model retraining."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import ContractError


def audit_released_temperatures(
    raw: pd.DataFrame, tasks: Mapping[int, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = {
        (str(source), str(solute), str(solvent)): frame.sort_values("Temperature_K")
        for (source, solute, solvent), frame in raw.groupby(
            ["Source", "SMILES_Solute", "Solvent"], sort=False
        )
    }
    components, results = [], []
    for kind in (1, 2):
        for record in tasks[kind].to_dict("records"):
            options = (
                [
                    (record["Solvent_A"], record["LogS_A"]),
                    (record["Solvent_B"], record["LogS_B"]),
                ]
                if kind == 1
                else list(
                    zip(
                        record["All_solvents"].split(";"),
                        map(float, record["All_LogS"].split(";")),
                    )
                )
            )
            local = []
            target = record["Temperature_K"]
            for index, (solvent, value) in enumerate(options):
                group = groups.get((record["DOI"], record["SMILES"], solvent))
                matching = (
                    group[np.isclose(group["LogS(mol/L)"], value, atol=1e-10, rtol=0)]
                    if group is not None
                    else pd.DataFrame()
                )
                native_t = (
                    sorted(matching.Temperature_K.unique()) if len(matching) else []
                )
                local.append(
                    dict(
                        task=kind,
                        id=record["id"],
                        DOI=record["DOI"],
                        SMILES=record["SMILES"],
                        option_index=index,
                        solvent=solvent,
                        released_LogS=value,
                        reported_panel_T_K=target,
                        native_T_unique_count=len(native_t),
                        native_T_values=";".join(map(str, native_t)),
                        native_T_K=native_t[0] if len(native_t) == 1 else np.nan,
                        exact_question_T_among_matches=bool(
                            any(abs(t - target) < 1e-8 for t in native_t)
                        ),
                    )
                )
            components.extend(local)
            component = pd.DataFrame(local)
            values, reasons, exact = [], [], []
            for option in local:
                group = groups.get((record["DOI"], record["SMILES"], option["solvent"]))
                if group is None:
                    values.append(np.nan)
                    reasons.append("missing_source")
                    exact.append(False)
                    continue
                unique = group.groupby("Temperature_K")["LogS(mol/L)"].agg(
                    ["min", "max", "mean"]
                )
                if (unique["max"] - unique["min"] > 1e-9).any():
                    values.append(np.nan)
                    reasons.append("duplicate_T_conflict")
                    exact.append(False)
                    continue
                x, y = unique.index.to_numpy(float), unique["mean"].to_numpy(float)
                found = np.flatnonzero(np.isclose(x, target, atol=1e-8, rtol=0))
                if len(found):
                    values.append(float(y[found[0]]))
                    reasons.append("native")
                    exact.append(True)
                elif len(x) >= 2 and x[0] < target < x[-1]:
                    values.append(float(np.interp(target, x, y)))
                    reasons.append("interpolated")
                    exact.append(False)
                else:
                    values.append(np.nan)
                    reasons.append("out_of_domain")
                    exact.append(False)
            available = bool(np.isfinite(values).all())
            all_unique = bool(component.native_T_unique_count.eq(1).all())
            native = component.native_T_K.values
            original_winner = int(np.argmax(component.released_LogS.values))
            new_winner = int(np.argmax(values)) if available else None
            margin = (
                float(sorted(values)[-1] - sorted(values)[-2]) if available else None
            )
            results.append(
                dict(
                    task=kind,
                    id=record["id"],
                    DOI=record["DOI"],
                    SMILES=record["SMILES"],
                    reported_panel_T_K=target,
                    all_options_unique_native_T=all_unique,
                    heterogeneous_native_T=bool(all_unique and np.ptp(native) > 1e-8),
                    native_T_span_K=float(np.ptp(native)) if all_unique else None,
                    all_values_measured_at_reported_panel_T=bool(
                        component.exact_question_T_among_matches.all()
                    ),
                    target_all_source_curves_available=available,
                    target_all_native=bool(available and all(exact)),
                    original_winner_index=original_winner,
                    target_winner_index=new_winner,
                    target_answer_changed=bool(
                        available and new_winner != original_winner
                    ),
                    target_margin_log10molL=margin,
                    original_threshold_retained=bool(
                        available and margin >= (0.5 if kind == 1 else 0.1)
                    ),
                    option_solvents=";".join(component.solvent),
                    target_LogS_values=";".join(map(str, values)),
                    adjudication_status=";".join(reasons),
                )
            )
    return pd.DataFrame(components), pd.DataFrame(results)
