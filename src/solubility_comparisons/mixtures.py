"""Explicitly requested overlap windows and a named pooling counterfactual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ContractError
from .curves import NativeCurve, common_native_union

SYSTEM_COLUMNS = [
    "SMILES_Solute",
    "solvent_lo",
    "solvent_hi",
    "Fraction_Type",
    "fraction_lo_key",
]


@dataclass(frozen=True)
class MixtureComparison:
    native_rows: pd.DataFrame
    point_comparisons: pd.DataFrame
    response_segments: pd.DataFrame
    windows: pd.DataFrame


def compare_native_windows(
    raw: pd.DataFrame, windows: pd.DataFrame
) -> MixtureComparison:
    """Window selection and chemical qualification come from caller evidence.

    Nothing in a numerical offset assigns an identity-error or phase label.
    Duplicate native temperatures are rejected rather than silently averaged.
    """
    data = raw.copy()
    data["raw_row_index"] = np.arange(len(data))
    data["doi_key"] = data.Source.str.strip().str.lower()
    swap = data.SMILES_Solvent1 > data.SMILES_Solvent2
    data["solvent_lo"] = np.where(swap, data.SMILES_Solvent2, data.SMILES_Solvent1)
    data["solvent_hi"] = np.where(swap, data.SMILES_Solvent1, data.SMILES_Solvent2)
    data["fraction_lo_key"] = np.where(
        swap, data.Fraction_Solvent2, data.Fraction_Solvent1
    ).round(8)
    native, points, segments, summaries = [], [], [], []
    for index, row in windows.iterrows():
        identifier = row.get("window_id", index)
        metadata = {key: row[key] for key in SYSTEM_COLUMNS}
        metadata.update(
            doi_A=row.doi_A,
            doi_B=row.doi_B,
            window_id=identifier,
            qualification=row.get("qualification", "UNREVIEWED_CHEMICAL_IDENTITY"),
        )
        mask = pd.Series(True, index=data.index)
        for column in SYSTEM_COLUMNS:
            mask &= data[column].eq(row[column])
        panel = data[mask & data.doi_key.isin([row.doi_A, row.doi_B])]
        source_frames, curves = {}, {}
        for source, frame in panel.groupby("doi_key"):
            frame = frame.sort_values("Temperature_K")
            values = frame["Solubility(mole_fraction)"]
            if not values.between(0, 1, inclusive="neither").all():
                raise ContractError(
                    "This window adapter requires positive solute mole fractions below one"
                )
            curves[source] = NativeCurve.from_samples(frame.Temperature_K, values)
            source_frames[source] = frame
            native.extend(
                frame.assign(
                    window_id=identifier, qualification=metadata["qualification"]
                ).to_dict("records")
            )
        if set(curves) != {row.doi_A, row.doi_B}:
            raise ContractError("Window lacks one of its declared sources")
        a, b = curves[row.doi_A], curves[row.doi_B]
        grid = common_native_union(a, b)
        if not len(grid):
            raise ContractError("Window has no common native domain")
        xa = np.array([a.evaluate(t, "linear_piecewise") for t in grid])
        xb = np.array([b.evaluate(t, "linear_piecewise") for t in grid])
        la = np.array([a.evaluate(t, "log10_piecewise") for t in grid])
        lb = np.array([b.evaluate(t, "log10_piecewise") for t in grid])
        for i, t in enumerate(grid):
            points.append(
                metadata
                | dict(
                    Temperature_K=t,
                    x_A_linear=xa[i],
                    x_B_linear=xb[i],
                    log10x_A_linear=la[i],
                    log10x_B_linear=lb[i],
                    delta_log10x_A_minus_B=la[i] - lb[i],
                    delta_x_A_minus_B=xa[i] - xb[i],
                    A_native=t in set(a.temperatures),
                    B_native=t in set(b.temperatures),
                )
            )
        for i in range(len(grid) - 1):
            segments.append(
                metadata
                | dict(
                    T0=grid[i],
                    T1=grid[i + 1],
                    delta_log10x_A=la[i + 1] - la[i],
                    delta_log10x_B=lb[i + 1] - lb[i],
                    response_difference=(la[i + 1] - la[i]) - (lb[i + 1] - lb[i]),
                )
            )
        summaries.append(
            metadata
            | dict(
                Tlo=grid.min(),
                Thi=grid.max(),
                n_native_A_full=len(a.temperatures),
                n_native_B_full=len(b.temperatures),
                n_union=len(grid),
                n_exact_T=len(set(a.temperatures) & set(b.temperatures)),
                full_domain_response_A=la[-1] - la[0],
                full_domain_response_B=lb[-1] - lb[0],
                full_domain_response_difference=(la[-1] - la[0]) - (lb[-1] - lb[0]),
            )
        )
    return MixtureComparison(
        pd.DataFrame(native),
        pd.DataFrame(points),
        pd.DataFrame(segments),
        pd.DataFrame(summaries),
    )


def arithmetic_pooling_counterfactual(points: pd.DataFrame) -> pd.DataFrame:
    """Equal-source mean of x with source B as a declared target, not official labels."""
    result = points.copy()
    result["pooled_x"] = (result.x_A_linear + result.x_B_linear) / 2
    result["target_x"] = result.x_B_linear
    result["absolute_shift"] = result.pooled_x - result.target_x
    result["relative_shift"] = result.pooled_x / result.target_x - 1
    result["ln_pooled_x"] = np.log(result.pooled_x)
    result["ln_target_x"] = np.log(result.target_x)
    result["ln_shift"] = result.ln_pooled_x - result.ln_target_x
    return result
