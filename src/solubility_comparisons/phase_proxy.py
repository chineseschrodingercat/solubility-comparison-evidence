"""Concentration-curve variants without reading a terminal-phase referee."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from pathlib import Path
import csv

import numpy as np
import pandas as pd

from .contracts import ContractError, guard_predictor_fields
from .curves import NativeCurve

DEFAULT_BASES = ("reported_C_mg_ml", "mass_volume_C_mg_ml")
DEFAULT_REPRESENTATIONS = (
    "log_piecewise",
    "linear_piecewise",
    "log_global_fit",
    "log_piecewise_first_cycle",
)


def read_gravimetric_curves(paths: Mapping[str, Path]) -> pd.DataFrame:
    """Normalize source CSVs using their printed decimal values.

    This adapter follows the archived clear-point header contract. Native
    reported C and amount/volume-derived C remain separate, with no rounding
    through an intermediate derived CSV before curve fitting.
    """
    rows = []
    for form, path in paths.items():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for record in csv.DictReader(stream, skipinitialspace=True):
                record = {key.strip(): value.strip() for key, value in record.items()}
                mass, volume = float(record["Amount/mg"]), float(record["Volume/mL"])
                rows.append(
                    dict(
                        form=form,
                        Tclear_C=float(record["Tclear1/2"]),
                        reported_C_mg_ml=float(record["Conc/mgmL-1"]),
                        mass_volume_C_mg_ml=mass / volume,
                        notes=record.get("notes", ""),
                    )
                )
    return pd.DataFrame(rows)


def phase_proxy_variants(
    observations: pd.DataFrame,
    tasks: Sequence[Mapping[str, Any]],
    *,
    bases: Sequence[str] = DEFAULT_BASES,
    representations: Sequence[str] = DEFAULT_REPRESENTATIONS,
    first_cycle_exclusion: str = "recrystallised",
) -> pd.DataFrame:
    """Exact-temperature averaging is an explicit archived construction rule.

    Observations must already identify the measured phase. Distinct phase rows
    are never pooled. The first-cycle variant uses the supplied assay note, not
    the terminal experiment's answer.
    """
    guard_predictor_fields({column: None for column in observations.columns})
    for task in tasks:
        guard_predictor_fields(task)
    required = {"form", "Tclear_C", *bases}
    if not required <= set(observations):
        raise ContractError(
            f"Missing curve columns: {sorted(required-set(observations))}"
        )
    rows = []
    for task in tasks:
        context = task["context"]
        candidates = context["candidate_ids"]
        temperature = context.get("temperature_C")
        if temperature is None:
            raise ContractError(
                "Native curve temperature units are Celsius for this adapter"
            )
        for basis in bases:
            for representation in representations:
                values, supported = {}, True
                for phase in candidates:
                    selected = observations[observations.form.eq(phase)]
                    if representation.endswith("first_cycle"):
                        notes = selected.get(
                            "notes", pd.Series("", index=selected.index)
                        ).fillna("")
                        selected = selected[
                            ~notes.astype(str).str.contains(
                                first_cycle_exclusion, regex=False
                            )
                        ]
                    curve = NativeCurve.from_samples(
                        selected.Tclear_C,
                        selected[basis],
                        duplicate="mean_exact_temperature",
                    )
                    if not curve.supports(temperature):
                        supported = False
                        values[phase] = np.nan
                    else:
                        method = (
                            "log_piecewise"
                            if representation.endswith("first_cycle")
                            else representation
                        )
                        values[phase] = curve.evaluate(temperature, method)
                minimum = min(values, key=values.get) if supported else None
                rows.append(
                    dict(
                        task_id=task["task_id"],
                        lab=context.get("lab"),
                        T_C=temperature,
                        replicate=context.get("replicate"),
                        concentration_basis=basis,
                        representation=representation,
                        in_native_shared_range=supported,
                        **{
                            f"C_{phase}_mg_ml": value for phase, value in values.items()
                        },
                        numeric_I_II_candidate=(
                            min(("I", "II"), key=values.get)
                            if supported and {"I", "II"} <= set(values)
                            else None
                        ),
                        numeric_I_II_V_candidate=minimum,
                    )
                )
    return pd.DataFrame(rows).sort_values(
        ["lab", "T_C", "replicate", "concentration_basis", "representation"]
    )
