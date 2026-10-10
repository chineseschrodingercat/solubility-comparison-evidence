"""Full-graph keys and original-table row links; no identity inference from rank."""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Sequence

import pandas as pd

from .contracts import ContractError


@lru_cache(maxsize=16384)
def full_graph_smiles(smiles: str) -> str | None:
    """Preserve salts/fragments, stereo, isotopes and charge; no similarity key."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError(
            "Install the optional chemistry extra for full-graph matching"
        ) from exc
    molecule = (
        Chem.MolFromSmiles(smiles.strip())
        if isinstance(smiles, str) and smiles.strip()
        else None
    )
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        if molecule
        else None
    )


def source_doi(value: str) -> str | None:
    found = (
        re.search(r"10\.\d{4,9}/[^\s\u00a0]+", value.lower())
        if isinstance(value, str)
        else None
    )
    return found.group(0).rstrip(".,") if found else None


def link_original_columns(
    observations: pd.DataFrame,
    primary: pd.DataFrame,
    *,
    keys: Sequence[str],
    concentration: str,
    phase_column: str,
    sd_column: str | None = None,
    tolerance: float = 5e-8,
) -> pd.DataFrame:
    """Numerical matching links rows; phase is read from the original column.

    A nonunique match remains unresolved. A unique match does not certify a
    stable-phase referee, the sample bottle or unrelated molar conversions.
    """
    if tolerance < 0:
        raise ContractError("Link tolerance cannot be negative")
    rows = []
    for index, row in observations.iterrows():
        mask = pd.Series(True, index=primary.index)
        for key in keys:
            mask &= primary[key].eq(row[key])
        mask &= (primary[concentration] - row[concentration]).abs().lt(tolerance)
        matching = primary[mask]
        unique = len(matching) == 1
        sd_match = (
            abs(matching[sd_column].iloc[0] - row[sd_column]) < tolerance
            if unique and sd_column
            else None
        )
        rows.append(
            dict(
                observation_index=index,
                n_native_matches=len(matching),
                source_table_phase=matching[phase_column].iloc[0] if unique else None,
                sd_matches_native=bool(sd_match) if sd_match is not None else None,
                assignment_basis="original table column; not numerical phase order",
            )
        )
    return pd.DataFrame(rows)
