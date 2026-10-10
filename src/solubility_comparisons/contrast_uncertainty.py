"""Source-fit dispersion sensitivities with held-out raw temperature contrasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ContractError
from .source_support import PAIR_COLUMNS, SourceFit


def paired_source_diagnostics(
    references: pd.DataFrame, fits: pd.DataFrame, *, min_sources: int = 2
) -> pd.DataFrame:
    if min_sources < 2:
        raise ContractError(
            "At least two common source fits are required for dispersion"
        )
    selected = fits[
        fits.model.isin(["apelblat", "vanthoff"]) & fits.R2.ge(0.8) & fits.RMSE.le(0.3)
    ]
    groups = {
        key: [SourceFit.from_record(row) for row in frame.itertuples()]
        for key, frame in selected.groupby(PAIR_COLUMNS)
    }
    rows = []
    for ref in references[references.n_common >= min_sources].itertuples():
        train = [
            fit
            for fit in groups[(ref.Solute_Canon, ref.Solvent_Canon)]
            if fit.group_id != ref.heldout_gid
            and fit.lower <= ref.T0 < ref.T1 <= fit.upper
        ]
        if len(train) != ref.n_common or any(
            fit.group_id == ref.heldout_gid for fit in train
        ):
            raise ContractError(
                "Declared common support or held-out exclusion disagrees with fits"
            )
        a = np.array([fit.evaluate(ref.T0) for fit in train])
        b = np.array([fit.evaluate(ref.T1) for fit in train])
        mean = float(np.mean(b - a))
        if abs(mean - ref.common_source) >= 1e-10:
            raise ContractError("Recorded common-source point estimate not reproduced")
        independent = float(np.sqrt(np.var(a, ddof=1) + np.var(b, ddof=1)))
        paired = float(np.std(b - a, ddof=1))
        covariance = float(np.cov(a, b, ddof=1)[0, 1])
        if abs(paired**2 - (independent**2 - 2 * covariance)) >= 1e-10:
            raise ContractError("Paired source covariance identity failed")
        row = dict(
            Solute_Canon=ref.Solute_Canon,
            Solvent_Canon=ref.Solvent_Canon,
            heldout_gid=ref.heldout_gid,
            T0=ref.T0,
            T1=ref.T1,
            n_common=len(train),
            train_common_gids="|".join(str(fit.group_id) for fit in train),
            Gold=ref.Gold,
            Silver=ref.Silver,
            reference_delta=ref.reference_delta,
            P_mean=ref.pointwise,
            C_mean=mean,
            sd_independent=independent,
            sd_paired=paired,
            source_covariance=covariance,
            paired_to_independent_ratio=(
                paired / independent if independent > 0 else np.nan
            ),
        )
        for name, scale in (("independent", independent), ("paired", paired)):
            error = abs(mean - ref.reference_delta)
            row[name + "_error_over_scale"] = (
                error / scale if scale > 0 else 0 if error == 0 else np.inf
            )
            for multiplier in (1, 2):
                row[f"{name}_contains_raw_at{multiplier}"] = error <= multiplier * scale
                row[f"{name}_width_at{multiplier}"] = 2 * multiplier * scale
            decision = 1 if mean - 3 * scale > 0 else -1 if mean + 3 * scale < 0 else 0
            row[name + "_decision3"] = decision
            row[name + "_raw_sign"] = int(np.sign(ref.reference_delta))
            row[name + "_sign_error3"] = decision != 0 and decision != np.sign(
                ref.reference_delta
            )
        rows.append(row)
    return pd.DataFrame(rows)
