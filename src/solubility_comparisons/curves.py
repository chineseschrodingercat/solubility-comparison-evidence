"""Native support is checked before interpolation or empirical curve evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .contracts import ContractError

Representation = Literal[
    "linear_piecewise", "log_piecewise", "log_global_fit", "log10_piecewise"
]


@dataclass(frozen=True)
class NativeCurve:
    temperatures: np.ndarray
    concentrations: np.ndarray

    @classmethod
    def from_samples(
        cls,
        temperatures: Sequence[float],
        concentrations: Sequence[float],
        *,
        duplicate: str = "error",
    ) -> NativeCurve:
        frame = pd.DataFrame(
            dict(temperature=temperatures, concentration=concentrations)
        )
        if (
            frame.empty
            or not np.isfinite(frame.to_numpy(dtype=float)).all()
            or (frame.concentration <= 0).any()
        ):
            raise ContractError(
                "Native curves require finite temperatures and positive concentrations"
            )
        if duplicate not in {"error", "mean_exact_temperature"}:
            raise ContractError("Unknown duplicate handling rule")
        if frame.temperature.duplicated().any() and duplicate == "error":
            raise ContractError(
                "Duplicate native temperature: resolve state/record identity explicitly"
            )
        if duplicate == "mean_exact_temperature":
            frame = (
                frame.groupby("temperature", sort=True)
                .concentration.mean()
                .reset_index()
            )
        else:
            frame = frame.sort_values("temperature")
        return cls(
            frame.temperature.to_numpy(dtype=float),
            frame.concentration.to_numpy(dtype=float),
        )

    @property
    def domain(self) -> tuple[float, float]:
        return float(self.temperatures.min()), float(self.temperatures.max())

    def supports(self, temperature: float) -> bool:
        lo, hi = self.domain
        return bool(np.isfinite(temperature) and lo <= temperature <= hi)

    def evaluate(
        self, temperature: float, representation: Representation = "log_piecewise"
    ) -> float:
        """Return concentration, except log10_piecewise returns log10 concentration.

        The latter is explicitly a contrast-coordinate representation used by
        mixture-window diagnostics; it must not be mixed with a native C value.
        """
        if not self.supports(temperature):
            raise ContractError(
                "Temperature outside the native domain; no extrapolation"
            )
        if representation == "linear_piecewise":
            return float(np.interp(temperature, self.temperatures, self.concentrations))
        if representation == "log_piecewise":
            return float(
                np.exp(
                    np.interp(
                        temperature, self.temperatures, np.log(self.concentrations)
                    )
                )
            )
        if representation == "log10_piecewise":
            return float(
                np.interp(temperature, self.temperatures, np.log10(self.concentrations))
            )
        if representation == "log_global_fit":
            if len(self.temperatures) < 2:
                raise ContractError(
                    "A fitted temperature response needs at least two native temperatures"
                )
            fit = np.polyfit(self.temperatures, np.log(self.concentrations), 1)
            return float(np.exp(np.polyval(fit, temperature)))
        raise ContractError(f"Unknown curve representation {representation!r}")


def common_native_union(curve_a: NativeCurve, curve_b: NativeCurve) -> np.ndarray:
    lo, hi = max(curve_a.domain[0], curve_b.domain[0]), min(
        curve_a.domain[1], curve_b.domain[1]
    )
    if lo > hi:
        return np.array([], dtype=float)
    values = np.unique(np.r_[curve_a.temperatures, curve_b.temperatures])
    return values[(values >= lo) & (values <= hi)]
