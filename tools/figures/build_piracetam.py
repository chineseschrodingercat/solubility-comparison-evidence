"""Complete native published conditions; one shared reported driving-force comparison."""

from pathlib import Path
import hashlib, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _figure_paths import configuration

FIGURE_CONFIG = configuration()
HERE = FIGURE_CONFIG.output
SRC = FIGURE_CONFIG.audit_root / "manuscript_completion_20261009/validation_search"
files = [
    SRC / "piracetam_capacity_predictor_inputs.csv",
    SRC / "piracetam_all12_native_process_conditions.csv",
]
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
capacity, process = [pd.read_csv(p) for p in files]
assert len(capacity) == 2 and len(process) == 12 and (process.n_vials == 80).all()
assert (
    len(
        set(process[process.solvent == "ethanol"].reported_drive_kJ_mol)
        & set(process[process.solvent == "isopropanol"].reported_drive_kJ_mol)
    )
    == 1
)
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.linewidth": 1,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
colors = {"ethanol": "#2374AB", "isopropanol": "#8156A6"}
labels = {"ethanol": "Ethanol", "isopropanol": "Isopropanol"}
markers = {"ethanol": "o", "isopropanol": "s"}
fig = plt.figure(figsize=(7.2, 6.45))
grid = fig.add_gridspec(
    2, 2, left=0.12, right=0.97, bottom=0.085, top=0.925, hspace=0.44, wspace=0.36
)
a = fig.add_subplot(grid[0, 0])
b = fig.add_subplot(grid[0, 1])
c = fig.add_subplot(grid[1, :])
for i, r in enumerate(capacity.itertuples()):
    a.bar(i, r.C_g_per_g_solvent, width=0.55, color=colors[r.solvent])
    a.text(
        i,
        r.C_g_per_g_solvent + 0.0011,
        f"{r.C_g_per_g_solvent:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
a.set_xticks([0, 1], ["Ethanol", "Isopropanol"])
a.set_ylim(0, 0.044)
a.set_xlim(-0.6, 1.6)
a.set_ylabel("Form-VI solubility (g/g solvent)")
a.set_xlabel("288.15 K")
for solvent, g in process.groupby("solvent", sort=False):
    g = g.sort_values("reported_drive_kJ_mol")
    b.plot(
        g.reported_drive_kJ_mol,
        g.median_visual_induction_time_VI_s,
        marker=markers[solvent],
        markersize=4.5,
        color=colors[solvent],
        label=labels[solvent],
        linewidth=1.1,
    )
    fraction = g.Form_VI_fraction
    c.errorbar(
        g.reported_drive_kJ_mol,
        fraction,
        yerr=np.vstack(
            [
                fraction - g.source_style_Wilson95_VI_low,
                g.source_style_Wilson95_VI_high - fraction,
            ]
        ),
        fmt=markers[solvent] + "-",
        markersize=4.5,
        color=colors[solvent],
        label=labels[solvent],
        linewidth=1.1,
        capsize=3,
        elinewidth=0.8,
    )
    match = g[g.reported_drive_kJ_mol == 1.67].iloc[0]
    b.plot(
        1.67,
        match.median_visual_induction_time_VI_s,
        "D",
        markersize=7,
        color=colors[solvent],
        markeredgecolor="black",
        markeredgewidth=0.6,
    )
    c.plot(
        1.67,
        match.Form_VI_fraction,
        "D",
        markersize=7,
        color=colors[solvent],
        markeredgecolor="black",
        markeredgewidth=0.6,
    )
    text_y = 0.99 if solvent == "isopropanol" else 0.69
    c.annotate(
        f"{int(match['Form VI'])}/80",
        xy=(1.67, match.Form_VI_fraction),
        xytext=(1.83, text_y),
        fontsize=10,
        color=colors[solvent],
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1),
        arrowprops=dict(arrowstyle="-", color=colors[solvent], linewidth=0.7),
    )
for ax in [b, c]:
    ax.axvline(1.67, color="#777777", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlim(1.36, 2.11)
    ax.set_xlabel("Reported driving force (kJ/mol)")
    ax.legend(loc="upper right" if ax is b else "lower left", fontsize=9, frameon=False)
b.set_ylim(0, 6500)
b.set_ylabel("Median VI visual induction time (s)")
c.set_ylim(0.55, 1.03)
c.set_ylabel("Source-classified form-VI fraction")
c.text(
    1.67,
    0.568,
    "Only native shared value",
    ha="center",
    fontsize=9,
    color="#555555",
    bbox=dict(facecolor="white", edgecolor="none", pad=1),
)
for k, ax in enumerate([a, b, c]):
    ax.text(
        -0.16 if ax is not c else -0.078,
        1.025,
        chr(97 + k),
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(False)
for ext in ["pdf", "svg", "png"]:
    fig.savefig(HERE / f"SupplementaryFigure2_piracetam_objectives.{ext}", dpi=600)
plt.close(fig)
assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
capacity.to_csv(HERE / "SupplementaryFigure2_native_capacity.csv", index=False)
process.to_csv(HERE / "SupplementaryFigure2_all12_process_conditions.csv", index=False)
(HERE / "SupplementaryFigure2_build_manifest.json").write_text(
    json.dumps(
        {
            "source_sha256": hashes,
            "all12_plotted": True,
            "shared_drive_kJ_mol": 1.67,
            "phase_interval_scope": "source-style marginal Wilson; shared stock/bath structure, not source-level guarantees",
            "induction_time_conditional_on_VI": True,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(json.dumps({"all12_conditions_plotted": True, "source_files_unchanged": True}))
