"""Build Figure 5 from unchanged, already audited public-data analyses.

No scientific eligibility, rank, or outcome rule is learned or changed here.
Run with the documented plotting dependencies and explicit external paths.
"""

from pathlib import Path
import hashlib
import json
import platform
import shutil
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
from _figure_paths import configuration

FIGURE_CONFIG = configuration()
OUT = FIGURE_CONFIG.output
ROOT = FIGURE_CONFIG.audit_root / "downstream_search_20261009"
FONT = FIGURE_CONFIG.regular_font
assert FONT.is_file(), "Required Windows Arial font is unavailable."
font_manager.fontManager.addfont(str(FONT))
assert font_manager.FontProperties(fname=str(FONT)).get_name() == "Arial"
resolved_font = Path(
    font_manager.findfont(
        font_manager.FontProperties(family="Arial"), fallback_to_default=False
    )
)
assert resolved_font.name.lower().startswith("arial"), str(resolved_font)
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelsize": 10.4,
        "axes.labelweight": "bold",
        "axes.titlesize": 10.6,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9.6,
        "ytick.labelsize": 9.6,
        "axes.linewidth": 1.35,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "axes.unicode_minus": True,
    }
)
INPUTS = {
    "A_primary": "decision_datasets/Pillong_compound_policy_comparison.csv",
    "A_complete": "decision_datasets/Pillong_complete_dissolution_compound_policy.csv",
    "B_cells": "decision_datasets/Pillong_observed_valid_cells.csv",
    "C_contrasts": "external_quantitative/external_all_pairwise_comparisons.csv",
    "D_adjudications": "benchmark_propagation/all_released_temperature_adjudications.csv",
    "D_native_options": "benchmark_propagation/all_released_option_temperature_mapping.csv",
}
COPY_NAMES = {
    "A_primary": "Figure5_source_A_primary_compound_policies.csv",
    "A_complete": "Figure5_source_A_complete_compound_policies.csv",
    "B_cells": "Figure5_source_B_observed_cells.csv",
    "C_contrasts": "Figure5_source_C_external_contrasts.csv",
    "D_adjudications": "Figure5_source_D_temperature_adjudications.csv",
    "D_native_options": "Figure5_source_D_option_native_temperatures.csv",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


input_hashes = {}
for name, rel in INPUTS.items():
    src = ROOT / rel
    dst = OUT / COPY_NAMES[name]
    shutil.copyfile(src, dst)
    assert sha256(src) == sha256(dst)
    input_hashes[rel] = {"sha256": sha256(src), "copied_to": dst.name}
primary = pd.read_csv(OUT / COPY_NAMES["A_primary"])
complete = pd.read_csv(OUT / COPY_NAMES["A_complete"])
cells = pd.read_csv(OUT / COPY_NAMES["B_cells"])
contrasts = pd.read_csv(OUT / COPY_NAMES["C_contrasts"])
tasks = pd.read_csv(OUT / COPY_NAMES["D_adjudications"])
assert len(primary) == primary.PID.nunique() == 319
assert len(complete) == complete.PID.nunique() == 303
assert len(cells) == 5710 and cells.PID.nunique() == 319
assert len(contrasts) == 42 and contrasts.panel_id.nunique() == 15
assert contrasts.strict_solute.nunique() == contrasts.doi.nunique() == 4
direction_agrees = np.sign(contrasts.reference_delta) == np.sign(contrasts.P_delta)
assert direction_agrees.sum() == 41
assert contrasts[["reference_delta", "P_delta"]].notna().all().all()
ecdfs = []
for name, frame in [
    ("Primary—all observed", primary),
    ("Exploratory—complete dissolution", complete),
]:
    x = np.sort(100 * frame.top_minus_all.to_numpy())
    for rank, value in enumerate(x, 1):
        ecdfs.append(
            {
                "population": name,
                "n_compounds": len(x),
                "difference_pp": value,
                "rank": rank,
                "ecdf": rank / len(x),
            }
        )
pd.DataFrame(ecdfs).to_csv(OUT / "Figure5_source_A_ecdf_points.csv", index=False)
s_order = ["RS", "KS", "TS", "PS"]
c_order = ["XX", "YX", "CT", "DR", "FI", "AM"]
counts = pd.crosstab(cells.S_code, cells.C_code).reindex(
    index=s_order, columns=c_order, fill_value=0
)
assert counts.to_numpy().sum() == 5710
totals = counts.sum(axis=1)
assert totals.tolist() == [2473, 240, 651, 2346]
distribution = counts.stack().rename("n_cells").reset_index()
distribution["category_total"] = distribution.S_code.map(totals)
distribution["percent_cells"] = 100 * distribution.n_cells / distribution.category_total
distribution.to_csv(OUT / "Figure5_source_B_outcome_distribution.csv", index=False)
task_summary = []
for task, d in tasks.groupby("task", sort=True):
    eligible = d.target_all_source_curves_available.astype(bool)
    changed = d.loc[eligible, "target_answer_changed"].astype(bool).sum()
    task_summary.append(
        {
            "task": int(task),
            "released_questions": len(d),
            "same_native_T": int((~d.heterogeneous_native_T.astype(bool)).sum()),
            "different_native_T": int(d.heterogeneous_native_T.astype(bool).sum()),
            "different_native_T_percent": float(100 * d.heterogeneous_native_T.mean()),
            "max_native_T_span_K": float(d.native_T_span_K.max()),
            "conditionable_questions": int(eligible.sum()),
            "changed_conditioned_answers": int(changed),
            "all_native_target_questions": int(d.target_all_native.astype(bool).sum()),
            "unavailable_questions": int((~eligible).sum()),
        }
    )
task_summary = pd.DataFrame(task_summary)
assert task_summary.released_questions.tolist() == [3699, 878]
assert task_summary.different_native_T.tolist() == [516, 142]
assert task_summary.conditionable_questions.tolist() == [3618, 856]
assert task_summary.changed_conditioned_answers.tolist() == [0, 0]
assert task_summary.unavailable_questions.tolist() == [81, 22]
task_summary.to_csv(OUT / "Figure5_source_D_task_summary.csv", index=False)
BLUE, ORANGE, GREEN, PURPLE, RED = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#7A3E9D",
    "#C92A42",
)
fig = plt.figure(figsize=(190 / 25.4, 184 / 25.4), facecolor="white")
axes = [
    fig.add_axes((0.115, 0.595, 0.345, 0.335)),
    fig.add_axes((0.645, 0.595, 0.33, 0.335)),
    fig.add_axes((0.115, 0.12, 0.345, 0.35)),
    fig.add_axes((0.645, 0.12, 0.33, 0.35)),
]
for letter, ax in zip("ABCD", axes):
    ax.grid(False, which="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.35)
        spine.set_color("#171717")
    ax.tick_params(direction="out", colors="#171717")
    ax.text(
        -0.215,
        1.035,
        letter.lower(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        weight="bold",
        fontsize=22,
        clip_on=False,
    )
ax = axes[0]
for frame, color, line, label in [
    (primary, BLUE, "-", "Primary (319)"),
    (complete, ORANGE, "--", "Complete (303)"),
]:
    x = np.sort(100 * frame.top_minus_all.to_numpy())
    ax.step(
        np.r_[-100, x, 100],
        np.r_[0, np.arange(1, len(x) + 1) / len(x), 1],
        where="post",
        color=color,
        lw=2.2,
        ls=line,
        label=label,
    )
ax.axvline(0, color="#6C6C6C", lw=1.1, ls=":", zorder=0)
ax.set(
    xlim=(-103, 103),
    ylim=(-0.015, 1.055),
    xticks=[-100, -50, 0, 50, 100],
    yticks=[0, 0.25, 0.5, 0.75, 1.0],
)
ax.set_xlabel("Policy success difference (pp)", labelpad=7)
ax.set_ylabel("Fraction of compounds", labelpad=7)
legend_A = ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.02, 0.985),
    frameon=False,
    fontsize=9.0,
    borderaxespad=0,
    handlelength=1.6,
    handletextpad=0.45,
    labelspacing=0.5,
)
ax.text(
    0.56,
    0.13,
    "Mean: −6.08 pp\nComplete: −4.67 pp",
    transform=ax.transAxes,
    fontsize=9.2,
    ha="left",
    va="bottom",
    linespacing=1.45,
)
ax = axes[1]
colors = [BLUE, GREEN, "#CC79A7", ORANGE, "#7A7A7A", "#D55E00"]
y = np.arange(3, -1, -1)
left = np.zeros(4)
for code, color in zip(c_order, colors):
    percent = 100 * counts[code].to_numpy() / totals.to_numpy()
    ax.barh(
        y,
        percent,
        left=left,
        height=0.65,
        facecolor=color,
        edgecolor="white",
        linewidth=0.65,
        label=code,
    )
    left += percent
assert np.allclose(left, 100)
ax.set(xlim=(0, 100), ylim=(-0.55, 5.0), xticks=[0, 25, 50, 75, 100], yticks=y)
ax.set_yticklabels([f"{code}\nn = {int(n):,}" for code, n in zip(s_order, totals)])
ax.set_xlabel("Observed cells within category (%)", labelpad=7)
ax.set_ylabel("Dissolution category", labelpad=7)
legend_B = ax.legend(
    handles=[Patch(facecolor=c, edgecolor=c, label=k) for k, c in zip(c_order, colors)],
    loc="upper left",
    bbox_to_anchor=(0.03, 0.975),
    ncol=3,
    frameon=False,
    fontsize=9.0,
    handlelength=1.2,
    handletextpad=0.45,
    columnspacing=1.4,
    borderaxespad=0,
    labelspacing=0.5,
)
ax = axes[2]
ax.plot([-0.84, 0.68], [-0.84, 0.68], color="#6C6C6C", lw=1.2, ls=":", zorder=0)
ax.axvline(0, color="#B0B0B0", lw=0.8, zorder=0)
ax.axhline(0, color="#B0B0B0", lw=0.8, zorder=0)
ax.scatter(
    contrasts.loc[direction_agrees, "reference_delta"],
    contrasts.loc[direction_agrees, "P_delta"],
    s=40,
    color=BLUE,
    edgecolor="#172F3A",
    linewidth=0.6,
    alpha=0.82,
    zorder=3,
)
ax.scatter(
    contrasts.loc[~direction_agrees, "reference_delta"],
    contrasts.loc[~direction_agrees, "P_delta"],
    s=150,
    marker="*",
    color=RED,
    edgecolor="#7D1124",
    linewidth=0.8,
    zorder=4,
)
ax.set(
    xlim=(-0.84, 0.68),
    ylim=(-0.84, 0.68),
    xticks=[-0.8, -0.4, 0, 0.4],
    yticks=[-0.8, -0.4, 0, 0.4],
)
ax.set_xlabel("Reference contrast (Δlog10 c)", labelpad=7)
ax.set_ylabel("Pointwise contrast (Δlog10 c)", labelpad=7)
ax.text(
    0.035,
    0.95,
    "42 contrasts; 15 panels\n4 solutes; 4 source labels\nTop candidate: 15/15",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9.2,
    linespacing=1.35,
)
legend_C = ax.legend(
    handles=[
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            color=BLUE,
            mec="#172F3A",
            ms=6,
            label="Agree (41)",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="*",
            color=RED,
            mec="#7D1124",
            ms=10,
            label="Oppose (1)",
        ),
    ],
    loc="lower right",
    bbox_to_anchor=(0.99, 0.035),
    frameon=False,
    fontsize=8.9,
    borderaxespad=0,
    handletextpad=0.45,
    labelspacing=0.6,
)
ax = axes[3]
bar_y = np.array([1.45, 0.45])
different = task_summary.different_native_T_percent.to_numpy()
same = 100 - different
ax.barh(bar_y, same, height=0.57, color=BLUE, edgecolor="white", linewidth=0.7)
ax.barh(
    bar_y,
    different,
    left=same,
    height=0.57,
    color=PURPLE,
    edgecolor="white",
    linewidth=0.7,
)
for yy, fraction, count in zip(bar_y, different, task_summary.different_native_T):
    ax.text(
        100 - fraction / 2,
        yy,
        str(count),
        color="white",
        fontsize=9.1,
        ha="center",
        va="center",
        weight="bold",
    )
ax.set(xlim=(0, 100), ylim=(-1.3, 2.75), xticks=[0, 25, 50, 75, 100], yticks=bar_y)
ax.set_yticklabels(["Pairwise\nn = 3,699", "Best solvent\nn = 878"])
ax.set_xlabel("Released questions (%)", labelpad=7)
legend_D = ax.legend(
    handles=[
        Patch(facecolor=BLUE, label="Same native T"),
        Patch(facecolor=PURPLE, label="Different native T"),
    ],
    loc="upper left",
    bbox_to_anchor=(0.02, 0.985),
    frameon=False,
    fontsize=9.0,
    handlelength=1.3,
    handletextpad=0.5,
    borderaxespad=0,
    labelspacing=0.45,
)
ax.text(
    0.025,
    0.19,
    "Changed after conditioning\nPairwise: 0/3,618; best: 0/856\nUnavailable: 81; 22",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9.15,
    linespacing=1.35,
)
base = OUT / "Figure5_decision_targets"
for suffix, options in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {})]:
    fig.savefig(base.with_suffix("." + suffix), **options)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
legend_geometry = {}
for letter, ax, legend in zip("ABCD", axes, [legend_A, legend_B, legend_C, legend_D]):
    box = legend.get_window_extent(renderer)
    local = box.transformed(ax.transAxes.inverted())
    assert (
        local.x0 >= -1e-06
        and local.y0 >= -1e-06
        and (local.x1 <= 1 + 1e-06)
        and (local.y1 <= 1 + 1e-06)
    ), (letter, local.bounds)
    legend_geometry[letter] = list(local.bounds)
    if letter == "A":
        assert (
            local.x1 < 0.5
        ), "Panel A legend intrudes on the central ECDF/reference region."
provenance = {
    "figure": "Figure5_decision_targets",
    "panel_count": 4,
    "physical_width_mm": 190,
    "physical_height_mm": 184,
    "raster_dpi": 600,
    "font": {
        "family": "Arial",
        "resolved_path": str(resolved_font),
        "sha256": sha256(resolved_font),
    },
    "software": {
        "python": sys.version,
        "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    },
    "original_inputs": input_hashes,
    "axes_plot_areas_mm": [
        dict(width=ax.get_position().width * 190, height=ax.get_position().height * 184)
        for ax in axes
    ],
    "legend_axes_fraction_bboxes": legend_geometry,
    "means_pp": {
        "primary_319": float(100 * primary.top_minus_all.mean()),
        "exploratory_complete_303": float(100 * complete.top_minus_all.mean()),
    },
    "panel_B_cells": 5710,
    "panel_B_unique_compounds": 319,
    "panel_C_contrasts": 42,
    "panel_C_panels": 15,
    "panel_C_sign_agreement": 41,
    "panel_D": task_summary.to_dict("records"),
    "uncertainty": "No independent uncertainty or confidence intervals added.",
    "outputs": {
        p.name: sha256(p)
        for p in sorted(OUT.glob("Figure5_*"))
        if p.is_file()
        and p.name not in {"Figure5_provenance.json", "Figure5_artifact_manifest.json"}
        and ("review" not in p.name.lower())
    },
    "build_script_sha256": sha256(__file__),
}
(OUT / "Figure5_provenance.json").write_text(
    json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {
            "outputs": str(base),
            "means_pp": provenance["means_pp"],
            "legend_inside_axes": True,
            "task_summary": task_summary.to_dict("records"),
        },
        ensure_ascii=False,
        indent=2,
    )
)
plt.close(fig)
