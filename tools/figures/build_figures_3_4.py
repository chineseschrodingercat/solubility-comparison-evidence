"""Build Figures 3–4 from audited, unchanged source tables.



Run with the existing Anaconda Python. No external data download is performed.

"""

from pathlib import Path
import hashlib, json, shutil, sys
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import ScalarFormatter, NullLocator
from matplotlib.font_manager import findfont
from _figure_paths import configuration

FIGURE_CONFIG = configuration()
OUT = FIGURE_CONFIG.output
SRC = FIGURE_CONFIG.audit_root / "downstream_search_20261009/solid_state"
OUT.mkdir(parents=True, exist_ok=True)
FONT = findfont("Arial", fallback_to_default=False)
FONT_BOLD = findfont(
    mpl.font_manager.FontProperties(family="Arial", weight="bold"),
    fallback_to_default=False,
)
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "axes.linewidth": 1.35,
        "axes.edgecolor": "#171717",
        "text.color": "#171717",
        "axes.grid": False,
        "grid.alpha": 0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)
PHASE = {"I": "#E69F00", "II": "#0072B2", "V": "#CC79A7"}
MARK = {"I": "o", "II": "s", "V": "^"}
AGREE = "#E8EFF4"
CONFLICT = "#D55E00"
MIXED = "#6A51A3"
NEW = "#009E73"
inputs = {
    "Figure3_source_clearpoint_55rows.csv": "SMZ_clearpoint_rows_preserved.csv",
    "Figure3_source_terminal_13records.csv": "SMZ_primary_eligible_decisions.csv",
    "Figure3_source_sensitivity_104records.csv": "SMZ_all_sensitivity_decisions.csv",
    "Figure4_source_tolfenamic_color_counts.csv": "tolfenamic_13_14_SI_actual_color_counts.csv",
    "Figure4_source_tolfenamic_conditions.csv": "tolfenamic_13_14_native_matched_conditions.csv",
    "Figure4_source_ritonavir_18points.csv": "ritonavir_all18_crash_cooling_Fig4_points.csv",
    "Figure4_source_ritonavir_complete_matrix.csv": "ritonavir_complete_solvent_endpoint_matrix.csv",
}
for dst, source in inputs.items():
    shutil.copyfile(SRC / source, OUT / dst)
raw = pd.read_csv(OUT / "Figure3_source_clearpoint_55rows.csv")
terminal = pd.read_csv(OUT / "Figure3_source_terminal_13records.csv")
sens = pd.read_csv(OUT / "Figure3_source_sensitivity_104records.csv")
terminal = terminal.sort_values(["lab", "T_C", "replicate"]).reset_index(drop=True)
contexts = list(zip(terminal.lab, terminal.T_C, terminal.replicate))
context_labels = [
    f"{int(t)}{chr(96 + int(rep))}" if lab == "Innsbruck" else str(int(t))
    for lab, t, rep in contexts
]
variant_order = [
    ("reported_C_mg_ml", "log_piecewise", "Reported: log"),
    ("reported_C_mg_ml", "linear_piecewise", "Reported: linear"),
    ("reported_C_mg_ml", "log_global_fit", "Reported: log fit"),
    ("reported_C_mg_ml", "log_piecewise_first_cycle", "Reported: first cycle"),
    ("mass_volume_C_mg_ml", "log_piecewise", "m/V: log"),
    ("mass_volume_C_mg_ml", "linear_piecewise", "m/V: linear"),
    ("mass_volume_C_mg_ml", "log_global_fit", "m/V: log fit"),
    ("mass_volume_C_mg_ml", "log_piecewise_first_cycle", "m/V: first cycle"),
]
matrix = []
for basis, rep, label in variant_order:
    d = sens[sens.concentration_basis.eq(basis) & sens.representation.eq(rep)]
    lookup = {
        (r.lab, r.T_C, r.replicate): r.numeric_I_II_V_candidate
        != r.observed_terminal_phase
        for r in d.itertuples()
    }
    assert len(lookup) == 13
    matrix.append([lookup[c] for c in contexts])
matrix = np.array(matrix, dtype=bool)
common = matrix.all(axis=0)
assert len(raw) == 55 and len(terminal) == 13 and (len(sens) == 104)
assert matrix[0].sum() == 4 and common.sum() == 4
assert (terminal.numeric_I_II_candidate != terminal.observed_terminal_phase).sum() == 8
matrix_source = pd.DataFrame(
    matrix.astype(int), columns=[f"{lab}|{t}|{rep}" for lab, t, rep in contexts]
)
matrix_source.insert(0, "variant", [x[2] for x in variant_order])
matrix_source.to_csv(OUT / "Figure3_plotted_conflict_matrix.csv", index=False)


def style(ax):
    ax.grid(False, which="both", axis="both")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#171717")
        s.set_linewidth(1.35)
    ax.tick_params(top=False, right=False)


def inside_legend(ax, handles, **kwargs):
    lg = ax.legend(
        handles=handles,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
        borderpad=0.32,
        handlelength=1.25,
        handletextpad=0.5,
        labelspacing=0.4,
        **kwargs,
    )
    return lg


def phase_handles():
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=MARK[p],
            markersize=7,
            markerfacecolor=PHASE[p],
            markeredgecolor="#171717",
            markeredgewidth=0.7,
            label=p,
        )
        for p in PHASE
    ]


def letter(fig, ax, text, x=None):
    bb = ax.get_position()
    fig.text(
        bb.x0 - 0.205 if x is None else x,
        bb.y1 + 0.009,
        text.lower(),
        fontfamily="Arial",
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


qa = {
    "runtime": sys.executable,
    "matplotlib": mpl.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "Arial_regular": FONT,
    "Arial_bold": FONT_BOLD,
    "figures": {},
}


def finish(fig, name, data_points_for_legends):
    fig.canvas.draw()
    info = {
        "size_inches": fig.get_size_inches().tolist(),
        "size_mm": (fig.get_size_inches() * 25.4).tolist(),
        "axes": [],
        "legend_point_overlap_checks": [],
    }
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        bb = ax.get_position()
        info["axes"].append(
            {
                "width_in": float(bb.width * fig.get_figwidth()),
                "height_in": float(bb.height * fig.get_figheight()),
                "all_four_spines": all((s.get_visible() for s in ax.spines.values())),
                "any_visible_gridline": any(
                    (x.get_visible() for x in ax.get_xgridlines() + ax.get_ygridlines())
                ),
            }
        )
    for ax, points in data_points_for_legends:
        lg = ax.get_legend()
        if lg is None:
            continue
        box = lg.get_window_extent(renderer)
        xy = ax.transData.transform(np.asarray(points, dtype=float))
        overlaps = [i for i, p in enumerate(xy) if box.contains(*p)]
        info["legend_point_overlap_checks"].append(
            {
                "panel": ax.get_title(),
                "n_points_checked": len(xy),
                "overlap_indices": overlaps,
            }
        )
        assert not overlaps, (name, ax.get_title(), overlaps)
    assert not any((a["any_visible_gridline"] for a in info["axes"]))
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=400)
    qa["figures"][name] = info
    plt.close(fig)


fig = plt.figure(figsize=(7.8, 7.8))
a = fig.add_axes([0.115, 0.515, 0.365, 0.415])
b = fig.add_axes([0.69, 0.515, 0.27, 0.415])
c = fig.add_axes([0.23, 0.09, 0.73, 0.3])
for ax in [a, b, c]:
    style(ax)
for ax, l, xpos in [(a, "A", 0.035), (b, "B", 0.6), (c, "C", 0.035)]:
    letter(fig, ax, l, x=xpos)
a.set_title("Dynamic clear-point measurements", pad=9)
points_a = []
for phase in PHASE:
    d = raw[raw.form.eq(phase)].sort_values("Tclear_C")
    a.plot(
        d.Tclear_C, d.reported_C_mg_ml, color=PHASE[phase], lw=1.7, alpha=0.85, zorder=1
    )
    a.scatter(
        d.Tclear_C,
        d.mass_volume_C_mg_ml,
        s=43,
        marker=MARK[phase],
        facecolors="white",
        edgecolors=PHASE[phase],
        lw=1.15,
        zorder=2,
    )
    a.scatter(
        d.Tclear_C,
        d.reported_C_mg_ml,
        s=20,
        marker=MARK[phase],
        c=PHASE[phase],
        edgecolors="#171717",
        lw=0.45,
        zorder=3,
    )
    points_a.extend(zip(d.Tclear_C, d.mass_volume_C_mg_ml))
    points_a.extend(zip(d.Tclear_C, d.reported_C_mg_ml))
a.set_yscale("log")
a.set_xlim(5, 71.5)
a.set_ylim(5, 85)
a.set_yticks([5, 10, 20, 40, 80])
a.yaxis.set_major_formatter(ScalarFormatter())
a.yaxis.set_minor_locator(NullLocator())
a.set_xlabel("Clear-point temperature (°C)")
a.set_ylabel("Concentration (mg/mL)")
ha = phase_handles() + [
    Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="#171717",
        markeredgecolor="#171717",
        markersize=4.5,
        label="Reported C (filled)",
    ),
    Line2D(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor="#171717",
        markersize=7,
        label="Mass/volume (open)",
    ),
]
inside_legend(a, ha, loc="upper left")
b.set_title("Proxies and terminal PXRD", pad=9)
points_b = []
for x, col in [
    (0, "numeric_I_II_candidate"),
    (1, "numeric_I_II_V_candidate"),
    (2, "observed_terminal_phase"),
]:
    for j, r in terminal.iterrows():
        p = r[col]
        b.scatter(
            x,
            j,
            c=PHASE[p],
            marker=MARK[p],
            s=76,
            edgecolor="#171717",
            linewidth=0.7,
            zorder=3,
        )
        points_b.append((x, j))
b.set_xlim(-0.6, 2.6)
b.set_ylim(12.7, -1.85)
b.set_yticks(
    range(13),
    [
        "IB " + v if lab == "Innsbruck" else "UC " + v
        for (lab, t, rep), v in zip(contexts, context_labels)
    ],
)
b.set_xticks([0, 1, 2], ["I/II\nproxy", "I/II/V\nproxy", "Terminal\nPXRD"])
b.set_ylabel("Laboratory and temperature (°C)")
inside_legend(
    b,
    phase_handles(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    columnspacing=0.45,
)
c.set_title("Three-form proxy sensitivity", pad=9)
for row in range(8):
    for col in range(13):
        v = matrix[row, col]
        c.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1,
                1,
                facecolor=CONFLICT if v else AGREE,
                edgecolor="none",
                linewidth=0,
                antialiased=False,
            )
        )
        c.text(
            col,
            row,
            "×" if v else "·",
            ha="center",
            va="center",
            color="white" if v else "#374957",
            fontsize=15,
            fontweight="bold",
        )
c.set_xlim(-0.5, 12.5)
c.set_ylim(7.5, -1.85)
c.set_yticks(range(8), [v[2] for v in variant_order])
c.set_xticks(range(13), context_labels)
for lbl, flag in zip(c.get_xticklabels(), common):
    if flag:
        lbl.set_color(CONFLICT)
        lbl.set_fontweight("bold")
c.set_xlabel("Same 13 records; red labels conflict in every variant")
inside_legend(
    c,
    [
        Patch(facecolor=AGREE, edgecolor="none", label="Match"),
        Patch(facecolor=CONFLICT, edgecolor="none", label="Conflict"),
    ],
    loc="upper right",
    ncol=2,
    columnspacing=0.8,
)
finish(
    fig,
    "Figure3_SMZ_phase_evidence",
    [(a, points_a), (b, points_b), (c, [(j, i) for i in range(8) for j in range(13)])],
)
tolf = pd.read_csv(OUT / "Figure4_source_tolfenamic_color_counts.csv").sort_values(
    "expt"
)
rit = pd.read_csv(OUT / "Figure4_source_ritonavir_18points.csv")
rit_matrix = pd.read_csv(OUT / "Figure4_source_ritonavir_complete_matrix.csv")
rit4 = rit_matrix[rit_matrix.SLE_form_I_x.notna()].copy()
solvents = ["acetone", "ethyl acetate", "acetonitrile", "toluene"]
short = ["Acetone", "EtOAc", "MeCN", "Toluene"]
assert len(rit) == 18 and len(tolf) == 2 and (len(rit4) == 4)
fig, axs = plt.subplots(2, 2, figsize=(8.0, 7.4))
fig.subplots_adjust(
    left=0.115, right=0.975, bottom=0.1, top=0.93, wspace=0.36, hspace=0.43
)
a, b, c, d = axs.ravel()
for ax, l in zip(axs.ravel(), "ABCD"):
    style(ax)
    letter(fig, ax, l, x=ax.get_position().x0 - 0.086)
a.set_title("First-visible phase composition", pad=9)
counts = tolf[["pure_white_I", "pure_yellow_II", "mixed_I_II"]].to_numpy()
n = counts.sum(axis=1)
share = counts / n[:, None] * 100
bottom = np.zeros(2)
for i, (name, color) in enumerate(
    [("I", PHASE["I"]), ("II", PHASE["II"]), ("Mixed", MIXED)]
):
    a.bar(
        [0, 1],
        share[:, i],
        bottom=bottom,
        width=0.58,
        color=color,
        edgecolor="#171717",
        lw=0.8,
        label=name,
    )
    for j in range(2):
        if share[j, i] >= 9:
            a.text(
                j,
                bottom[j] + share[j, i] / 2,
                f"{counts[j, i]:.0f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=11,
            )
    bottom += share[:, i]
a.set_xlim(-0.58, 1.58)
a.set_ylim(0, 123)
a.set_yticks([0, 25, 50, 75, 100])
a.set_xticks(
    [0, 1],
    [f"Reused · series 13\n{int(n[0])} cycles", f"New · series 14\n{int(n[1])} cycles"],
)
a.set_ylabel("Crystallized cycles (%)")
inside_legend(
    a,
    [
        Patch(facecolor=PHASE["I"], edgecolor="#171717", label="I"),
        Patch(facecolor=PHASE["II"], edgecolor="#171717", label="II"),
        Patch(facecolor=MIXED, edgecolor="#171717", label="Mixed"),
    ],
    loc="upper center",
    ncol=3,
    columnspacing=0.75,
)
composition = tolf.copy()
for i, name in enumerate(["share_I_pct", "share_II_pct", "share_mixed_pct"]):
    composition[name] = share[:, i]
composition["n_crystallized_cycles"] = n
composition.to_csv(OUT / "Figure4_plotted_tolfenamic_composition.csv", index=False)
b.set_title("Mixture-assignment sensitivity", pad=9)
rules = ["Mixed to II", "Mixed to I", "Mixed excluded"]
sens_tolf = []
points_b = []
for i, r in tolf.reset_index(drop=True).iterrows():
    values = []
    for rule in rules:
        I = r.pure_white_I + (r.mixed_I_II if rule == "Mixed to I" else 0)
        II = r.pure_yellow_II + (r.mixed_I_II if rule == "Mixed to II" else 0)
        v = II / (I + II) * 100
        values.append(v)
        sens_tolf.append(
            {
                "series": r.expt,
                "mixture_rule": rule,
                "I_cycles": I,
                "II_cycles": II,
                "II_pct_of_classifiable_crystallized_cycles": v,
            }
        )
    b.plot(
        range(3), values, color="#171717" if i == 0 else NEW, lw=1.6, ls=":", zorder=1
    )
    b.scatter(
        range(3),
        values,
        s=58,
        marker="^" if i == 0 else "D",
        c="#171717" if i == 0 else NEW,
        edgecolor="#171717",
        lw=0.65,
        zorder=3,
    )
    points_b.extend(zip(range(3), values))
b.set_xlim(-0.35, 2.35)
b.set_ylim(0, 111)
b.set_yticks([0, 25, 50, 75, 100])
b.set_xticks(range(3), ["Mixed\nto II", "Mixed\nto I", "Mixed\nexcluded"])
b.set_ylabel("Assigned form II (%)")
inside_legend(
    b,
    [
        Line2D([], [], marker="^", c="#171717", ls=":", markersize=6, label="Reused"),
        Line2D([], [], marker="D", c=NEW, ls=":", markersize=6, label="New"),
    ],
    loc="center right",
    bbox_to_anchor=(0.98, 0.63),
)
pd.DataFrame(sens_tolf).to_csv(
    OUT / "Figure4_plotted_tolfenamic_sensitivity.csv", index=False
)
c.set_title("Digitized crash-cooling outcomes", pad=9)
points_c = []
for outcome, marker, color in [
    ("I", "o", PHASE["I"]),
    ("II", "s", PHASE["II"]),
    ("no_crystal_2_days", "^", "#5B6770"),
]:
    sub = rit[rit.crash_cooling_result.eq(outcome)]
    xx = [solvents.index(s) for s in sub.solvent]
    yy = sub.S_FormI_plot_digitized.to_numpy()
    c.scatter(
        xx, yy, s=64, marker=marker, c=color, edgecolor="#171717", lw=0.7, zorder=3
    )
    points_c.extend(zip(xx, yy))
c.set_xlim(-0.5, 3.5)
c.set_ylim(0.85, 6.7)
c.set_yticks(range(1, 7))
c.set_xticks(range(4), short)
c.set_ylabel("Form-I supersaturation, S")
c.set_xlabel("Pure solvent")
inside_legend(
    c,
    [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=PHASE["I"],
            markeredgecolor="#171717",
            label="I",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markerfacecolor=PHASE["II"],
            markeredgecolor="#171717",
            label="II",
        ),
        Line2D(
            [],
            [],
            marker="^",
            linestyle="none",
            markerfacecolor="#5B6770",
            markeredgecolor="#171717",
            label="None",
        ),
    ],
    loc="upper center",
    ncol=3,
    columnspacing=0.7,
)
d.set_title("Solvent-order preservation", pad=9)
rank_rows = []
points_d = []
for i, s in enumerate(solvents):
    r = rit4[rit4.solvent.eq(s)].iloc[0]
    capacity_rank = int(r.capacity_rank_descending)
    drive_rank = int(rit4.required_drive_J_mol.rank(method="min").loc[r.name])
    assert capacity_rank == drive_rank == i + 1
    d.scatter(
        i - 0.065,
        capacity_rank,
        s=64,
        c="#171717",
        marker="o",
        edgecolor="#171717",
        lw=0.7,
        zorder=3,
    )
    d.scatter(
        i + 0.065,
        drive_rank,
        s=74,
        facecolor="white",
        edgecolor=NEW,
        marker="D",
        lw=1.5,
        zorder=4,
    )
    d.text(
        i,
        capacity_rank - 0.27,
        f"{r.SLE_form_I_x * 1000:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    d.text(
        i,
        drive_rank + 0.27,
        f"{r.required_drive_J_mol:.0f}",
        ha="center",
        va="top",
        fontsize=10,
        color=NEW,
        fontweight="bold",
    )
    points_d.extend([(i - 0.065, capacity_rank), (i + 0.065, drive_rank)])
    rank_rows.append(
        {
            "solvent": s,
            "x_form_I": r.SLE_form_I_x,
            "capacity_rank_highest_first": capacity_rank,
            "required_drive_J_mol": r.required_drive_J_mol,
            "drive_rank_lowest_first": drive_rank,
            "equal_induction_time_s": 13320,
            "scope": "Separate kinetic protocol; fitted equal-time criterion, not a yield or Fig4 outcome",
        }
    )
d.set_xlim(-0.5, 3.5)
d.set_ylim(4.7, 0.35)
d.set_yticks(range(1, 5))
d.set_xticks(range(4), short)
d.set_ylabel("Rank (highest x / lowest drive first)")
d.set_xlabel("Pure solvent")
inside_legend(
    d,
    [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#171717",
            markeredgecolor="#171717",
            label="Capacity (x × 10³)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=NEW,
            markeredgewidth=1.5,
            label="Drive (J/mol)",
        ),
    ],
    loc="lower left",
    bbox_to_anchor=(0.015, 0.025),
    fontsize=10,
)
pd.DataFrame(rank_rows).to_csv(
    OUT / "Figure4_plotted_ritonavir_rank_preservation.csv", index=False
)
finish(
    fig,
    "Figure4_process_and_phase",
    [(a, [(0, 100), (1, 100)]), (b, points_b), (c, points_c), (d, points_d)],
)
legend3 = "**Figure 3 | Terminal-phase evidence limits thermodynamic judgments from clear-point concentrations.**\n\n**A,** All 55 sulfamerazine clear-point records for forms I (24), II (17) and V (14) in MeCN/water (80:20 v/v). Filled symbols show reported concentrations; open symbols show mass/volume interpretations of the same records, not additional measurements or uncertainty bounds. Lines guide the eye; concentration is logarithmic. **B,** Frozen two-form and three-form point proxies versus terminal PXRD for all 13 eligible longest-time records: ten duplicate Innsbruck vials (IB, a/b) and three UCL conditions (UC). The proxies conflict with PXRD in eight and four records, respectively. **C,** All eight concentration-basis/representation variants. Red temperature labels identify four records conflicting in every variant: IB 45°C in duplicate and UC 46/48°C. Log/linear denote piecewise interpolation; log fit denotes a global ln(concentration) fit; first cycle excludes recrystallized I cycles. These are one molecule, two laboratories and one collaborative study, not 13 independent studies. Dynamic phase retention has sparse PXRD temperature anchors; terminal judgments derive from long ternary slurries. No unreported uncertainty or formal interval is added. Source: 10.1021/acs.cgd.5c01406.\n\n"
legend4 = "**Figure 4 | Process context and endpoint identity delimit crystallization comparisons.**\n\n**A,** Tolfenamic-acid first-visible pure I, pure II and mixed I/II counts in IPA for reused, acetone-cleaned vials (series 13) and new vials (series 14). Percentages condition on crystallization: 157 and 213 cycles; three noncrystallized series-14 cycles are retained in source tables. Counts label larger segments. Colors were PXRD-calibrated with random vial checks, not per-vial terminal equilibrium assignments. **B,** Assigned form-II percentages under all three mixture conventions. Dotted lines guide the eye across categories. Repeated vial cycles are not independent surfaces; this is a descriptive process contrast. **C,** All 18 ritonavir crash-cooling screen points at 283.15 K, with supersaturation digitized from the published plot. “None” denotes no crystallization after two days. Source phases were characterized after recovery/drying; the screen used a 50°C/90-min hold and 20°C/min cooling. **D,** Four-solvent form-I solubility-capacity ranks and ascending fitted driving-force ranks at an equal induction time of 13,320 s are identical. Annotations give x×10³ (black) and J/mol (green). This kinetic protocol differs from C and is not joined to its plotted S≈3.1 outcomes. Solubility residues were reported form I after 2 h at 10°C. No yield or unreported uncertainty is estimated. Sources: 10.1021/acs.cgd.6c00631; 10.1021/acs.molpharmaceut.4c00234.\n\n"
(OUT / "Figure3_SMZ_phase_evidence_legend.md").write_text(legend3, encoding="utf-8")
(OUT / "Figure4_process_and_phase_legend.md").write_text(legend4, encoding="utf-8")
(OUT / "Figure3_4_render_metadata.json").write_text(
    json.dumps(qa, indent=2), encoding="utf-8"
)
manifest = {"inputs": [], "outputs": []}
for dst, source in inputs.items():
    p = OUT / dst
    manifest["inputs"].append(
        {
            "copied_file": dst,
            "original_file": str(SRC / source),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    )
for p in sorted(OUT.iterdir()):
    if (
        p.is_file()
        and (
            p.name.startswith(("Figure3", "Figure4"))
            or p.name == "build_figures_3_4.py"
        )
        and (p.name != "Figure3_4_source_and_output_hashes.json")
    ):
        manifest["outputs"].append(
            {
                "file": p.name,
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
(OUT / "Figure3_4_source_and_output_hashes.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
print(json.dumps(qa, indent=2))
