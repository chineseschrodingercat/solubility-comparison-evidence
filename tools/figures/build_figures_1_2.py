"""Source-backed Figures1/2. Original scientific analyses remain unchanged."""

from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from _figure_paths import configuration

FIGURE_CONFIG = configuration()
OUT = FIGURE_CONFIG.output
AUDIT = FIGURE_CONFIG.audit_root
REGULAR = FIGURE_CONFIG.regular_font
BOLD = FIGURE_CONFIG.bold_font
assert REGULAR.exists() and BOLD.exists()
fm.fontManager.addfont(str(REGULAR))
fm.fontManager.addfont(str(BOLD))
assert (
    Path(
        fm.findfont(fm.FontProperties(family="Arial"), fallback_to_default=False)
    ).name.lower()
    == "arial.ttf"
)
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10.5,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.35,
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 400,
    }
)
BLUE = "#0072B2"
ORANGE = "#D95F02"
RED = "#C82F31"
GREY = "#60656D"
PURPLE = "#7B3294"
DARK = "#202020"
WIDTH_MM = 190.0
HEIGHT_MM = 172.0


def style(ax, letter, title):
    ax.grid(False, which="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.35)
        spine.set_color(DARK)
    ax.tick_params(direction="out", length=4, width=1.25, labelsize=10)
    ax.text(
        -0.14,
        1.1,
        letter.lower(),
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.set_title(title, fontsize=11.5, pad=10, loc="left")


def layout():
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.03, 1.0],
        left=0.135,
        right=0.975,
        bottom=0.125,
        top=0.91,
        wspace=0.33,
        hspace=0.65,
    )
    return (
        fig,
        [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :]),
        ],
    )


def export(fig, name):
    fig.canvas.draw()
    panels = [
        dict(
            panel=chr(65 + i),
            x_mm=ax.get_position().x0 * WIDTH_MM,
            y_mm=ax.get_position().y0 * HEIGHT_MM,
            width_mm=ax.get_position().width * WIDTH_MM,
            height_mm=ax.get_position().height * HEIGHT_MM,
        )
        for i, ax in enumerate(fig.axes)
    ]
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=400, facecolor="white")
    plt.close(fig)
    return panels


def manifest(prefix, paths, panels, scope):
    (OUT / f"{prefix}_source_manifest.json").write_text(
        json.dumps(
            dict(
                width_mm=WIDTH_MM,
                height_mm=HEIGHT_MM,
                panels=panels,
                matplotlib_version=mpl.__version__,
                Arial_path=str(REGULAR),
                Arial_bold_path=str(BOLD),
                Arial_sha256=hashlib.sha256(REGULAR.read_bytes()).hexdigest(),
                sources=[
                    dict(
                        path=str(p),
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                        bytes=p.stat().st_size,
                    )
                    for p in paths
                ],
                scope=scope,
                gridlines=False,
                four_spines=True,
                PDF_fonttype=42,
                SVG_editable_text=True,
            ),
            indent=2,
        ),
        encoding="utf8",
    )


def pred(f, t):
    return f.A + f.B / t + (f.C * np.log(t) if f.model == "apelblat" else 0)


def figure1():
    p1 = AUDIT / "closure_20261008/global_membership/all_adjacent_intervals.csv"
    p2 = (
        AUDIT
        / "closure_20261008/tpa_audit/TPA_actual_DMAC_Silver_consensus_membership.csv"
    )
    p3 = AUDIT / "baselines/SC3_replay/data/interim/04_fits.csv"
    allint = pd.read_csv(p1)
    g = allint[allint.Gold].copy()
    sw = g[g.membership_switch]
    rev = sw[sw.consensus_delta_logS * sw.continuing_groups_delta_logS < 0]
    assert (
        len(g) == 4172
        and len(sw) == 525
        and (len(rev) == 87)
        and (g.groupby(["Solute_Canon", "Solvent_Canon"]).ngroups == 335)
    )
    assert (
        (rev.dT < 1).sum() == 62
        and ((rev.dT >= 1) & (rev.dT < 5)).sum() == 11
        and ((rev.dT >= 5).sum() == 14)
    )
    g.to_csv(OUT / "Figure1_panelA_all4172_intervals.csv", index=False)
    counts = pd.DataFrame(
        [
            dict(category="All Gold intervals", count=4172),
            dict(category="Source set changed", count=525),
            dict(category="Opposite contrast sign", count=87),
        ]
    )
    counts.to_csv(OUT / "Figure1_panelA_counts.csv", index=False)
    tpa = pd.read_csv(p2)
    zoom = tpa[tpa.Temperature_K >= 363.15].copy()
    assert len(zoom) == 2
    fit = pd.read_csv(p3)
    fit = fit[
        (fit.Solute_Canon == "O=C(O)c1ccc(C(=O)O)cc1")
        & (fit.Solvent_Canon == "CC(=O)N(C)C")
    ]
    fig, axs = layout()
    a, b, c = axs
    style(a, "A", "Source support in Gold")
    yy = [2, 1, 0]
    cols = [GREY, BLUE, RED]
    a.barh(yy, counts["count"], height=0.52, color=cols, edgecolor=DARK, linewidth=1.15)
    a.set_yticks(yy, ["All intervals", "Source-set\nchanges", "Opposite\nsigns"])
    a.set_xlim(0, 4900)
    a.set_ylim(-0.7, 3.1)
    a.set_xticks([0, 2000, 4000])
    a.set_xlabel("Adjacent intervals (count)")
    for y, n in zip(yy, counts["count"]):
        if n == 4172:
            a.text(
                n - 85, y, f"{n:,}", va="center", ha="right", fontsize=11, color="white"
            )
        else:
            a.text(n + 100, y, f"{n:,}", va="center", fontsize=11)
    a.text(
        0.04,
        0.965,
        "335 solute–solvent pairs",
        transform=a.transAxes,
        fontsize=10,
        va="top",
    )
    a.text(
        0.42,
        0.37,
        "Reversal gaps\n<1 K: 62\n1–<5 K: 11\n≥5 K: 14",
        transform=a.transAxes,
        fontsize=9.5,
        va="center",
    )
    style(b, "B", "TPA in DMAC: Silver labels")
    curve_rows = []
    for gid, col, ls, label in [
        (814, BLUE, "-", "2021 source fit"),
        (1321, GREY, "--", "2008 source fit"),
    ]:
        f = fit[fit.gid == gid].iloc[0]
        tt = np.linspace(max(363.1, f.T_min), min(363.24, f.T_max), 120)
        yyv = pred(f, tt)
        b.plot(tt, yyv, color=col, ls=ls, lw=2.3, label=label)
        curve_rows.extend(
            [
                dict(gid=gid, T_K=t, LogS=y, T_fit_min=f.T_min, T_fit_max=f.T_max)
                for t, y in zip(tt, yyv)
            ]
        )
    b.plot(
        zoom.Temperature_K,
        zoom.LogS_consensus,
        "s:",
        color=RED,
        lw=2.1,
        markersize=7.5,
        markeredgecolor=DARK,
        markeredgewidth=0.75,
        label="Actual Silver labels",
    )
    b.set_xlim(363.1, 363.24)
    b.set_ylim(-0.36, 0.23)
    b.set_xticks([363.1, 363.15, 363.2], ["363.10", "363.15", "363.20"])
    b.set_yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2])
    b.set_xlabel("Temperature (K)")
    b.set_ylabel("log10 C\n(mol/L pure solvent)")
    b.legend(
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        frameon=False,
        fontsize=9.1,
        handlelength=2.0,
        labelspacing=0.3,
    )
    drop = 100 * (10 ** (zoom.LogS_consensus.iloc[1] - zoom.LogS_consensus.iloc[0]) - 1)
    assert np.isclose(drop, -26.6654307629, atol=1e-06)
    b.text(
        0.98,
        0.035,
        f"{drop:.1f}% over 0.05 K",
        transform=b.transAxes,
        fontsize=9.6,
        ha="right",
        va="bottom",
    )
    pd.DataFrame(curve_rows).to_csv(
        OUT / "Figure1_panelB_TPA_fit_curves.csv", index=False
    )
    zoom.to_csv(OUT / "Figure1_panelB_TPA_native_labels.csv", index=False)
    style(c, "C", "Native label temperatures and contributors")
    tpa = tpa.sort_values("Temperature_K").reset_index(drop=True)
    xx = np.arange(len(tpa))
    member_rows = []
    for y, gid, col in [(1, 814, BLUE), (0, 1321, GREY)]:
        present = (
            tpa.contributor_gids.astype(str)
            .str.split("|")
            .apply(lambda z: str(gid) in z)
            .to_numpy()
        )
        c.scatter(
            xx[present],
            np.full(present.sum(), y),
            s=74,
            marker="s",
            color=col,
            edgecolor=DARK,
            linewidth=0.75,
            zorder=3,
        )
        c.scatter(
            xx[~present],
            np.full((~present).sum(), y),
            s=72,
            marker="x",
            color=GREY,
            linewidth=1.6,
            zorder=3,
        )
        for i, r in tpa.iterrows():
            member_rows.append(
                dict(
                    T_K=r.Temperature_K,
                    categorical_position=i,
                    gid=gid,
                    contributes=bool(present[i]),
                    n_groups=r.n_contributing_groups,
                    phase_assignment_scope=r.phase_assignment_scope,
                )
            )
    for i, n in enumerate(tpa.n_contributing_groups):
        c.text(i, 2, str(int(n)), ha="center", va="center", fontsize=10)
    c.set_yticks([0, 1, 2], ["2008 source", "2021 source", "Groups per\nlabel"])
    c.set_ylim(-0.5, 3.15)
    c.set_xlim(-0.6, len(tpa) - 0.4)
    c.set_xticks(
        xx,
        [f"{t:.2f}" for t in tpa.Temperature_K],
        rotation=55,
        ha="right",
        fontsize=9.5,
    )
    c.set_xlabel("Native label temperature (K; equally spaced for readability)")
    handles = [
        Line2D(
            [], [], marker="s", ls="", color=DARK, markersize=6, label="Contributing"
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            color=GREY,
            markersize=6,
            label="Outside fitted range",
        ),
    ]
    c.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.98),
        frameon=False,
        ncol=2,
        fontsize=9.4,
        handletextpad=0.35,
        columnspacing=1.4,
    )
    pd.DataFrame(member_rows).to_csv(OUT / "Figure1_panelC_membership.csv", index=False)
    panels = export(fig, "Figure1_source_support")
    manifest(
        "Figure1",
        [p1, p2, p3],
        panels,
        "Actual checkpoint/source fits; correlated intervals; source-composition diagnostic, not phase truth or dissolution enthalpy. TPA example is Silver. No extrapolated departing-source fit. C temperature axis is categorical and explicitly equally spaced.",
    )


def figure2():
    p1 = (
        AUDIT
        / "sprint_20261008/mixture_validation/derived_identity_overlay_576rows.csv"
    )
    p2 = (
        AUDIT
        / "sprint_20261008/mixture_validation/TMB_all144_wrong_mean_vs_correct_query345.csv"
    )
    p3 = (
        AUDIT
        / "sprint_20261008/mixture_validation/FOX7_all18_fixed_ratio_raw_to_primary_mapping.csv"
    )
    ids = pd.read_csv(p1)
    assert len(ids) == 576
    idcounts = ids.groupby("doi_key").size()
    assert sorted(idcounts.tolist()) == [252, 324]
    ic = pd.DataFrame(
        [
            dict(
                source="10.1016/j.molliq.2021.116750",
                published_molecule="2,4,5-trimethoxybenzaldehyde",
                assigned_database_molecule="3,4,5-trimethoxybenzaldehyde",
                n_records=324,
            ),
            dict(
                source="10.1016/j.molliq.2017.09.097",
                published_molecule="N-methyl trinitropyrazole",
                assigned_database_molecule="TNT",
                n_records=252,
            ),
        ]
    )
    ic.to_csv(OUT / "Figure2_panelA_identity_counts.csv", index=False)
    ids.to_csv(OUT / "Figure2_panelA_identity_576rows.csv", index=False)
    tmb = pd.read_csv(p2)
    assert len(tmb) == 144 and tmb.groupby("solvent_lo").size().eq(72).all()
    tmb["relative_shift_percent"] = 100 * tmb.wrong_mean_relative_shift
    tmb.to_csv(OUT / "Figure2_panelB_all144_counterfactual.csv", index=False)
    fox = pd.read_csv(p3)
    assert (
        len(fox) == 18
        and fox.groupby("doi").size().sort_values().tolist() == [8, 10]
        and np.allclose(fox.raw_x, fox.primary_x, atol=1e-12)
    )
    fox.to_csv(OUT / "Figure2_panelC_FOX7_native.csv", index=False)
    fig, axs = layout()
    a, b, c = axs
    style(a, "A", "Molecular attribution")
    a.barh(
        [1, 0],
        [324, 252],
        color=[BLUE, ORANGE],
        height=0.54,
        edgecolor=DARK,
        linewidth=1.2,
    )
    a.set_yticks([1, 0], ["2,4,5-TMB as\n3,4,5-TMB", "MTNP as TNT"])
    a.set_xlim(0, 405)
    a.set_ylim(-0.65, 2.0)
    a.set_xticks([0, 100, 200, 300, 400])
    a.set_xlabel("Misattributed records (count)")
    for y, n in [(1, 324), (0, 252)]:
        a.text(n + 10, y, str(n), va="center", fontsize=11)
    a.text(
        0.035,
        0.97,
        "Two original sources",
        transform=a.transAxes,
        va="top",
        fontsize=10,
    )
    style(b, "B", "TMB query counterfactual")
    ecdf = []
    for key, col, ls, label in [
        ("CC(C)=O", BLUE, "-", "Acetone–water (72)"),
        ("CN(C)C=O", ORANGE, "--", "DMF–water (72)"),
    ]:
        z = (
            tmb[tmb.solvent_lo.eq(key)]
            .sort_values("relative_shift_percent")
            .reset_index(drop=True)
        )
        x = z.relative_shift_percent.to_numpy()
        y = np.arange(1, len(z) + 1) / len(z)
        b.step(
            np.r_[-45, x, 0],
            np.r_[0, y, 1],
            where="post",
            lw=2.4,
            ls=ls,
            color=col,
            label=label,
        )
        special = ~z.B_native
        assert special.sum() in [0, 2]
        b.scatter(
            x[special],
            y[special],
            marker="x",
            color=DARK,
            s=46,
            linewidth=1.5,
            zorder=4,
        )
        for i, r in z.iterrows():
            ecdf.append(
                dict(
                    solvent_lo=key,
                    T_K=r.Temperature_K,
                    fraction=r.fraction_lo_key,
                    relative_shift_percent=x[i],
                    ECDF=y[i],
                    ACS_cell_native=bool(r.B_native),
                )
            )
    b.set_xlim(-45, 0)
    b.set_ylim(-0.03, 1.07)
    b.set_xticks([-40, -30, -20, -10, 0])
    b.set_yticks([0, 0.25, 0.5, 0.75, 1])
    b.set_xlabel("Query shift (%)")
    b.set_ylabel("Empirical fraction")
    h, l = b.get_legend_handles_labels()
    h.append(Line2D([], [], marker="x", ls="", color=DARK, markersize=6))
    l.append("2 interpolated cells")
    b.legend(
        h,
        l,
        loc="lower right",
        frameon=False,
        fontsize=8.9,
        labelspacing=0.25,
        handlelength=2.0,
        borderpad=0.2,
    )
    pd.DataFrame(ecdf).to_csv(OUT / "Figure2_panelB_ECDF_values.csv", index=False)
    style(c, "C", "FOX-7 in NMP/water: native curves")
    c.axvspan(299.85, 333.15, color="#EEECE6", alpha=0.6, zorder=0)
    for doi, col, marker, label in [
        ("10.1080/07370652.2021.2017075", BLUE, "o", "Laser: solvent addition"),
        ("10.1080/07370652.2022.2028034", ORANGE, "s", "FBRM: heating"),
    ]:
        z = fox[fox.doi.eq(doi)].sort_values("T")
        c.plot(
            z["T"],
            1000 * z.raw_x,
            color=col,
            marker=marker,
            markersize=6.5,
            lw=2.3,
            markeredgecolor=DARK,
            markeredgewidth=0.7,
            label=label,
        )
    c.set_xlim(295, 342)
    c.set_ylim(0.25, 8.6)
    c.set_xticks([300, 310, 320, 330, 340])
    c.set_yticks([1, 3, 5, 7])
    c.set_xlabel("Temperature (K)")
    c.set_ylabel("Mole-fraction solubility (×10−3)")
    c.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.97),
        frameon=False,
        fontsize=10,
        labelspacing=0.32,
    )
    c.text(
        0.98,
        0.045,
        "Solute-free NMP fraction = 0.1575",
        ha="right",
        va="bottom",
        transform=c.transAxes,
        fontsize=9.5,
    )
    panels = export(fig, "Figure2_identity_and_contrasts")
    manifest(
        "Figure2",
        [p1, p2, p3],
        panels,
        "Identity counts are correlated source records, not independent errors. TMB is a fixed arithmetic-mean counterfactual in mole fraction, not released database/SC3 consensus. All144 query cells shown; two old curated ACS omissions remain explicitly interpolated. FOX7 native data preserve different operational endpoints; shading is common T support, not uncertainty.",
    )


if __name__ == "__main__":
    figure1()
    figure2()
    print(
        json.dumps(
            dict(
                figures=["Figure1_source_support", "Figure2_identity_and_contrasts"],
                panels_each=3,
                width_mm=WIDTH_MM,
                height_mm=HEIGHT_MM,
                Arial=str(REGULAR),
            )
        )
    )
