"""Plot the locked retrospective source stress without changing scientific results."""

from pathlib import Path
import json, hashlib, platform, shutil
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import font_manager
from _figure_paths import configuration

FIGURE_CONFIG = configuration()
OUT = FIGURE_CONFIG.output
SRC = (
    FIGURE_CONFIG.audit_root
    / "impact_upgrade_20261009/evaluation/tolbutamide2020_stress"
)
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"I": "#E69F00", "II": "#0072B2", "unsupported": "#E8ECEF"}
MARKERS = {"FI": "o", "FII": "s"}
font = font_manager.findfont("Arial", fallback_to_default=False)
assert Path(font).name.lower().startswith("arial")
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 11,
        "axes.labelsize": 11.5,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.35,
        "axes.grid": False,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [
        json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x
    ]


def style(ax):
    ax.grid(False, which="both", axis="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.35)
        spine.set_color("#171717")


def main():
    paths = {
        name: SRC / name
        for name in [
            "source_input_snapshot.csv",
            "native_predictions/numerical_only_IT.jsonl",
            "interpolated_predictions/numerical_only_PT.jsonl",
            "interpolated_PT_safe_inputs.jsonl",
            "separate_source_reported_phase_judge.jsonl",
            "stress_summary.json",
        ]
    }
    before = {name: sha(path) for name, path in paths.items()}
    d = pd.read_csv(paths["source_input_snapshot.csv"])
    assert len(d) == 30
    shutil.copyfile(
        paths["source_input_snapshot.csv"],
        OUT / "Figure4_source_native_observations.csv",
    )
    pred_it = {
        r["task_id"]: r
        for r in read(paths["native_predictions/numerical_only_IT.jsonl"])
    }
    pred_pt = {
        r["task_id"]: r
        for r in read(paths["interpolated_predictions/numerical_only_PT.jsonl"])
    }
    pt_inputs = read(paths["interpolated_PT_safe_inputs.jsonl"])
    judges = read(paths["separate_source_reported_phase_judge.jsonl"])
    summary = json.loads(paths["stress_summary.json"].read_text(encoding="utf-8"))
    assert (
        summary["native_exact"]["IT_answered"] == 3
        and summary["native_exact"]["IT_errors"] == 2
    )
    assert summary["interpolated_supplement"]["PT_answered"] == 2
    fig = plt.figure(figsize=(7.2, 6.65), facecolor="white")
    a = fig.add_axes([0.12, 0.585, 0.355, 0.34])
    b = fig.add_axes([0.635, 0.585, 0.335, 0.34])
    c = fig.add_axes([0.255, 0.135, 0.705, 0.295])
    for ax in [a, b, c]:
        style(ax)
    for text, xy in [("A", (0.015, 0.97)), ("B", (0.535, 0.97)), ("C", (0.015, 0.47))]:
        fig.text(*xy, text.lower(), fontsize=23, fontweight="bold", va="top", ha="left")
    for ax, method in [(a, "IT_gravimetric"), (b, "PT")]:
        z = d[d.method.eq(method)]
        if method == "PT":
            z = d[d.method.str.startswith("PT")]
        for source_phase, phase in [("FI", "I"), ("FII", "II")]:
            g = z[z.source_phase_designation.eq(source_phase)].sort_values("T_K")
            assert len(g) == (
                5 if method == "IT_gravimetric" else 9 if phase == "I" else 11
            )
            ax.plot(
                g.T_K,
                g.native_C_g_per_kg_solvent,
                color=COLORS[phase],
                linewidth=1.8,
                marker=MARKERS[source_phase],
                markersize=5.7,
                markeredgecolor="#171717",
                markeredgewidth=0.7,
                label="Form " + phase,
            )
            if method == "IT_gravimetric":
                ax.errorbar(
                    g.T_K,
                    g.native_C_g_per_kg_solvent,
                    yerr=g.reported_SE_over4samples,
                    fmt="none",
                    color=COLORS[phase],
                    ecolor=COLORS[phase],
                    elinewidth=1.3,
                    capsize=2.8,
                    capthick=1.2,
                    zorder=4,
                )
        ax.set_xlabel("Temperature (K)")
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=10.1,
            handlelength=1.5,
            borderaxespad=0.6,
            labelspacing=0.35,
        )
    a.set_ylabel("Reported concentration\n(g kg$^{-1}$)")
    a.set_xlim(276, 300.5)
    a.set_ylim(22, 88)
    a.set_xticks([278.15, 288.15, 298.15], ["278.15", "288.15", "298.15"])
    a.text(0.95, 0.06, "Isothermal", transform=a.transAxes, ha="right", fontsize=10.5)
    b.set_xlim(276, 306)
    b.set_ylim(20, 116)
    b.set_xticks([278, 288, 298, 304])
    b.text(
        0.95,
        0.06,
        "Heating 0.1 K h$^{-1}$",
        transform=b.transAxes,
        ha="right",
        fontsize=10.2,
    )
    for r in pt_inputs:
        vals = r["features"]["values"]
        if all((v is not None for v in vals.values())):
            for phase in ["I", "II"]:
                b.scatter(
                    r["context"]["temperature_K"],
                    vals[phase],
                    marker="D",
                    s=60,
                    facecolors="none",
                    edgecolors=COLORS[phase],
                    linewidths=1.6,
                    zorder=5,
                )
    conditions = sorted(judges, key=lambda r: float(r["task_id"].split("_T")[1]))
    matrix = []
    labels = ["IT native", "PT interpolated", "48 h FTIR report"]
    for j, referee in enumerate(conditions):
        tid = referee["task_id"]
        for row, (label, predictions) in enumerate(
            [(labels[0], pred_it), (labels[1], pred_pt), (labels[2], None)]
        ):
            if predictions is None:
                phase = referee["accepted_answers"][0]
            else:
                pred = predictions[tid]
                phase = (
                    None
                    if pred["abstain"]
                    else max(
                        pred["answer_probabilities"],
                        key=pred["answer_probabilities"].get,
                    )
                )
            y = 3 - row
            c.scatter(
                j,
                y,
                marker="s",
                s=830,
                facecolor=COLORS[phase or "unsupported"],
                edgecolor="#171717",
                linewidth=0.8,
            )
            c.text(
                j,
                y,
                phase or "—",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white" if phase == "II" else "#171717",
            )
            matrix.append(
                {
                    "task_id": tid,
                    "view": label,
                    "reported_T_K": float(tid.split("_T")[1]),
                    "selected_or_reported_phase": phase,
                    "support_status": "unsupported" if phase is None else "supported",
                    "point_disagreement_with_report": (
                        None
                        if phase is None or predictions is None
                        else phase not in referee["accepted_answers"]
                    ),
                    "reference_is_reported_FTIR_not_direct_equilibrium_truth": True,
                }
            )
    c.set_xlim(-0.65, 4.65)
    c.set_ylim(0.05, 3.65)
    c.set_xticks(range(5), [r["task_id"].split("_T")[1] for r in conditions])
    c.set_yticks([3, 2, 1], labels)
    c.set_xlabel("Reference assay temperature (K)")
    c.tick_params(axis="y", length=0, pad=8)
    c.legend(
        handles=[
            Patch(facecolor=COLORS["I"], label="I"),
            Patch(facecolor=COLORS["II"], label="II"),
            Patch(
                facecolor=COLORS["unsupported"],
                edgecolor="#171717",
                label="Unsupported",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=9.8,
        handlelength=1.0,
        borderpad=0.2,
        columnspacing=1.1,
    )
    pd.DataFrame(matrix).to_csv(
        OUT / "Figure4_plotted_comparison_matrix.csv", index=False
    )
    prefix = OUT / "Figure4_Tolbutamide_assay_scope"
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(
            prefix.with_suffix("." + extension),
            dpi=450 if extension == "png" else 300,
            facecolor="white",
        )
    records = {
        "runtime": platform.python_version(),
        "matplotlib": mpl.__version__,
        "Arial": font,
        "width_mm": 7.2 * 25.4,
        "height_mm": 6.65 * 25.4,
        "panels": {
            k: {
                "width_mm": ax.get_position().width * 7.2 * 25.4,
                "height_mm": ax.get_position().height * 6.65 * 25.4,
            }
            for k, ax in zip(["A", "B", "C"], [a, b, c])
        },
        "source_hashes": before,
        "source_unchanged": before == {n: sha(p) for n, p in paths.items()},
        "legacy_results_preserved": True,
        "no_calibrated_uncertainty_added": True,
        "IT_errorbars": "original reported SE over four samples; not uncertainty of phase ordering",
        "all_30_native_observations_retained": True,
        "point_matrix": matrix,
    }
    assert records["source_unchanged"]
    (OUT / "Figure4_build_manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plt.close(fig)
    print(
        json.dumps(
            {"outputs": str(prefix), "source_unchanged": True, "native_rows": len(d)}
        )
    )


if __name__ == "__main__":
    main()
