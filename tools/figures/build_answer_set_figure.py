"""Plot frozen development output; keep task families and reference modes separate."""

from pathlib import Path
import json, hashlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from _figure_paths import configuration

FIGURE_CONFIG = configuration()
HERE = FIGURE_CONFIG.output
ROOT = FIGURE_CONFIG.audit_root / "manuscript_completion_20261009"
DATA = ROOT / "method" / "development_outputs_v1_2"
files = [
    DATA / "DEV/point/policy_scope_summary.csv",
    DATA / "NATIVE_EXACT_PRIMARY/point/policy_scope_summary.csv",
]
before = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
}
dev, tol = [pd.read_csv(p) for p in files]
policies = [
    ("numerical", "Numerical"),
    ("finite_S_numerical_only", "Finite S"),
    ("declared_D", "Declared D"),
    ("observed_D_plus_O", "D + observed O"),
]
specs = [
    ("SMZ_EXPOSED", dev, "", "Sulfamerazine: slurry phase", 13),
    ("SOLPROP_EXPOSED", dev, "", "SolProp: concentration order", 42),
    ("TOL20_NATIVE_EXACT_PRIMARY", tol, "_IT", "Tolbutamide: 48 h phase, native IT", 5),
]
rows = []
for family, frame, suffix, title, n in specs:
    for name, label in policies:
        part = frame[(frame.resource_id == family) & (frame.policy == name + suffix)]
        assert len(part) == 1
        r = part.iloc[0]
        assert r.n_tasks == n and r.n_scoreable == n
        answer = int(r.n_accepted_scoreable)
        wrong = int(r.n_errors)
        rows.append(
            dict(
                family=family,
                policy=name + suffix,
                label=label,
                n=n,
                agree=answer - wrong,
                disagree=wrong,
                unanswered=n - answer,
                risk_status="undefined" if not answer else "point_reference_only",
            )
        )
counts = pd.DataFrame(rows)
counts.to_csv(HERE / "Figure6_plotted_policy_counts.csv", index=False)
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.grid": False,
    }
)
colors = {"agree": "#2374AB", "disagree": "#D9793B", "unanswered": "#DDDFE2"}
fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.55))
fig.subplots_adjust(left=0.208, right=0.975, bottom=0.075, top=0.965, hspace=0.4)
for k, ((family, frame, suffix, title, n), ax) in enumerate(zip(specs, axes)):
    part = counts[counts.family == family]
    for i, r in enumerate(part.itertuples()):
        left = 0
        for col in ["agree", "disagree", "unanswered"]:
            value = getattr(r, col)
            ax.barh(
                3 - i,
                value,
                left=left,
                height=0.49,
                color=colors[col],
                edgecolor="white",
                linewidth=0.5,
            )
            left += value
        ax.text(
            n * 1.045,
            3 - i,
            f"{r.agree} / {r.disagree} / {r.unanswered}",
            va="center",
            ha="left",
            fontsize=9.5,
        )
    ax.set_xlim(0, n * 1.39)
    ax.set_ylim(-0.58, 4.8)
    ax.set_yticks([3, 2, 1, 0], [x[1] for x in policies])
    ax.tick_params(axis="y", length=0, labelsize=9.5)
    ax.set_xticks([0, n // 2, n])
    ax.set_xlabel("Number of questions", fontsize=10, labelpad=2)
    ax.text(
        0.014,
        0.975,
        chr(97 + k),
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.072,
        0.963,
        f"{title} (n = {n})",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    ax.legend(
        handles=[
            Patch(facecolor=colors[c], label=l)
            for c, l in [
                ("agree", "Agree"),
                ("disagree", "Disagree"),
                ("unanswered", "Unanswered"),
            ]
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.83),
        ncol=3,
        fontsize=8.7,
        frameon=False,
        handlelength=1.3,
        columnspacing=1.1,
        handletextpad=0.4,
        borderaxespad=0,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(False)
for ext in ["pdf", "svg", "png"]:
    fig.savefig(HERE / f"Figure6_comparison_support.{ext}", dpi=600)
plt.close(fig)
assert before == {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
}
manifest = {
    "source_files_sha256": before,
    "figure_inches": [7.2, 6.55],
    "panels": 3,
    "reference_mode": "point, distinct task families; no pooled error rate",
    "empty_coverage_risk": "undefined, not zero",
    "previously_exposed": True,
    "counts": rows,
}
(HERE / "Figure6_build_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {"rows_plotted": len(rows), "all_sources_unchanged": True, "counts": rows}
    )
)
