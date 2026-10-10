"""Render each figure family in its own process to isolate plotting settings."""

from pathlib import Path
import shutil
import subprocess
import sys
from _figure_paths import configuration


def main() -> None:
    cfg = configuration()
    scripts = (
        "build_figures_1_2.py",
        "build_figures_3_4.py",
        "build_figure5.py",
        "build_tolbutamide.py",
        "build_answer_set_figure.py",
        "build_piracetam.py",
    )
    for script in scripts:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / script),
                "--audit-root",
                str(cfg.audit_root),
                "--output",
                str(cfg.output),
                "--font-regular",
                str(cfg.regular_font),
                "--font-bold",
                str(cfg.bold_font),
            ],
            check=True,
        )
    for extension in ("png", "pdf", "svg"):
        shutil.copy2(
            cfg.output / ("Figure4_process_and_phase." + extension),
            cfg.output / ("SupplementaryFigure1_process_and_phase." + extension),
        )
    print("Rendered six main figures and two supplementary figures")


if __name__ == "__main__":
    main()
