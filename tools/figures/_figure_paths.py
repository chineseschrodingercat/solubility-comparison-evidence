"""Shared path and font configuration for the publication figures."""

from argparse import ArgumentParser
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from matplotlib import font_manager


@dataclass(frozen=True)
class FigureConfiguration:
    audit_root: Path
    output: Path
    regular_font: Path
    bold_font: Path


@lru_cache(maxsize=1)
def configuration() -> FigureConfiguration:
    parser = ArgumentParser(
        description="Render from the separate scientific audit resource"
    )
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font-regular", type=Path)
    parser.add_argument("--font-bold", type=Path)
    args = parser.parse_args()
    root, output = args.audit_root.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error("The external audit directory does not exist")
    if output == root or output.is_relative_to(root):
        parser.error("Use an output directory outside the original input tree")
    output.mkdir(parents=True, exist_ok=True)
    for supplied in (args.font_regular, args.font_bold):
        if supplied:
            if not supplied.is_file():
                parser.error("The supplied Arial font file does not exist")
            font_manager.fontManager.addfont(str(supplied))
    regular = args.font_regular or Path(
        font_manager.findfont(
            font_manager.FontProperties(family="Arial"), fallback_to_default=False
        )
    )
    bold = args.font_bold or Path(
        font_manager.findfont(
            font_manager.FontProperties(family="Arial", weight="bold"),
            fallback_to_default=False,
        )
    )
    return FigureConfiguration(root, output, regular, bold)
