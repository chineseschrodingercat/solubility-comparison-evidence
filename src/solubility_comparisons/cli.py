"""Command-line entry points operate on explicit, externally supplied records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .answer_sets import predict_safe_view
from .baselines import BASELINE_POLICIES, predict
from .benchmark_support import audit_released_temperatures
from .config import load_config
from .contracts import ContractError, unique_index, validate_task
from .contrast_uncertainty import paired_source_diagnostics
from .evaluation import evaluate
from .identity import link_original_columns
from .io import file_manifest, fresh_directory, read_jsonl, write_jsonl
from .mixtures import arithmetic_pooling_counterfactual, compare_native_windows
from .panels import compare_reference_panels
from .phase_proxy import phase_proxy_variants
from .process_endpoints import compare_native_objectives
from .process_screen import pillong_cells, screen_policy_comparison
from .quantities import g100g_to_pure_solvent_molar, mole_fraction_to_pure_solvent_molar
from .source_support import membership_audit, select_contributing_fits


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def _inputs(path: Path) -> list[dict[str, Any]]:
    rows = [validate_task(row) for row in read_jsonl(path)]
    unique_index(rows)
    return rows


def _predictions(directory: Path) -> dict[str, list[dict[str, Any]]]:
    result = {path.stem: read_jsonl(path) for path in sorted(directory.glob("*.jsonl"))}
    if not result:
        raise ContractError("Prediction directory contains no JSONL policy files")
    return result


def _write_baselines(
    tasks: list[dict[str, Any]], directory: Path, policies: Sequence[str]
) -> None:
    for policy in policies:
        write_jsonl(
            directory / f"{policy}.jsonl", [predict(task, policy) for task in tasks]
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solubility-compare", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional version1 JSON/TOML; relative paths resolve beside it",
    )
    sub = parser.add_subparsers(dest="operation", required=True)
    paths = {
        "predict": ("inputs",),
        "evaluate": ("inputs", "judge", "predictions"),
        "replay": ("inputs", "judge"),
        "answer-sets": ("inputs",),
        "membership": ("fits", "keys", "bronze", "gold", "silver"),
        "mixtures": ("raw", "windows"),
        "phase-proxy": ("observations", "inputs"),
        "process-screen": ("raw",),
        "benchmark-support": ("raw", "task1", "task2"),
        "panels": ("cells", "fits"),
        "contrast-dispersion": ("references", "fits"),
        "identity-link": ("observations", "primary"),
        "process-objectives": ("capacities", "conditions"),
        "convert-quantity": ("raw",),
    }
    for operation, names in paths.items():
        command = sub.add_parser(operation)
        for name in names:
            command.add_argument("--" + name, type=Path)
        command.add_argument(
            "--output",
            type=Path,
            help="New output directory; existing directories are refused",
        )
    for name in ("predict", "replay"):
        sub.choices[name].add_argument(
            "--policy",
            choices=BASELINE_POLICIES,
            action="append",
            help="Default: all archived baselines",
        )
    sub.choices["evaluate"].add_argument(
        "--mode", choices=("point", "uncertainty_resolved"), default="point"
    )
    sub.choices["answer-sets"].add_argument(
        "--profile", choices=("declared", "frozen-development-v1"), default="declared"
    )
    sub.choices["process-screen"].add_argument("--complete-only", action="store_true")
    sub.choices["contrast-dispersion"].add_argument(
        "--min-sources", type=int, default=2
    )
    identity = sub.choices["identity-link"]
    identity.add_argument("--keys", nargs="+", required=True)
    identity.add_argument("--concentration", required=True)
    identity.add_argument("--phase-column", required=True)
    identity.add_argument("--sd-column")
    identity.add_argument("--tolerance", type=float, default=5e-8)
    conversion = sub.choices["convert-quantity"]
    conversion.add_argument(
        "--basis", choices=("mole-fraction", "g100g-solvent"), required=True
    )
    conversion.add_argument("--value-column", required=True)
    conversion.add_argument("--density-column", required=True)
    conversion.add_argument("--molar-mass-column", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config) if args.config else {}
        for name, path in config.get("paths", {}).items():
            if not hasattr(args, name):
                raise ContractError(
                    f"Path {name!r} is not accepted by {args.operation}"
                )
            if getattr(args, name) is None:
                setattr(args, name, path)
        required = {
            "predict": ("inputs",),
            "evaluate": ("inputs", "judge", "predictions"),
            "replay": ("inputs", "judge"),
            "answer-sets": ("inputs",),
            "membership": ("fits", "keys", "bronze", "gold", "silver"),
            "mixtures": ("raw", "windows"),
            "phase-proxy": ("observations", "inputs"),
            "process-screen": ("raw",),
            "benchmark-support": ("raw", "task1", "task2"),
            "panels": ("cells", "fits"),
            "contrast-dispersion": ("references", "fits"),
            "identity-link": ("observations", "primary"),
            "process-objectives": ("capacities", "conditions"),
            "convert-quantity": ("raw",),
        }[args.operation]
        if args.output is None or any(getattr(args, key) is None for key in required):
            raise ContractError(
                "Supply --output and all named input paths, directly or through config"
            )
        input_paths = [getattr(args, key) for key in required if key != "predictions"]
        if args.operation == "evaluate":
            input_paths.extend(sorted(args.predictions.glob("*.jsonl")))
        if not all(path.is_file() for path in input_paths):
            raise ContractError("A named input file does not exist")
        output = fresh_directory(args.output)
        initial_inputs = file_manifest(input_paths)
        if args.operation in {"predict", "replay"}:
            tasks = _inputs(args.inputs)
            prediction_directory = output / "predictions"
            _write_baselines(
                tasks, prediction_directory, args.policy or BASELINE_POLICIES
            )
            if args.operation == "replay":
                predictions = _predictions(prediction_directory)
                judges = read_jsonl(args.judge)
                for mode in ("point", "uncertainty_resolved"):
                    evaluate(tasks, judges, predictions, mode=mode).write(output / mode)
        elif args.operation == "evaluate":
            evaluate(
                _inputs(args.inputs),
                read_jsonl(args.judge),
                _predictions(args.predictions),
                mode=args.mode,
            ).write(output)
        elif args.operation == "answer-sets":
            tasks = [validate_task(row) for row in read_jsonl(args.inputs)]
            collections: dict[tuple[str, str], list[dict[str, Any]]] = {}
            sets = []
            for task in tasks:
                answers, predictions = predict_safe_view(task, profile=args.profile)
                sets.append(
                    dict(
                        task_id=task["task_id"],
                        analysis_scope=task.get("analysis_scope", "DEV"),
                        method_view=task.get("method_view"),
                        **answers,
                    )
                )
                scope = task.get("analysis_scope", "DEV")
                if scope == "INTERPOLATED_SUPPLEMENT":
                    scope = "WITHIN_RANGE_INTERPOLATED_SUPPLEMENT"
                for prediction in predictions:
                    if task.get("method_view") in {"IT", "PT"}:
                        prediction["policy"] += "_" + task["method_view"]
                    collections.setdefault((scope, prediction["policy"]), []).append(
                        prediction
                    )
            for (scope, policy), predictions in collections.items():
                unique_index(predictions)
                write_jsonl(output / scope / f"{policy}.jsonl", predictions)
            write_jsonl(output / "answer_sets.jsonl", sets)
        elif args.operation == "membership":
            fits = select_contributing_fits(
                pd.read_csv(args.fits), pd.read_csv(args.keys)
            )
            rows, intervals, summary = membership_audit(
                fits,
                pd.read_csv(args.bronze),
                pd.read_csv(args.gold),
                pd.read_csv(args.silver),
            )
            rows.to_csv(output / "row_consensus_reproduction.csv", index=False)
            intervals.to_csv(output / "all_adjacent_intervals.csv", index=False)
            intervals[intervals.membership_switch].to_csv(
                output / "membership_switch_intervals.csv", index=False
            )
            _dump(output / "summary.json", summary)
        elif args.operation == "mixtures":
            result = compare_native_windows(
                pd.read_csv(args.raw), pd.read_csv(args.windows)
            )
            for name, table in (
                ("native_rows", result.native_rows),
                ("point_comparisons", result.point_comparisons),
                ("response_segments", result.response_segments),
                ("windows", result.windows),
            ):
                table.to_csv(output / f"{name}.csv", index=False)
            arithmetic_pooling_counterfactual(result.point_comparisons).to_csv(
                output / "pooling_counterfactual.csv", index=False
            )
        elif args.operation == "phase-proxy":
            phase_proxy_variants(
                pd.read_csv(args.observations), _inputs(args.inputs)
            ).to_csv(output / "phase_proxy_variants.csv", index=False)
        elif args.operation == "process-screen":
            cells = pillong_cells(pd.read_csv(args.raw))
            policies, pairs, summary = screen_policy_comparison(
                cells, complete_only=args.complete_only
            )
            cells.to_csv(output / "all_cells.csv", index=False)
            policies.to_csv(output / "compound_policies.csv", index=False)
            pairs.to_csv(output / "same_grade_different_outcomes.csv", index=False)
            _dump(output / "summary.json", summary)
        elif args.operation == "benchmark-support":
            components, panels = audit_released_temperatures(
                pd.read_csv(args.raw),
                {1: pd.read_csv(args.task1), 2: pd.read_csv(args.task2)},
            )
            components.to_csv(
                output / "all_released_option_temperature_mapping.csv", index=False
            )
            panels.to_csv(
                output / "all_released_temperature_adjudications.csv", index=False
            )
        elif args.operation == "panels":
            panels, comparisons = compare_reference_panels(
                pd.read_csv(args.cells), pd.read_csv(args.fits)
            )
            panels.to_csv(output / "native_panels.csv", index=False)
            comparisons.to_csv(output / "pairwise_comparisons.csv", index=False)
        elif args.operation == "contrast-dispersion":
            paired_source_diagnostics(
                pd.read_csv(args.references),
                pd.read_csv(args.fits),
                min_sources=args.min_sources,
            ).to_csv(output / "source_contrast_diagnostics.csv", index=False)
        elif args.operation == "identity-link":
            link_original_columns(
                pd.read_csv(args.observations),
                pd.read_csv(args.primary),
                keys=args.keys,
                concentration=args.concentration,
                phase_column=args.phase_column,
                sd_column=args.sd_column,
                tolerance=args.tolerance,
            ).to_csv(output / "source_column_links.csv", index=False)
        elif args.operation == "process-objectives":
            compare_native_objectives(
                pd.read_csv(args.capacities), pd.read_csv(args.conditions)
            ).to_csv(output / "native_objective_comparisons.csv", index=False)
        elif args.operation == "convert-quantity":
            frame = pd.read_csv(args.raw)
            conversion = (
                mole_fraction_to_pure_solvent_molar
                if args.basis == "mole-fraction"
                else g100g_to_pure_solvent_molar
            )
            frame["mol_per_L_pure_solvent"] = [
                conversion(
                    row[args.value_column],
                    row[args.density_column],
                    row[args.molar_mass_column],
                )
                for row in frame.to_dict("records")
            ]
            frame.to_csv(output / "converted_quantities.csv", index=False)
        _dump(
            output / "run_manifest.json",
            dict(
                package_version=__version__,
                operation=args.operation,
                created_utc=datetime.now(timezone.utc).isoformat(),
                inputs=initial_inputs,
                role="Caller supplies source/evidence/exposure contracts; software does not certify semantic provenance",
            ),
        )
        print(f"Completed {args.operation}: {output}")
        return 0
    except (ContractError, FileExistsError, KeyError, ValueError) as exc:
        print(f"Input/output error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
