"""Verify a release against an external audit archive, without copying its data.

Only verification metadata is written. Input arrays and scientific output
tables remain in the separately supplied archive or in memory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from solubility_comparisons import __version__
from solubility_comparisons.answer_sets import predict_safe_view
from solubility_comparisons.baselines import BASELINE_POLICIES, predict
from solubility_comparisons.benchmark_support import audit_released_temperatures
from solubility_comparisons.contrast_uncertainty import paired_source_diagnostics
from solubility_comparisons.evaluation import evaluate
from solubility_comparisons.io import read_jsonl, sha256
from solubility_comparisons.mixtures import (
    compare_native_windows,
    arithmetic_pooling_counterfactual,
)
from solubility_comparisons.panels import compare_reference_panels
from solubility_comparisons.phase_proxy import (
    phase_proxy_variants,
    read_gravimetric_curves,
)
from solubility_comparisons.process_screen import (
    pillong_cells,
    screen_policy_comparison,
)
from solubility_comparisons.source_support import (
    membership_audit,
    select_contributing_fits,
)


def equal_frame(actual: pd.DataFrame, reference: pd.DataFrame) -> None:
    assert_frame_equal(
        actual.reset_index(drop=True),
        reference.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )


def compare_table_file(actual: pd.DataFrame, path: Path) -> None:
    # Serialize/parse in memory to use the archived CSV's same representation.
    from io import StringIO

    equal_frame(pd.read_csv(StringIO(actual.to_csv(index=False))), pd.read_csv(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_root.resolve()
    checks: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    protected_code = {
        str(path.relative_to(root)): sha256(path)
        for path in (
            root / "impact_upgrade_20261009/evaluation/predict_baselines.py",
            root / "impact_upgrade_20261009/evaluation/evaluate.py",
            root / "manuscript_completion_20261009/method/build_answer_sets.py",
            root / "manuscript_completion_20261009/method/replay_safe_views.py",
        )
    }

    def remember(path: Path) -> None:
        inputs[str(path.relative_to(root))] = sha256(path)

    baseline = root / "impact_upgrade_20261009/evaluation"
    task_path, judge_path = baseline / "dev_inputs.jsonl", baseline / "dev_judge.jsonl"
    remember(task_path)
    remember(judge_path)
    tasks, judges = read_jsonl(task_path), read_jsonl(judge_path)
    predictions = {
        policy: [predict(task, policy) for task in tasks]
        for policy in BASELINE_POLICIES
    }
    for policy, rows in predictions.items():
        assert rows == read_jsonl(
            baseline / "dev_predictions" / (policy + ".jsonl")
        ), f"baseline {policy} payload changed"
    checks.append(
        dict(
            operation="frozen_baseline_predictions",
            status="PASS",
            files_compared=len(predictions),
            full_payload_equality=True,
        )
    )
    table_count = 0
    for mode, folder in (
        ("point", "dev_evaluation"),
        ("uncertainty_resolved", "dev_uncertainty_resolved"),
    ):
        report = evaluate(tasks, judges, predictions, mode=mode)
        for name, table in report.tables.items():
            compare_table_file(table, baseline / folder / (name + ".csv"))
            table_count += 1
    checks.append(
        dict(
            operation="frozen_baseline_evaluation",
            status="PASS",
            tables_compared=table_count,
            exact_csv_value_equality=True,
        )
    )

    finite = root / "manuscript_completion_20261009/method/development_outputs_v1_2"
    finite_files = 0
    for source in (
        "DEV_predictor_safe_views.jsonl",
        "TOL20_predictor_safe_views.jsonl",
    ):
        path = finite / source
        remember(path)
        collections = {}
        for task in read_jsonl(path):
            _, rows = predict_safe_view(task, profile="frozen-development-v1")
            cohort = task.get("analysis_scope", "DEV")
            if cohort == "INTERPOLATED_SUPPLEMENT":
                cohort = "WITHIN_RANGE_INTERPOLATED_SUPPLEMENT"
            for row in rows:
                if task.get("method_view") in {"IT", "PT"}:
                    row["policy"] += "_" + task["method_view"]
                collections.setdefault((cohort, row["policy"]), []).append(row)
        for (cohort, policy), rows in collections.items():
            original = (
                finite / "DEV_predictions" / (policy + ".jsonl")
                if cohort == "DEV"
                else finite / cohort / "predictions" / (policy + ".jsonl")
            )
            assert rows == read_jsonl(
                original
            ), f"finite {cohort}/{policy} payload changed"
            finite_files += 1
    checks.append(
        dict(
            operation="finite_safe_view_predictions",
            status="PASS",
            files_compared=finite_files,
            full_payload_equality=True,
        )
    )

    replay = root / "baselines/SC3_replay"
    fit_path = replay / "data/interim/04_fits.csv"
    remember(fit_path)
    fits = pd.read_csv(fit_path)
    key_path = (
        root / "continuation/global_identity/actual_contributing_source_metadata.csv"
    )
    remember(key_path)
    selected = select_contributing_fits(fits, pd.read_csv(key_path))
    tiers = {
        tier: pd.read_csv(replay / "data/sc3" / (tier + ".csv"))
        for tier in ("bronze", "gold", "silver")
    }
    rows, intervals, summary = membership_audit(
        selected, tiers["bronze"], tiers["gold"], tiers["silver"]
    )
    support = root / "closure_20261008/global_membership"
    compare_table_file(rows, support / "row_consensus_reproduction.csv")
    compare_table_file(intervals, support / "all_adjacent_intervals.csv")
    assert summary == json.loads((support / "summary.json").read_text(encoding="utf8"))
    checks.append(
        dict(
            operation="released_source_support",
            status="PASS",
            tables_compared=2,
            summary_equality=True,
            exact_csv_value_equality=True,
        )
    )

    solid = root / "downstream_search_20261009/solid_state"
    numeric_tasks = [task for task in tasks if task["resource_id"] == "SMZ_EXPOSED"]
    source_curves = read_gravimetric_curves(
        {
            phase: solid / f"Form_{word}_solubility_ACN_80_water_20.csv"
            for phase, word in (("I", "one"), ("II", "two"), ("V", "five"))
        }
    )
    variants = phase_proxy_variants(source_curves, numeric_tasks)
    original = pd.read_csv(solid / "SMZ_all_sensitivity_decisions.csv")
    columns = [
        "lab",
        "T_C",
        "replicate",
        "concentration_basis",
        "representation",
        "in_native_shared_range",
        "C_I_mg_ml",
        "C_II_mg_ml",
        "C_V_mg_ml",
        "numeric_I_II_candidate",
        "numeric_I_II_V_candidate",
    ]
    from io import StringIO

    equal_frame(
        pd.read_csv(StringIO(variants[columns].to_csv(index=False))), original[columns]
    )
    checks.append(
        dict(
            operation="phase_curve_variants",
            status="PASS",
            exact_value_equality=True,
            referee_not_read_by_curve_kernel=True,
        )
    )

    mixture = root / "sprint_20261008/mixture_validation"
    windows = pd.read_csv(mixture / "frozen20_identity_qualification_view.csv")
    result = compare_native_windows(
        pd.read_csv(root / "discovery_20261008/expansion/MixtureSolDB.csv"), windows
    )
    compare_table_file(
        result.point_comparisons,
        mixture / "all20_native_union_processing_comparison.csv",
    )
    constructed = arithmetic_pooling_counterfactual(result.point_comparisons)
    original = pd.read_csv(mixture / "TMB_all144_wrong_mean_vs_correct_query345.csv")
    subset = constructed[constructed.window_id.isin(original.window_id)]
    renames = {
        "pooled_x": "wrong_mean_x",
        "target_x": "correct_query345_x",
        "absolute_shift": "wrong_mean_x_shift",
        "relative_shift": "wrong_mean_relative_shift",
        "ln_pooled_x": "ln_wrong_mean_x",
        "ln_target_x": "ln_correct_query345",
    }
    columns = ["window_id", "Temperature_K", *renames.values(), "ln_shift"]
    from io import StringIO

    actual = pd.read_csv(
        StringIO(subset.rename(columns=renames)[columns].to_csv(index=False))
    )
    equal_frame(actual, original[columns])
    checks.append(
        dict(
            operation="mixture_native_windows_and_pooling",
            status="PASS",
            exact_csv_value_equality=True,
        )
    )

    screen = root / "downstream_search_20261009/decision_datasets"
    cells = pillong_cells(pd.read_csv(screen / "Pillong2017_original.csv"))
    compare_table_file(cells, screen / "Pillong_all_cells_with_missing.csv")
    policies, _, _ = screen_policy_comparison(cells)
    compare_table_file(policies, screen / "Pillong_compound_policy_comparison.csv")
    secondary, _, _ = screen_policy_comparison(cells, complete_only=True)
    reference = pd.read_csv(screen / "Pillong_complete_dissolution_compound_policy.csv")
    secondary_columns = [
        "PID",
        "top_uniform_success",
        "all_uniform_success",
        "top_minus_all",
    ]
    equal_frame(
        pd.read_csv(StringIO(secondary[secondary_columns].to_csv(index=False))),
        reference[secondary_columns],
    )
    checks.append(
        dict(
            operation="process_screen_policies",
            status="PASS",
            exact_csv_value_equality=True,
            primary_and_complete_only=True,
        )
    )

    external = root / "downstream_search_20261009/external_quantitative"
    panels, contrasts = compare_reference_panels(
        pd.read_csv(
            external / "external_all_candidate_cells.csv", float_precision="round_trip"
        ),
        fits,
    )
    compare_table_file(panels, external / "external_all_native_panels.csv")
    compare_table_file(contrasts, external / "external_all_pairwise_comparisons.csv")
    checks.append(
        dict(
            operation="external_native_panels",
            status="PASS",
            exact_csv_value_equality=True,
        )
    )

    uncertainty = root / "sprint_20261008/contrast_uncertainty"
    dispersion = paired_source_diagnostics(
        pd.read_csv(
            root
            / "discovery_20261008/contrast_methods/all_loso_reference_intervals.csv"
        ),
        fits,
    )
    compare_table_file(
        dispersion, uncertainty / "all854_eligible_raw_heldout_uncertainty_cells.csv"
    )
    checks.append(
        dict(
            operation="heldout_source_dispersion",
            status="PASS",
            exact_csv_value_equality=True,
        )
    )

    benchmark = root / "downstream_search_20261009/benchmark_propagation"
    task_tables = {
        kind: pd.read_csv(next((benchmark / "released/data").glob(f"task{kind}_*.csv")))
        for kind in (1, 2)
    }
    components, adjudications = audit_released_temperatures(
        pd.read_csv(benchmark / "BigSolDBv2.0.csv"), task_tables
    )
    compare_table_file(
        components, benchmark / "all_released_option_temperature_mapping.csv"
    )
    compare_table_file(
        adjudications, benchmark / "all_released_temperature_adjudications.csv"
    )
    checks.append(
        dict(
            operation="released_benchmark_temperature_support",
            status="PASS",
            exact_csv_value_equality=True,
        )
    )

    assert all(sha256(root / path) == digest for path, digest in inputs.items())
    assert all(sha256(root / path) == digest for path, digest in protected_code.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                package_version=__version__,
                python=platform.python_version(),
                numpy=np.__version__,
                pandas=pd.__version__,
                status="PASS",
                external_inputs_not_copied=True,
                scientific_output_tables_not_written=True,
                upstream_input_hashes=inputs,
                frozen_code_hashes=protected_code,
                frozen_inputs_and_code_unchanged=True,
                checks=checks,
            ),
            indent=2,
        ),
        encoding="utf8",
    )
    print(json.dumps(dict(status="PASS", checks=len(checks))))


if __name__ == "__main__":
    main()
