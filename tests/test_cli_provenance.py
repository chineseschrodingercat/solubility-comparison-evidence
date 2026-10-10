import json

from solubility_comparisons.baselines import predict
from solubility_comparisons.cli import main
from solubility_comparisons.io import sha256, write_jsonl


def test_evaluation_manifest_covers_prediction_content(tmp_path, phase_task, judge):
    task_file, judge_file = tmp_path / "input.jsonl", tmp_path / "judge.jsonl"
    predictions = tmp_path / "predictions"
    write_jsonl(task_file, [phase_task])
    write_jsonl(judge_file, [judge])
    prediction_file = predictions / "numerical_only.jsonl"
    write_jsonl(prediction_file, [predict(phase_task, "numerical_only")])
    output = tmp_path / "evaluation"
    assert (
        main(
            [
                "evaluate",
                "--inputs",
                str(task_file),
                "--judge",
                str(judge_file),
                "--predictions",
                str(predictions),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    hashes = {record["name"]: record["sha256"] for record in manifest["inputs"]}
    assert hashes[prediction_file.name] == sha256(prediction_file)


def test_cli_refuses_to_overwrite_prior_output(tmp_path, phase_task):
    task_file = tmp_path / "input.jsonl"
    write_jsonl(task_file, [phase_task])
    output = tmp_path / "existing"
    output.mkdir()
    assert main(["predict", "--inputs", str(task_file), "--output", str(output)]) == 2
