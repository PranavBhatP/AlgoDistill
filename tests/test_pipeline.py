"""Unit tests for End-to-End Distillation Pipeline and Execution Evaluator."""

import pytest
from src.pipeline.e2e_pipeline import EndToEndDistillationPipeline, PipelineResult
from src.eval.execution_eval import ExecutionEvaluator
from src.data.dataset_builder import DatasetItem
from src.judge.sandbox_judge import Verdict


def make_test_item(problem_id: str, tag: str) -> DatasetItem:
    return DatasetItem(
        problem_id=problem_id,
        title=f"Pipeline Test {problem_id}",
        ground_truth_tag=tag,
        rating=1000,
        statement=f"Given an array, calculate sum modulo 10^9+7 using {tag}.",
        input_spec="Line 1: N integers",
        output_spec="Single integer",
        teacher_rationale=f"Step 1: parse input. Step 2: compute using {tag}.",
        teacher_solution="import sys\n\ndef solve():\n    print(10)\n\nif __name__ == '__main__':\n    solve()\n",
        teacher_stated_tag=tag,
        sample_tests=[{"input": "1 2 3 4\n", "output": "10\n"}]
    )


def test_end_to_end_pipeline():
    pipeline = EndToEndDistillationPipeline()
    item = make_test_item("PIPE_1", "dynamic programming")
    pipeline.stage2.train([item])

    result: PipelineResult = pipeline.run(
        problem_statement=item.statement,
        problem_id=item.problem_id,
        sample_tests=item.sample_tests,
        input_spec=item.input_spec,
        output_spec=item.output_spec,
        run_baseline=True
    )

    assert result.predicted_tag is not None
    assert result.conditioned_result.code is not None
    assert result.conditioned_result.verdict == Verdict.AC.value
    assert result.baseline_result is not None
    assert result.total_pipeline_time_ms > 0


def test_execution_evaluator():
    pipeline = EndToEndDistillationPipeline()
    items = [
        make_test_item("EVAL_1", "dynamic programming"),
        make_test_item("EVAL_2", "greedy")
    ]
    pipeline.stage2.train(items)

    evaluator = ExecutionEvaluator(pipeline)
    report = evaluator.evaluate_dataset(items)

    assert report.total_evaluated == 2
    assert report.conditioned_pass_at_1 >= 0.0
    assert report.baseline_pass_at_1 >= 0.0
    assert "AC" in report.conditioned_verdict_counts
