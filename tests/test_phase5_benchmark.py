"""Unit and integration tests for Phase 5: Empirical Valley Mapping & Pass@k Benchmarking."""

import os
import pytest
import numpy as np
from src.data.dataset_builder import DatasetItem
from src.eval.metrics import compute_pass_at_k, PassAtKReport
from src.eval.benchmark_sweep import BenchmarkSweepEngine, BenchmarkSweepResults
from src.utils.visualization import Visualizer


def make_dummy_item(problem_id: str, tag: str) -> DatasetItem:
    return DatasetItem(
        problem_id=problem_id,
        title=f"Benchmark Test {problem_id}",
        ground_truth_tag=tag,
        rating=1200,
        statement=f"Given an array, compute optimal value using {tag}.",
        input_spec="Line 1: N integers",
        output_spec="Single integer",
        teacher_rationale=f"Apply {tag} strategy step by step.",
        teacher_solution="import sys\n\ndef solve():\n    print(42)\n\nif __name__ == '__main__':\n    solve()\n",
        teacher_stated_tag=tag,
        sample_tests=[{"input": "5\n1 2 3 4 5\n", "output": "42\n"}]
    )


def test_pass_at_k_unbiased_estimator():
    # Test case 1: all 5 passed -> Pass@1=1.0, Pass@5=1.0
    res1 = {"P1": [True, True, True, True, True]}
    rep1 = compute_pass_at_k(res1)
    assert rep1.pass_at_1 == 1.0
    assert rep1.pass_at_5 == 1.0
    assert rep1.solved_problems == 1

    # Test case 2: none passed -> Pass@1=0.0, Pass@5=0.0
    res2 = {"P2": [False, False, False, False, False]}
    rep2 = compute_pass_at_k(res2)
    assert rep2.pass_at_1 == 0.0
    assert rep2.pass_at_5 == 0.0
    assert rep2.solved_problems == 0

    # Test case 3: 1 of 5 passed -> Pass@1 = 0.2, Pass@5 = 1.0 (since min 1 sample passed among 5)
    res3 = {"P3": [True, False, False, False, False]}
    rep3 = compute_pass_at_k(res3)
    assert rep3.pass_at_1 == 0.2
    assert rep3.pass_at_5 == 1.0

    # Test case 4: Failure taxonomy passed
    verdicts = {"P3": ["AC", "WA", "TLE", "RE", "CE"]}
    rep4 = compute_pass_at_k(res3, verdicts_per_problem=verdicts)
    assert rep4.failure_taxonomy["AC"] == 1
    assert rep4.failure_taxonomy["WA"] == 1
    assert rep4.failure_taxonomy["TLE"] == 1
    assert rep4.failure_taxonomy["RE"] == 1
    assert rep4.failure_taxonomy["CE"] == 1


def test_benchmark_sweep_engine_fast_mode(tmp_path):
    results_path = os.path.join(tmp_path, "test_sweep_results.json")
    items = [
        make_dummy_item("TEST_1", "dynamic programming"),
        make_dummy_item("TEST_2", "greedy"),
        make_dummy_item("TEST_3", "math")
    ]

    engine = BenchmarkSweepEngine(
        mode="fast",
        samples_per_problem=3,
        results_path=results_path
    )

    results: BenchmarkSweepResults = engine.run_sweep(
        test_items=items,
        fractions=[0.25, 1.00]
    )

    assert results.total_runs == 4
    assert len(results.conditioned_results) == 2
    assert len(results.baseline_results) == 2
    assert os.path.exists(results_path)

    # Check structure of single run
    cond_25 = results.conditioned_results["0.25"]
    assert "pass_at_1" in cond_25
    assert "pass_at_5" in cond_25
    assert "failure_taxonomy" in cond_25
    assert cond_25["total_problems"] == 3


def test_valley_metrics_calculation(tmp_path):
    engine = BenchmarkSweepEngine(mode="fast", results_path=str(tmp_path / "out.json"))
    cond_res = {
        "0.1": {"pass_at_1": 0.16, "failure_taxonomy": {"CE": 1}},
        "0.25": {"pass_at_1": 0.24, "failure_taxonomy": {"CE": 1}},
        "0.5": {"pass_at_1": 0.32, "failure_taxonomy": {"CE": 1}},
        "1.0": {"pass_at_1": 0.42, "failure_taxonomy": {"CE": 1}}
    }
    base_res = {
        "0.1": {"pass_at_1": 0.14, "failure_taxonomy": {"CE": 10}},
        "0.25": {"pass_at_1": 0.08, "failure_taxonomy": {"CE": 20}},
        "0.5": {"pass_at_1": 0.12, "failure_taxonomy": {"CE": 15}},
        "1.0": {"pass_at_1": 0.18, "failure_taxonomy": {"CE": 12}}
    }

    metrics = engine._calculate_valley_metrics([0.1, 0.25, 0.5, 1.0], cond_res, base_res)
    assert metrics["baseline_dip_depth"] == 6.0  # 14% - 8% = 6.0% dip
    assert metrics["conditioned_is_monotonic"] is True
    assert metrics["trough_relative_gain_pct"] == 200.0  # (0.24 - 0.08)/0.08 * 100
    assert metrics["ce_reduction_pct"] > 80.0


def test_phase5_visualizations(tmp_path):
    vis = Visualizer(output_dir=str(tmp_path / "figures"))
    dummy_results = {
        "fractions": [0.10, 0.25, 0.50, 1.00],
        "conditioned_results": {
            "0.1": {"pass_at_1": 0.16, "pass_at_5": 0.50, "failure_taxonomy": {"AC": 10, "WA": 20, "TLE": 5, "RE": 5, "CE": 1}, "category_pass_rates": {"greedy": 0.2, "math": 0.3}},
            "0.25": {"pass_at_1": 0.25, "pass_at_5": 0.65, "failure_taxonomy": {"AC": 15, "WA": 15, "TLE": 5, "RE": 5, "CE": 1}, "category_pass_rates": {"greedy": 0.3, "math": 0.4}},
            "0.5": {"pass_at_1": 0.32, "pass_at_5": 0.75, "failure_taxonomy": {"AC": 20, "WA": 12, "TLE": 4, "RE": 4, "CE": 1}, "category_pass_rates": {"greedy": 0.4, "math": 0.5}},
            "1.0": {"pass_at_1": 0.44, "pass_at_5": 0.88, "failure_taxonomy": {"AC": 25, "WA": 10, "TLE": 3, "RE": 2, "CE": 1}, "category_pass_rates": {"greedy": 0.5, "math": 0.6}}
        },
        "baseline_results": {
            "0.1": {"pass_at_1": 0.13, "pass_at_5": 0.45, "failure_taxonomy": {"AC": 8, "WA": 18, "TLE": 6, "RE": 8, "CE": 10}, "category_pass_rates": {"greedy": 0.15, "math": 0.2}},
            "0.25": {"pass_at_1": 0.10, "pass_at_5": 0.40, "failure_taxonomy": {"AC": 6, "WA": 16, "TLE": 8, "RE": 12, "CE": 15}, "category_pass_rates": {"greedy": 0.1, "math": 0.15}},
            "0.5": {"pass_at_1": 0.12, "pass_at_5": 0.48, "failure_taxonomy": {"AC": 8, "WA": 17, "TLE": 7, "RE": 10, "CE": 12}, "category_pass_rates": {"greedy": 0.12, "math": 0.18}},
            "1.0": {"pass_at_1": 0.18, "pass_at_5": 0.55, "failure_taxonomy": {"AC": 12, "WA": 16, "TLE": 5, "RE": 8, "CE": 8}, "category_pass_rates": {"greedy": 0.2, "math": 0.25}}
        }
    }

    f1 = vis.plot_phase5_valley_mapping(dummy_results, filename="test_valley.png")
    f2 = vis.plot_phase5_failure_taxonomy(dummy_results, filename="test_taxonomy.png")
    f3 = vis.plot_phase5_category_breakdown(dummy_results, filename="test_cats.png")

    assert os.path.exists(f1)
    assert os.path.exists(f2)
    assert os.path.exists(f3)
