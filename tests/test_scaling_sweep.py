"""Unit tests for Phase 4: Multi-Scale QLoRA Training Sweep."""

import os
import tempfile
import pytest
import torch

from src.data.dataset_builder import DatasetItem
from src.models.scaling_sweep import (
    SweepConfig,
    SweepRunResult,
    ScalingSweepManager,
    CausalDistillationDataset
)
from src.utils.visualization import Visualizer


def make_dummy_item(problem_id: str, tag: str) -> DatasetItem:
    return DatasetItem(
        problem_id=problem_id,
        title=f"Sample Problem {problem_id}",
        ground_truth_tag=tag,
        rating=1200,
        statement=f"Given an array of integers, solve using {tag}.",
        input_spec="Line 1: integer n",
        output_spec="Single integer result",
        teacher_rationale=f"Step 1: Identify that {tag} is optimal.\nStep 2: Maintain state.",
        teacher_solution="def solve():\n    print(1)\nif __name__ == '__main__':\n    solve()\n",
        teacher_stated_tag=tag,
        sample_tests=[{"input": "5\n", "output": "1\n"}],
        is_consistency_verified=True
    )


def test_sweep_config():
    cfg = SweepConfig(
        fractions=[0.25, 1.0],
        epochs=2,
        mode="fast"
    )
    assert cfg.epochs == 2
    assert cfg.fractions == [0.25, 1.0]
    assert cfg.mode == "fast"


def test_fast_scaling_sweep_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "checkpoints")
        results_path = os.path.join(tmpdir, "sweep_results.json")

        cfg = SweepConfig(
            fractions=[0.25, 0.50],
            epochs=2,
            mode="fast",
            output_dir=output_dir,
            results_path=results_path
        )

        manager = ScalingSweepManager(cfg)

        tags = ["dynamic programming", "greedy", "graphs", "math"]
        items = [make_dummy_item(f"P{i}", tags[i % len(tags)]) for i in range(16)]

        results = manager.run_sweep(train_items=items, fractions=[0.25, 0.50])

        assert results["total_runs"] == 4
        assert len(results["runs_list"]) == 4
        assert "0.25" in results["conditioned_runs"]
        assert "0.25" in results["baseline_runs"]
        assert "0.5" in results["conditioned_runs"]
        assert "0.5" in results["baseline_runs"]

        # Check loss trajectory
        c_res = results["conditioned_runs"]["0.25"]
        assert len(c_res["epoch_losses"]) == 2
        assert c_res["final_loss"] > 0
        assert os.path.exists(results_path)

        # Test visualization generation
        vis = Visualizer(output_dir=tmpdir)
        fig_path = vis.plot_phase4_training_loss_sweep(
            sweep_results=results,
            filename="test_phase4_sweep.png"
        )
        assert os.path.exists(fig_path)
