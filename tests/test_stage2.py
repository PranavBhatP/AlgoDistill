"""Unit tests for Stage 2 Generator and Scaling Partitioner."""

import pytest
from src.models.stage2_generator import Stage2CodeGenerator, Stage2Config, GenerationOutput
from src.data.dataset_builder import DatasetItem
from src.data.scaling_partitioner import ScalingPartitioner


def make_sample_item(problem_id: str, tag: str) -> DatasetItem:
    return DatasetItem(
        problem_id=problem_id,
        title=f"Sample {problem_id}",
        ground_truth_tag=tag,
        rating=1000,
        statement=f"Given an array, find the maximum subarray sum using {tag} principles.",
        input_spec="Line 1: N integers",
        output_spec="Single integer",
        teacher_rationale=f"We use {tag} to maintain running optimal states.",
        teacher_solution="import sys\ndef solve():\n    print(42)\nif __name__ == '__main__':\n    solve()\n",
        teacher_stated_tag=tag,
        sample_tests=[{"input": "1 2 3\n", "output": "42\n"}]
    )


def test_stage2_generator_fast_mode():
    config = Stage2Config(mode="fast")
    gen = Stage2CodeGenerator(config)

    items = [
        make_sample_item("P1", "dynamic programming"),
        make_sample_item("P2", "greedy"),
        make_sample_item("P3", "graphs")
    ]
    gen.train(items)

    # Conditioned generation
    cond_out: GenerationOutput = gen.generate(
        problem_statement="Find maximum subarray sum using dynamic programming.",
        predicted_tag="dynamic programming"
    )
    assert cond_out.code is not None
    assert len(cond_out.code) > 0
    assert cond_out.conditioned_tag == "dynamic programming"

    # Baseline unconditioned generation
    base_out: GenerationOutput = gen.generate(
        problem_statement="Find maximum subarray sum using greedy approach.",
        predicted_tag=None
    )
    assert base_out.code is not None
    assert base_out.conditioned_tag is None


def test_scaling_partitioner():
    items = []
    tags = ["dynamic programming", "greedy", "graphs", "math", "trees"]
    for i in range(50):
        items.append(make_sample_item(f"P{i}", tags[i % len(tags)]))

    subsets = ScalingPartitioner.create_fractional_subsets(items, fractions=[0.10, 0.25, 0.50, 1.00])

    assert 0.10 in subsets
    assert 0.25 in subsets
    assert 0.50 in subsets
    assert 1.00 in subsets

    assert len(subsets[0.10]) <= len(subsets[0.25])
    assert len(subsets[0.25]) <= len(subsets[0.50])
    assert len(subsets[0.50]) <= len(subsets[1.00])
    assert len(subsets[1.00]) == 50
