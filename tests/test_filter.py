"""Tests for rationale-consistency filter."""

import pytest
from src.filter.rationale_filter import RationaleFilter, FilterVerdict
from src.data.dataset_builder import DatasetItem


def test_rationale_filter_retained():
    """Verify valid rationale with matching keywords is retained."""
    filt = RationaleFilter()
    item = DatasetItem(
        problem_id="TEST-1",
        title="Coin Change Test",
        ground_truth_tag="dynamic programming",
        rating=1200,
        statement="Find min coins",
        input_spec="n coins",
        output_spec="min coins",
        teacher_rationale="We define a dynamic programming state dp[i] representing minimum coins. The state transition is optimal substructure.",
        teacher_solution="print(1)",
        teacher_stated_tag="dynamic programming",
        sample_tests=[]
    )
    res = filt.evaluate_item(item)
    assert res.verdict == FilterVerdict.RETAINED
    assert res.is_valid is True
    assert res.consistency_score >= 0.5


def test_rationale_filter_tag_contradiction():
    """Verify explicit tag contradiction is rejected."""
    filt = RationaleFilter()
    item = DatasetItem(
        problem_id="TEST-2",
        title="Contradictory Test",
        ground_truth_tag="dynamic programming",
        rating=1200,
        statement="Find min coins",
        input_spec="n coins",
        output_spec="min coins",
        teacher_rationale="We solve this using greedy choice property.",
        teacher_solution="print(1)",
        teacher_stated_tag="greedy",
        sample_tests=[]
    )
    res = filt.evaluate_item(item)
    assert res.verdict == FilterVerdict.REJECTED_TAG_MISMATCH
    assert res.is_valid is False


def test_rationale_filter_insufficient():
    """Verify overly short rationale is rejected for insufficient reasoning."""
    filt = RationaleFilter()
    item = DatasetItem(
        problem_id="TEST-3",
        title="Short Test",
        ground_truth_tag="dynamic programming",
        rating=1200,
        statement="Find min coins",
        input_spec="n coins",
        output_spec="min coins",
        teacher_rationale="Just do DP.",
        teacher_solution="print(1)",
        teacher_stated_tag="dynamic programming",
        sample_tests=[]
    )
    res = filt.evaluate_item(item)
    assert res.verdict == FilterVerdict.REJECTED_INSUFFICIENT_REASONING
    assert res.is_valid is False
