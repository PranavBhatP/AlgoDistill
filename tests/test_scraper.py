"""Tests for Codeforces problem scraper and curated dataset."""

import pytest
from src.data.codeforces_scraper import (
    CodeforcesScraper,
    Problem,
    TestCase,
    TARGET_TAG_TAXONOMY,
    TAG_NORMALIZATION
)


def test_taxonomy_completeness():
    """Verify target taxonomy has all 10 standard categories."""
    assert len(TARGET_TAG_TAXONOMY) == 10
    expected = [
        "dynamic programming", "greedy", "graphs", "math", "data structures",
        "trees", "brute force", "strings", "number theory", "binary search"
    ]
    for tag in expected:
        assert tag in TARGET_TAG_TAXONOMY


def test_tag_normalization():
    """Verify alias normalization to canonical taxonomy."""
    assert TAG_NORMALIZATION["dp"] == "dynamic programming"
    assert TAG_NORMALIZATION["dfs and similar"] == "graphs"
    assert TAG_NORMALIZATION["shortest paths"] == "graphs"
    assert TAG_NORMALIZATION["ternary search"] == "binary search"


def test_curated_dataset_loading():
    """Verify curated dataset loads complete problems across all 10 categories."""
    scraper = CodeforcesScraper()
    problems = scraper.get_curated_dataset()
    assert len(problems) >= 20

    tags_found = {p.ground_truth_tag for p in problems}
    for tag in TARGET_TAG_TAXONOMY:
        assert tag in tags_found, f"Category {tag} missing in curated dataset!"

    for p in problems:
        assert p.problem_id.startswith("CF-")
        assert len(p.statement) > 20
        assert len(p.sample_tests) >= 1
        for st in p.sample_tests:
            assert isinstance(st.input_data, str)
            assert isinstance(st.output_data, str)
