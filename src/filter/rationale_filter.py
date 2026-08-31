"""Rationale-Consistency Semantic Filter for Reasoning Distillation.

Validates that a teacher model's chain-of-thought rationale causally and semantically
aligns with the problem's ground-truth algorithmic paradigm, discarding hallucinated,
contradictory, or mismatched reasoning traces.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import re
import logging
from src.data.dataset_builder import DatasetItem

logger = logging.getLogger(__name__)


class FilterVerdict(str, Enum):
    """Verdict from the rationale consistency filter."""
    RETAINED = "RETAINED"
    REJECTED_TAG_MISMATCH = "REJECTED_TAG_MISMATCH"
    REJECTED_INSUFFICIENT_REASONING = "REJECTED_INSUFFICIENT_REASONING"
    REJECTED_CONTRADICTORY_LOGIC = "REJECTED_CONTRADICTORY_LOGIC"


@dataclass
class FilterResult:
    """Detailed result of filtering a single teacher trace."""
    problem_id: str
    ground_truth_tag: str
    teacher_stated_tag: str
    verdict: FilterVerdict
    is_valid: bool
    consistency_score: float
    reasons: List[str]
    rationale_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data


# Domain-specific paradigm signatures and diagnostic keywords
PARADIGM_SIGNATURES: Dict[str, List[str]] = {
    "dynamic programming": [
        "dynamic programming", "dp", "memoization", "subproblem", "state transition",
        "optimal substructure", "overlapping subproblems", "base case", "tabulation",
        "knapsack", "recurrence"
    ],
    "greedy": [
        "greedy", "locally optimal", "greedy choice", "sort and choose", "exchange argument",
        "monotonic", "interval scheduling", "greedy allocation"
    ],
    "graphs": [
        "graph", "vertex", "vertices", "edge", "edges", "dfs", "bfs", "breadth-first",
        "depth-first", "connected component", "shortest path", "traversal", "adjacency", "portal"
    ],
    "math": [
        "math", "mathematical", "formula", "parity", "even", "odd", "combinatorics",
        "equation", "algebra", "arithmetic", "closed-form"
    ],
    "data structures": [
        "data structure", "heap", "priority queue", "hash map", "hash table", "hashmap",
        "stack", "queue", "dsu", "disjoint set", "segment tree", "fenwick", "frequency array"
    ],
    "trees": [
        "tree", "root", "leaf", "ancestor", "subtree", "depth", "forest", "hierarchy",
        "manager", "parent"
    ],
    "brute force": [
        "brute force", "simulate", "simulation", "iterate", "try all", "exhaustive",
        "scan each", "direct check"
    ],
    "strings": [
        "string", "character", "characters", "substring", "prefix", "suffix", "vowel",
        "consonant", "lexicographical", "palindrome"
    ],
    "number theory": [
        "number theory", "prime", "primes", "divisor", "divisors", "gcd", "lcm",
        "modulo", "modular", "sieve", "factorization"
    ],
    "binary search": [
        "binary search", "search space", "monotonic", "bisect", "upper_bound",
        "lower_bound", "logarithmic", "bisection", "two pointers"
    ],
}


class RationaleFilter:
    """Evaluates and filters teacher reasoning traces for causal validity."""

    def __init__(self, min_rationale_length: int = 40, threshold_score: float = 0.5):
        self.min_rationale_length = min_rationale_length
        self.threshold_score = threshold_score

    def evaluate_item(self, item: DatasetItem) -> FilterResult:
        """Evaluate a dataset item's rationale against its ground truth tag."""
        reasons = []
        gt_tag = item.ground_truth_tag.lower().strip()
        stated_tag = item.teacher_stated_tag.lower().strip() if item.teacher_stated_tag else ""
        rationale = item.teacher_rationale.lower()

        # 1. Check for explicit stated strategy contradiction
        tag_contradiction = False
        if stated_tag and stated_tag != gt_tag:
            for other_tag in PARADIGM_SIGNATURES:
                if other_tag != gt_tag and other_tag in stated_tag:
                    tag_contradiction = True
                    reasons.append(f"Explicit stated tag '{stated_tag}' contradicts ground truth '{gt_tag}'.")
                    break

        if tag_contradiction:
            return FilterResult(
                problem_id=item.problem_id,
                ground_truth_tag=gt_tag,
                teacher_stated_tag=stated_tag,
                verdict=FilterVerdict.REJECTED_TAG_MISMATCH,
                is_valid=False,
                consistency_score=0.0,
                reasons=reasons,
                rationale_snippet=item.teacher_rationale[:140] + "..."
            )

        # 2. Basic length and substance check
        if len(rationale.split()) < 8 or len(rationale) < self.min_rationale_length:
            return FilterResult(
                problem_id=item.problem_id,
                ground_truth_tag=gt_tag,
                teacher_stated_tag=stated_tag,
                verdict=FilterVerdict.REJECTED_INSUFFICIENT_REASONING,
                is_valid=False,
                consistency_score=0.1,
                reasons=["Rationale too brief or lacking substantive algorithmic reasoning."],
                rationale_snippet=item.teacher_rationale[:120] + "..."
            )

        # 3. Analyze semantic keyword signatures in the rationale
        target_keywords = PARADIGM_SIGNATURES.get(gt_tag, [gt_tag])
        matches = [kw for kw in target_keywords if re.search(r'\b' + re.escape(kw) + r'\b', rationale)]

        # Check for other conflicting paradigm dominance
        conflicting_paradigms = []
        for other_tag, other_kws in PARADIGM_SIGNATURES.items():
            if other_tag == gt_tag:
                continue
            other_matches = [kw for kw in other_kws if re.search(r'\b' + re.escape(kw) + r'\b', rationale)]
            # If another paradigm is overwhelmingly dominant with zero target keywords
            if len(other_matches) >= 3 and len(matches) == 0:
                conflicting_paradigms.append(other_tag)

        # Compute consistency score
        score = 0.0
        if matches:
            score = min(1.0, 0.4 + 0.2 * len(matches))
        elif gt_tag in rationale:
            score = 0.6
        else:
            score = 0.2

        if tag_contradiction or conflicting_paradigms:
            score = max(0.0, score - 0.5)

        # Determine verdict
        if tag_contradiction:
            verdict = FilterVerdict.REJECTED_TAG_MISMATCH
            is_valid = False
        elif conflicting_paradigms:
            verdict = FilterVerdict.REJECTED_CONTRADICTORY_LOGIC
            reasons.append(f"Rationale heavily references contradictory paradigm: {', '.join(conflicting_paradigms)}")
            is_valid = False
        elif score >= self.threshold_score and (matches or gt_tag in rationale):
            verdict = FilterVerdict.RETAINED
            is_valid = True
            reasons.append(f"Strong paradigm alignment with keywords: {', '.join(matches[:4])}")
        else:
            verdict = FilterVerdict.REJECTED_INSUFFICIENT_REASONING
            is_valid = False
            reasons.append(f"Lacks verifiable domain concepts for category '{gt_tag}'.")

        return FilterResult(
            problem_id=item.problem_id,
            ground_truth_tag=gt_tag,
            teacher_stated_tag=stated_tag,
            verdict=verdict,
            is_valid=is_valid,
            consistency_score=score,
            reasons=reasons,
            rationale_snippet=item.teacher_rationale[:140] + "..."
        )

    def filter_dataset(
        self, dataset: List[DatasetItem]
    ) -> Tuple[List[DatasetItem], List[DatasetItem], Dict[str, Any]]:
        """Filter dataset, returning retained items, discarded items, and detailed statistics."""
        retained: List[DatasetItem] = []
        discarded: List[DatasetItem] = []
        results: List[FilterResult] = []
        verdict_counts: Dict[str, int] = {v.value: 0 for v in FilterVerdict}
        per_tag_stats: Dict[str, Dict[str, int]] = {}

        for item in dataset:
            res = self.evaluate_item(item)
            results.append(res)
            verdict_counts[res.verdict.value] += 1

            tag = item.ground_truth_tag
            if tag not in per_tag_stats:
                per_tag_stats[tag] = {"total": 0, "retained": 0, "rejected": 0}
            per_tag_stats[tag]["total"] += 1

            if res.is_valid:
                item.is_consistency_verified = True
                retained.append(item)
                per_tag_stats[tag]["retained"] += 1
            else:
                discarded.append(item)
                per_tag_stats[tag]["rejected"] += 1

        retention_rate = (len(retained) / len(dataset)) * 100.0 if dataset else 0.0

        summary = {
            "total_evaluated": len(dataset),
            "retained_count": len(retained),
            "discarded_count": len(discarded),
            "retention_rate_pct": round(retention_rate, 2),
            "verdict_breakdown": verdict_counts,
            "per_tag_stats": per_tag_stats,
            "filter_results": [r.to_dict() for r in results]
        }

        logger.info(
            "Consistency Filtering Complete: %d/%d retained (%.1f%%). Discarded %d invalid traces.",
            len(retained), len(dataset), retention_rate, len(discarded)
        )
        return retained, discarded, summary
