"""Scaling Partitioner for 'Valley of Code Reasoning' sweep.

Partitions stratified training datasets into fractional sweeps [10%, 25%, 50%, 100%]
preserving category balance to investigate the empirical reasoning valley curve.
"""

from typing import List, Dict, Any, Tuple
import math
import random
from collections import defaultdict

from src.data.dataset_builder import DatasetItem


class ScalingPartitioner:
    """Partitions training data into progressive scale fractions."""

    DEFAULT_FRACTIONS = [0.10, 0.25, 0.50, 1.00]

    @classmethod
    def create_fractional_subsets(
        cls,
        items: List[DatasetItem],
        fractions: List[float] = DEFAULT_FRACTIONS,
        seed: int = 42
    ) -> Dict[float, List[DatasetItem]]:
        """Create category-stratified nested or independent fractional subsets."""
        rng = random.Random(seed)
        items_by_cat: Dict[str, List[DatasetItem]] = defaultdict(list)
        for it in items:
            items_by_cat[it.ground_truth_tag].append(it)

        # Shuffle each category deterministically
        for cat in items_by_cat:
            rng.shuffle(items_by_cat[cat])

        subsets: Dict[float, List[DatasetItem]] = {}
        for frac in sorted(fractions):
            if frac >= 1.0:
                subsets[frac] = list(items)
                continue

            subset: List[DatasetItem] = []
            for cat, cat_items in items_by_cat.items():
                k = max(1, math.ceil(len(cat_items) * frac))
                subset.extend(cat_items[:k])

            subsets[frac] = subset

        return subsets
