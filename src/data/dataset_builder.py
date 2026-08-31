"""Dataset builder for algorithmic reasoning distillation.

Constructs balanced datasets across algorithmic categories, generates teacher
traces, partitions into Train/Val/Test splits, and exports standard JSONL files.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import os
import json
import logging
import random
from collections import defaultdict

from .codeforces_scraper import CodeforcesScraper, Problem, TARGET_TAG_TAXONOMY
from .teacher_generator import TeacherTraceGenerator, TeacherTrace

logger = logging.getLogger(__name__)


@dataclass
class DatasetItem:
    """A complete distillation training/evaluation item."""
    problem_id: str
    title: str
    ground_truth_tag: str
    rating: int
    statement: str
    input_spec: str
    output_spec: str
    teacher_rationale: str
    teacher_solution: str
    teacher_stated_tag: str
    sample_tests: List[Dict[str, str]]
    time_limit_ms: int = 2000
    memory_limit_mb: int = 256
    is_consistency_verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetItem":
        return cls(**d)


class DatasetBuilder:
    """Builds and partitions the algorithmic dataset."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.splits_dir = os.path.join(data_dir, "splits")
        self.filtered_dir = os.path.join(data_dir, "filtered")
        self.scraper = CodeforcesScraper()
        self.teacher_gen = TeacherTraceGenerator()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary data directory structure."""
        for path in [self.data_dir, self.raw_dir, self.splits_dir, self.filtered_dir]:
            os.makedirs(path, exist_ok=True)

    def build_raw_dataset(self, use_offline_curated: bool = True) -> List[DatasetItem]:
        """Build the full dataset combining problem definitions and teacher traces."""
        taco_path = os.path.join(self.data_dir, "taco_100_problems.json")
        trace_path = os.path.join(self.data_dir, "teacher_traces_100.json")

        if os.path.exists(taco_path) and os.path.exists(trace_path):
            with open(taco_path, "r", encoding="utf-8") as f:
                problems_data = json.load(f)
            with open(trace_path, "r", encoding="utf-8") as f:
                traces_data = json.load(f)
            trace_map = {t["problem_id"]: t for t in traces_data}

            dataset: List[DatasetItem] = []
            for p in problems_data:
                t = trace_map.get(p["problem_id"], {})
                item = DatasetItem(
                    problem_id=p["problem_id"],
                    title=p["title"],
                    ground_truth_tag=p["ground_truth_tag"],
                    rating=p.get("rating", 1000),
                    statement=p["statement"],
                    input_spec=p.get("input_spec", "Standard input format."),
                    output_spec=p.get("output_spec", "Standard output format."),
                    teacher_rationale=t.get("rationale", ""),
                    teacher_solution=t.get("solution_code", p.get("raw_solution", "")),
                    teacher_stated_tag=t.get("stated_algorithmic_strategy", p["ground_truth_tag"]),
                    sample_tests=p.get("sample_tests", []),
                    time_limit_ms=p.get("time_limit_ms", 2000),
                    memory_limit_mb=p.get("memory_limit_mb", 512),
                    is_consistency_verified=False
                )
                dataset.append(item)

            raw_path = os.path.join(self.raw_dir, "raw_problems_traces.jsonl")
            self.save_jsonl(dataset, raw_path)
            logger.info("Loaded %d TACO dataset items with teacher traces to %s", len(dataset), raw_path)
            return dataset

        # Fallback to curated 30 problems
        problems = self.scraper.get_curated_dataset()
        traces = self.teacher_gen.generate_all_traces(problems)
        trace_map = {t.problem_id: t for t in traces}

        dataset: List[DatasetItem] = []
        for p in problems:
            t = trace_map.get(p.problem_id)
            if not t:
                continue
            item = DatasetItem(
                problem_id=p.problem_id,
                title=p.title,
                ground_truth_tag=p.ground_truth_tag,
                rating=p.rating,
                statement=p.statement,
                input_spec=p.input_spec,
                output_spec=p.output_spec,
                teacher_rationale=t.rationale,
                teacher_solution=t.solution_code,
                teacher_stated_tag=t.stated_algorithmic_strategy,
                sample_tests=[asdict(st) if hasattr(st, "input_data") else st for st in p.sample_tests],
                time_limit_ms=p.time_limit_ms,
                memory_limit_mb=p.memory_limit_mb,
                is_consistency_verified=False
            )
            dataset.append(item)

        raw_path = os.path.join(self.raw_dir, "raw_problems_traces.jsonl")
        self.save_jsonl(dataset, raw_path)
        logger.info("Saved %d raw dataset items to %s", len(dataset), raw_path)
        return dataset

    def create_stratified_splits(
        self,
        dataset: List[DatasetItem],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[List[DatasetItem], List[DatasetItem], List[DatasetItem]]:
        """Partition dataset into category-stratified train, validation, and test splits."""
        random.seed(seed)
        tag_groups = defaultdict(list)
        for item in dataset:
            tag_groups[item.ground_truth_tag].append(item)

        train_split, val_split, test_split = [], [], []

        for tag, items in tag_groups.items():
            shuffled = list(items)
            random.shuffle(shuffled)
            n = len(shuffled)
            if n == 1:
                train_split.append(shuffled[0])
            elif n == 2:
                train_split.append(shuffled[0])
                test_split.append(shuffled[1])
            else:
                n_train = max(1, int(n * train_ratio))
                n_val = max(1, int(n * val_ratio))
                if n_train + n_val >= n:
                    n_train = max(1, n - 2)
                    n_val = 1
                train_items = shuffled[:n_train]
                val_items = shuffled[n_train:n_train + n_val]
                test_items = shuffled[n_train + n_val:]
                if not test_items:
                    test_items = [val_items.pop()] if len(val_items) > 1 else [train_items.pop()]
                train_split.extend(train_items)
                val_split.extend(val_items)
                test_split.extend(test_items)

        # Save splits
        self.save_jsonl(train_split, os.path.join(self.splits_dir, "train.jsonl"))
        self.save_jsonl(val_split, os.path.join(self.splits_dir, "val.jsonl"))
        self.save_jsonl(test_split, os.path.join(self.splits_dir, "test.jsonl"))

        logger.info(
            "Splits created: Train=%d, Val=%d, Test=%d (Total=%d)",
            len(train_split), len(val_split), len(test_split), len(dataset)
        )
        return train_split, val_split, test_split

    @staticmethod
    def save_jsonl(items: List[DatasetItem], filepath: str) -> None:
        """Write items to JSONL format."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def load_jsonl(filepath: str) -> List[DatasetItem]:
        """Load items from JSONL format."""
        items = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(DatasetItem.from_dict(json.loads(line)))
        return items
