"""Metrics computation for algorithmic classification and generation evaluation."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


@dataclass
class ClassificationReport:
    """Evaluation report for Stage 1 Algorithmic Classifier."""
    accuracy: float
    top_3_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_mat: List[List[int]]
    classes: List[str]
    total_samples: int

    @property
    def top_1_accuracy(self) -> float:
        return self.accuracy

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PassAtKReport:
    """Evaluation report for Stage 2 Code Generation Execution."""
    pass_at_1: float
    pass_at_5: float
    total_problems: int
    solved_problems: int
    category_pass_rates: Dict[str, float]
    detailed_verdicts: Dict[str, int]
    pass_at_10: float = 0.0
    failure_taxonomy: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    y_probs: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None
) -> ClassificationReport:
    """Compute comprehensive classification metrics for Stage 1 Tag Classifier."""
    unique_classes = classes or sorted(list(set(y_true + y_pred)))
    tag_to_idx = {c: i for i, c in enumerate(unique_classes)}

    y_true_indices = [tag_to_idx.get(t, -1) for t in y_true]
    y_pred_indices = [tag_to_idx.get(t, -1) for t in y_pred]

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=unique_classes, average=None, zero_division=0)
    macro_f1 = float(np.mean(f1))
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_classes, average="weighted", zero_division=0
    )

    # Top-3 Accuracy calculation if probabilities are available
    top_3_acc = acc
    if y_probs is not None and len(y_probs) == len(y_true):
        correct_top_3 = 0
        for idx, true_class_idx in enumerate(y_true_indices):
            if true_class_idx != -1:
                top_3_preds = np.argsort(y_probs[idx])[-3:]
                if true_class_idx in top_3_preds:
                    correct_top_3 += 1
        top_3_acc = correct_top_3 / len(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=unique_classes).tolist()

    return ClassificationReport(
        accuracy=round(acc, 4),
        top_3_accuracy=round(top_3_acc, 4),
        macro_f1=round(macro_f1, 4),
        weighted_f1=round(float(weighted_f1), 4),
        per_class_precision={c: round(float(p[i]), 4) for i, c in enumerate(unique_classes)},
        per_class_recall={c: round(float(r[i]), 4) for i, c in enumerate(unique_classes)},
        per_class_f1={c: round(float(f1[i]), 4) for i, c in enumerate(unique_classes)},
        confusion_mat=cm,
        classes=unique_classes,
        total_samples=len(y_true)
    )


def compute_pass_at_k(
    results_per_problem: Dict[str, List[bool]],
    problem_tags: Optional[Dict[str, str]] = None,
    verdicts_per_problem: Optional[Dict[str, List[str]]] = None
) -> PassAtKReport:
    """Compute pass@k (unbiased HumanEval/Codeforces pass rate) and failure taxonomy across problems.
    
    results_per_problem: map of problem_id -> list of booleans (True if generated sample solved test cases)
    problem_tags: optional map of problem_id -> category string
    verdicts_per_problem: optional map of problem_id -> list of verdicts ("AC", "WA", "TLE", "RE", "CE")
    """
    total_problems = len(results_per_problem)
    if total_problems == 0:
        return PassAtKReport(0.0, 0.0, 0, 0, {}, {}, 0.0, {})

    from math import comb
    pass_1_list = []
    pass_5_list = []
    pass_10_list = []
    solved_count = 0
    cat_solved = {}
    cat_total = {}
    failure_taxonomy: Dict[str, int] = defaultdict(int)

    for prob_id, sample_results in results_per_problem.items():
        n = len(sample_results)
        c = sum(1 for r in sample_results if r)
        if c > 0:
            solved_count += 1

        # pass@1 estimator
        pass_1 = c / n if n > 0 else 0.0
        pass_1_list.append(pass_1)

        # pass@5 unbiased estimator
        k5 = min(5, n)
        if n == 0:
            pass_k5 = 0.0
        elif n - c < k5:
            pass_k5 = 1.0
        else:
            pass_k5 = 1.0 - (comb(n - c, k5) / comb(n, k5))
        pass_5_list.append(pass_k5)

        # pass@10 unbiased estimator
        k10 = min(10, n)
        if n == 0:
            pass_k10 = 0.0
        elif n - c < k10:
            pass_k10 = 1.0
        else:
            pass_k10 = 1.0 - (comb(n - c, k10) / comb(n, k10))
        pass_10_list.append(pass_k10)

        # Category tracking
        tag = (problem_tags or {}).get(prob_id, "unknown")
        cat_total[tag] = cat_total.get(tag, 0) + 1
        if c > 0:
            cat_solved[tag] = cat_solved.get(tag, 0) + 1

        # Failure taxonomy tracking
        if verdicts_per_problem and prob_id in verdicts_per_problem:
            for v in verdicts_per_problem[prob_id]:
                v_clean = v.upper() if v else "UNKNOWN"
                failure_taxonomy[v_clean] += 1
        else:
            # Fallback from boolean results
            for r in sample_results:
                failure_taxonomy["AC" if r else "WA"] += 1

    cat_rates = {t: round(cat_solved.get(t, 0) / cat_total[t], 4) for t in cat_total}

    mean_pass_1 = round(float(np.mean(pass_1_list)), 4) if pass_1_list else 0.0
    mean_pass_5 = round(float(np.mean(pass_5_list)), 4) if pass_5_list else 0.0
    mean_pass_10 = round(float(np.mean(pass_10_list)), 4) if pass_10_list else 0.0

    return PassAtKReport(
        pass_at_1=mean_pass_1,
        pass_at_5=mean_pass_5,
        total_problems=total_problems,
        solved_problems=solved_count,
        category_pass_rates=cat_rates,
        detailed_verdicts={
            "pass@1": mean_pass_1,
            "pass@5": mean_pass_5,
            "pass@10": mean_pass_10
        },
        pass_at_10=mean_pass_10,
        failure_taxonomy=dict(failure_taxonomy)
    )
