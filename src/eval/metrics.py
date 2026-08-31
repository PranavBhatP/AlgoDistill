"""Metrics computation for algorithmic classification and generation evaluation."""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
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
    problem_tags: Optional[Dict[str, str]] = None
) -> PassAtKReport:
    """Compute pass@k (unbiased HumanEval/Codeforces pass rate) across problems.
    
    results_per_problem: map of problem_id -> list of booleans (True if generated sample solved test cases)
    """
    total_problems = len(results_per_problem)
    if total_problems == 0:
        return PassAtKReport(0.0, 0.0, 0, 0, {}, {})

    pass_1_list = []
    pass_5_list = []
    solved_count = 0
    cat_solved = {}
    cat_total = {}

    for prob_id, sample_results in results_per_problem.items():
        n = len(sample_results)
        c = sum(1 for r in sample_results if r)
        if c > 0:
            solved_count += 1

        # pass@1 estimator
        pass_1 = c / n if n > 0 else 0.0
        pass_1_list.append(pass_1)

        # pass@5 unbiased estimator (or pass@n if n < 5)
        k = min(5, n)
        if n - c < k:
            pass_k = 1.0
        else:
            # 1 - comb(n - c, k) / comb(n, k)
            from math import comb
            pass_k = 1.0 - (comb(n - c, k) / comb(n, k))
        pass_5_list.append(pass_k)

        # Category tracking
        tag = (problem_tags or {}).get(prob_id, "unknown")
        cat_total[tag] = cat_total.get(tag, 0) + 1
        if c > 0:
            cat_solved[tag] = cat_solved.get(tag, 0) + 1

    cat_rates = {t: round(cat_solved.get(t, 0) / cat_total[t], 4) for t in cat_total}

    return PassAtKReport(
        pass_at_1=round(float(np.mean(pass_1_list)), 4),
        pass_at_5=round(float(np.mean(pass_5_list)), 4),
        total_problems=total_problems,
        solved_problems=solved_count,
        category_pass_rates=cat_rates,
        detailed_verdicts={"pass@1": round(float(np.mean(pass_1_list)), 4)}
    )
