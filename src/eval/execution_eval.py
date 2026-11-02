"""Execution Pass@k Evaluation Engine.

Executes generated code against sandboxed test cases to compute Pass@1 and Pass@k
rates across algorithmic categories, comparing Conditioned vs. Unconditioned models.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import time
from collections import defaultdict
import numpy as np

from src.data.dataset_builder import DatasetItem
from src.eval.metrics import compute_pass_at_k, PassAtKReport

if TYPE_CHECKING:
    from src.pipeline.e2e_pipeline import EndToEndDistillationPipeline, PipelineResult


@dataclass
class ComparativeExecutionReport:
    """Evaluation summary comparing Conditioned vs. Unconditioned Generator."""
    total_evaluated: int
    conditioned_pass_at_1: float
    baseline_pass_at_1: float
    relative_gain_pct: float
    conditioned_verdict_counts: Dict[str, int]
    baseline_verdict_counts: Dict[str, int]
    category_conditioned_pass_rates: Dict[str, float]
    category_baseline_pass_rates: Dict[str, float]
    average_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionEvaluator:
    """Evaluates generator models on sandboxed competitive programming problems."""

    def __init__(self, pipeline: Optional[Any] = None):
        if pipeline is None:
            from src.pipeline.e2e_pipeline import EndToEndDistillationPipeline
            self.pipeline = EndToEndDistillationPipeline()
        else:
            self.pipeline = pipeline

    def evaluate_dataset(
        self,
        items: List[DatasetItem],
        max_samples: Optional[int] = None
    ) -> ComparativeExecutionReport:
        """Run sandboxed evaluation on dataset items comparing conditioned vs baseline."""
        eval_items = items[:max_samples] if max_samples else items
        cond_results_per_problem: Dict[str, List[bool]] = {}
        base_results_per_problem: Dict[str, List[bool]] = {}
        problem_tags: Dict[str, str] = {}

        cond_verdicts: Dict[str, int] = defaultdict(int)
        base_verdicts: Dict[str, int] = defaultdict(int)
        latencies: List[float] = []

        for item in eval_items:
            res: PipelineResult = self.pipeline.run(
                problem_statement=item.statement,
                problem_id=item.problem_id,
                sample_tests=item.sample_tests,
                input_spec=item.input_spec,
                output_spec=item.output_spec,
                run_baseline=True
            )

            problem_tags[item.problem_id] = item.ground_truth_tag
            latencies.append(res.total_pipeline_time_ms)

            # Conditioned result
            is_cond_ac = (res.conditioned_result.verdict == "AC")
            cond_results_per_problem[item.problem_id] = [is_cond_ac]
            cond_verdicts[res.conditioned_result.verdict or "UNKNOWN"] += 1

            # Baseline result
            if res.baseline_result:
                is_base_ac = (res.baseline_result.verdict == "AC")
                base_results_per_problem[item.problem_id] = [is_base_ac]
                base_verdicts[res.baseline_result.verdict or "UNKNOWN"] += 1

        cond_report = compute_pass_at_k(cond_results_per_problem, problem_tags)
        base_report = compute_pass_at_k(base_results_per_problem, problem_tags)

        # Calculate relative gain
        rel_gain = 0.0
        if base_report.pass_at_1 > 0:
            rel_gain = ((cond_report.pass_at_1 - base_report.pass_at_1) / base_report.pass_at_1) * 100.0
        elif cond_report.pass_at_1 > 0:
            rel_gain = 100.0

        return ComparativeExecutionReport(
            total_evaluated=len(eval_items),
            conditioned_pass_at_1=cond_report.pass_at_1,
            baseline_pass_at_1=base_report.pass_at_1,
            relative_gain_pct=round(rel_gain, 2),
            conditioned_verdict_counts=dict(cond_verdicts),
            baseline_verdict_counts=dict(base_verdicts),
            category_conditioned_pass_rates=cond_report.category_pass_rates,
            category_baseline_pass_rates=base_report.category_pass_rates,
            average_latency_ms=round(float(np.mean(latencies)) if latencies else 0.0, 2)
        )
