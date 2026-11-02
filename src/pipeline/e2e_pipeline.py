"""End-to-End Two-Stage Distillation Pipeline.

Connects:
1. Stage 1: Algorithmic Tag Classifier (Problem -> [TAG])
2. Stage 2: Conditioned Code & Rationale Generator ([TAG] + Problem -> Code)
3. Sandbox Execution Judge: Sandboxed subprocess testing (Code -> Verdict)

Also provides side-by-side conditioned vs. unconditioned baseline comparison.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import os
import time
import logging

from src.models.stage1_classifier import Stage1TagClassifier, Stage1Config
from src.models.stage2_generator import Stage2CodeGenerator, Stage2Config, GenerationOutput
from src.judge.sandbox_judge import SandboxJudge, JudgeResult, Verdict

logger = logging.getLogger(__name__)


@dataclass
class PipelinePrediction:
    """Detailed prediction and execution result for a single model generation."""
    rationale: str
    code: str
    verdict: Optional[str] = None
    tests_passed: int = 0
    total_tests: int = 0
    max_runtime_ms: float = 0.0
    error_message: Optional[str] = None
    generation_mode: str = "conditioned"


@dataclass
class PipelineResult:
    """Complete end-to-end inference and evaluation result."""
    problem_id: str
    problem_statement: str
    predicted_tag: str
    tag_confidence: float
    top_3_tags: List[Dict[str, Any]]
    conditioned_result: PipelinePrediction
    baseline_result: Optional[PipelinePrediction] = None
    is_ac: bool = False
    total_pipeline_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EndToEndDistillationPipeline:
    """Two-Stage Chained Pipeline for Algorithmic Reasoning & Execution."""

    def __init__(
        self,
        stage1_classifier: Optional[Stage1TagClassifier] = None,
        stage2_generator: Optional[Stage2CodeGenerator] = None,
        judge: Optional[SandboxJudge] = None
    ):
        self.stage1 = stage1_classifier or Stage1TagClassifier(Stage1Config(mode="fast"))
        self.stage2 = stage2_generator or Stage2CodeGenerator(Stage2Config(mode="fast"))
        self.judge = judge or SandboxJudge(default_timeout_s=2.0, max_memory_mb=512)

    def run(
        self,
        problem_statement: str,
        problem_id: str = "demo_problem",
        sample_tests: Optional[List[Dict[str, str]]] = None,
        input_spec: str = "",
        output_spec: str = "",
        run_baseline: bool = True
    ) -> PipelineResult:
        """Execute end-to-end chained inference and sandboxed evaluation."""
        start_t = time.perf_counter()

        # Step 1: Stage 1 Algorithmic Classification
        stage1_pred = self.stage1.predict(problem_statement)
        predicted_tag = stage1_pred["predicted_tag"]
        confidence = stage1_pred["confidence"]
        top_k = stage1_pred["top_k"]

        # Step 2: Stage 2 Tag-Conditioned Generation
        cond_gen: GenerationOutput = self.stage2.generate(
            problem_statement=problem_statement,
            predicted_tag=predicted_tag,
            input_spec=input_spec,
            output_spec=output_spec
        )

        # Step 3: Sandboxed Execution of Conditioned Code
        cond_verdict = None
        tests_passed = 0
        total_tests = 0
        max_runtime_ms = 0.0
        error_msg = None

        if sample_tests:
            judge_res: JudgeResult = self.judge.evaluate_python_solution(cond_gen.code, sample_tests)
            cond_verdict = judge_res.verdict.value
            tests_passed = judge_res.tests_passed
            total_tests = judge_res.total_tests
            max_runtime_ms = judge_res.max_runtime_ms
            error_msg = judge_res.error_message

        cond_prediction = PipelinePrediction(
            rationale=cond_gen.rationale,
            code=cond_gen.code,
            verdict=cond_verdict,
            tests_passed=tests_passed,
            total_tests=total_tests,
            max_runtime_ms=max_runtime_ms,
            error_message=error_msg,
            generation_mode="conditioned"
        )

        # Step 4: Optional Unconditioned Baseline Generation (Ablation)
        base_prediction = None
        if run_baseline:
            base_gen: GenerationOutput = self.stage2.generate(
                problem_statement=problem_statement,
                predicted_tag=None,  # No tag conditioning
                input_spec=input_spec,
                output_spec=output_spec
            )
            base_verdict = None
            base_tests_passed = 0
            base_total_tests = 0
            base_max_runtime_ms = 0.0
            base_error_msg = None

            if sample_tests:
                base_judge_res: JudgeResult = self.judge.evaluate_python_solution(base_gen.code, sample_tests)
                base_verdict = base_judge_res.verdict.value
                base_tests_passed = base_judge_res.tests_passed
                base_total_tests = base_judge_res.total_tests
                base_max_runtime_ms = base_judge_res.max_runtime_ms
                base_error_msg = base_judge_res.error_message

            base_prediction = PipelinePrediction(
                rationale=base_gen.rationale,
                code=base_gen.code,
                verdict=base_verdict,
                tests_passed=base_tests_passed,
                total_tests=base_total_tests,
                max_runtime_ms=base_max_runtime_ms,
                error_message=base_error_msg,
                generation_mode="unconditioned"
            )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return PipelineResult(
            problem_id=problem_id,
            problem_statement=problem_statement,
            predicted_tag=predicted_tag,
            tag_confidence=confidence,
            top_3_tags=top_k,
            conditioned_result=cond_prediction,
            baseline_result=base_prediction,
            is_ac=(cond_verdict == Verdict.AC.value),
            total_pipeline_time_ms=elapsed_ms
        )
