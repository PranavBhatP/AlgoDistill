"""Subprocess-isolated sandbox judge for evaluating algorithmic solutions.

Executes code against competitive programming test cases with strict CPU time limits,
memory bounds, and formatted output verification.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional, Union
import os
import sys
import time
import tempfile
import subprocess
import resource
import logging

logger = logging.getLogger(__name__)


class Verdict(str, Enum):
    """Standard competitive programming verdicts."""
    AC = "AC"    # Accepted
    WA = "WA"    # Wrong Answer
    TLE = "TLE"  # Time Limit Exceeded
    MLE = "MLE"  # Memory Limit Exceeded
    RE = "RE"    # Runtime Error
    CE = "CE"    # Compilation / Syntax Error


@dataclass
class TestResult:
    """Result of executing a single test case."""
    test_index: int
    input_data: str
    expected_output: str
    actual_output: str
    verdict: Verdict
    runtime_ms: float
    error_message: Optional[str] = None


@dataclass
class ExecutionResult:
    """Aggregated execution result across all test cases."""
    verdict: Verdict
    tests_passed: int
    total_tests: int
    pass_rate: float
    max_runtime_ms: float
    test_results: List[TestResult] = field(default_factory=list)
    compilation_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        data["test_results"] = [
            {**asdict(tr), "verdict": tr.verdict.value} for tr in self.test_results
        ]
        return data


class SandboxJudge:
    """Sandboxed execution judge with resource limits."""

    def __init__(self, default_timeout_s: float = 2.0, max_memory_mb: int = 512):
        self.default_timeout_s = default_timeout_s
        self.max_memory_mb = max_memory_mb

    @staticmethod
    def _normalize_output(text: str) -> List[str]:
        """Normalize output text by stripping trailing whitespace and normalizing newlines."""
        if not text:
            return []
        lines = text.strip().replace("\r\n", "\n").split("\n")
        return [line.rstrip() for line in lines if line.rstrip()]

    def _set_limits(self, memory_mb: int) -> None:
        """Set subprocess resource limits on Unix/Linux."""
        try:
            # Memory ceiling (virtual memory)
            mem_bytes = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError) as e:
            logger.debug("Could not set strict resource limit: %s", e)

    def evaluate_python_solution(
        self,
        code: str,
        test_cases: List[Union[Dict[str, str], Any]],
        timeout_s: Optional[float] = None,
        memory_mb: Optional[int] = None
    ) -> ExecutionResult:
        """Evaluate Python code against multiple test cases in isolated subprocesses."""
        timeout = timeout_s or self.default_timeout_s
        mem_limit = memory_mb or self.max_memory_mb

        if not test_cases:
            return ExecutionResult(
                verdict=Verdict.AC,
                tests_passed=0,
                total_tests=0,
                pass_rate=1.0,
                max_runtime_ms=0.0
            )

        # Pre-validate python syntax
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return ExecutionResult(
                verdict=Verdict.CE,
                tests_passed=0,
                total_tests=len(test_cases),
                pass_rate=0.0,
                max_runtime_ms=0.0,
                compilation_error=f"SyntaxError: {e}"
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as script_file:
            script_file.write(code)
            script_path = script_file.name

        try:
            results: List[TestResult] = []
            max_ms = 0.0
            overall_verdict = Verdict.AC

            for idx, tc in enumerate(test_cases):
                inp = tc.get("input_data", "") if isinstance(tc, dict) else getattr(tc, "input_data", "")
                expected = tc.get("output_data", "") if isinstance(tc, dict) else getattr(tc, "output_data", "")

                start_t = time.perf_counter()
                try:
                    proc = subprocess.run(
                        [sys.executable, script_path],
                        input=inp,
                        text=True,
                        capture_output=True,
                        timeout=timeout,
                        preexec_fn=lambda: self._set_limits(mem_limit)
                    )
                    runtime_ms = (time.perf_counter() - start_t) * 1000.0
                    max_ms = max(max_ms, runtime_ms)

                    if proc.returncode != 0:
                        verdict = Verdict.RE
                        if overall_verdict == Verdict.AC:
                            overall_verdict = Verdict.RE
                        results.append(TestResult(
                            test_index=idx,
                            input_data=inp,
                            expected_output=expected,
                            actual_output=proc.stdout,
                            verdict=verdict,
                            runtime_ms=runtime_ms,
                            error_message=proc.stderr.strip()[:300]
                        ))
                        continue

                    # Compare normalized outputs
                    norm_expected = self._normalize_output(expected)
                    norm_actual = self._normalize_output(proc.stdout)

                    if norm_expected == norm_actual:
                        verdict = Verdict.AC
                    else:
                        verdict = Verdict.WA
                        if overall_verdict == Verdict.AC:
                            overall_verdict = Verdict.WA

                    results.append(TestResult(
                        test_index=idx,
                        input_data=inp,
                        expected_output=expected,
                        actual_output=proc.stdout,
                        verdict=verdict,
                        runtime_ms=runtime_ms
                    ))

                except subprocess.TimeoutExpired:
                    runtime_ms = timeout * 1000.0
                    max_ms = max(max_ms, runtime_ms)
                    if overall_verdict == Verdict.AC:
                        overall_verdict = Verdict.TLE
                    results.append(TestResult(
                        test_index=idx,
                        input_data=inp,
                        expected_output=expected,
                        actual_output="",
                        verdict=Verdict.TLE,
                        runtime_ms=runtime_ms,
                        error_message=f"Time limit exceeded (> {timeout}s)"
                    ))
                except Exception as e:
                    runtime_ms = (time.perf_counter() - start_t) * 1000.0
                    if overall_verdict == Verdict.AC:
                        overall_verdict = Verdict.RE
                    results.append(TestResult(
                        test_index=idx,
                        input_data=inp,
                        expected_output=expected,
                        actual_output="",
                        verdict=Verdict.RE,
                        runtime_ms=runtime_ms,
                        error_message=str(e)
                    ))

            passed_count = sum(1 for r in results if r.verdict == Verdict.AC)
            pass_rate = passed_count / len(test_cases) if test_cases else 0.0

            return ExecutionResult(
                verdict=overall_verdict,
                tests_passed=passed_count,
                total_tests=len(test_cases),
                pass_rate=pass_rate,
                max_runtime_ms=max_ms,
                test_results=results
            )

        finally:
            if os.path.exists(script_path):
                try:
                    os.remove(script_path)
                except OSError:
                    pass
