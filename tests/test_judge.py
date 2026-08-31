"""Tests for sandboxed subprocess execution judge."""

import pytest
import textwrap
from src.judge.sandbox_judge import SandboxJudge, Verdict


def test_sandbox_judge_ac():
    """Verify sandboxed judge awards AC for correct Python solution."""
    judge = SandboxJudge(default_timeout_s=2.0)
    code = textwrap.dedent("""
        import sys
        def solve():
            lines = sys.stdin.read().split()
            if not lines:
                return
            a, b = int(lines[0]), int(lines[1])
            print(a + b)
        if __name__ == '__main__':
            solve()
    """)
    test_cases = [
        {"input_data": "2 3\n", "output_data": "5\n"},
        {"input_data": "10 20\n", "output_data": "30\n"},
    ]
    res = judge.evaluate_python_solution(code, test_cases)
    assert res.verdict == Verdict.AC
    assert res.tests_passed == 2
    assert res.total_tests == 2
    assert res.pass_rate == 1.0


def test_sandbox_judge_wa():
    """Verify sandboxed judge detects Wrong Answer."""
    judge = SandboxJudge(default_timeout_s=2.0)
    code = "print(42)"
    test_cases = [{"input_data": "1\n", "output_data": "100\n"}]
    res = judge.evaluate_python_solution(code, test_cases)
    assert res.verdict == Verdict.WA
    assert res.tests_passed == 0


def test_sandbox_judge_tle():
    """Verify sandboxed judge enforces timeout and flags TLE."""
    judge = SandboxJudge(default_timeout_s=0.5)
    code = textwrap.dedent("""
        import time
        time.sleep(1.5)
        print('Done')
    """)
    test_cases = [{"input_data": "1\n", "output_data": "Done\n"}]
    res = judge.evaluate_python_solution(code, test_cases)
    assert res.verdict == Verdict.TLE
    assert res.tests_passed == 0


def test_sandbox_judge_re():
    """Verify sandboxed judge handles runtime errors (division by zero, exception)."""
    judge = SandboxJudge(default_timeout_s=2.0)
    code = "x = 1 / 0"
    test_cases = [{"input_data": "1\n", "output_data": "1\n"}]
    res = judge.evaluate_python_solution(code, test_cases)
    assert res.verdict == Verdict.RE


def test_sandbox_judge_syntax_error():
    """Verify compilation / syntax error verdict."""
    judge = SandboxJudge(default_timeout_s=2.0)
    code = "def solve( incomplete syntax"
    test_cases = [{"input_data": "1\n", "output_data": "1\n"}]
    res = judge.evaluate_python_solution(code, test_cases)
    assert res.verdict == Verdict.CE
