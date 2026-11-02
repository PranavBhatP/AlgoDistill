"""Unit and integration tests for Phase 6: Multi-Turn Execution Feedback & Self-Repair Loop."""

import os
import pytest
from src.data.dataset_builder import DatasetItem

from src.judge.sandbox_judge import SandboxJudge, Verdict, ExecutionResult
from src.pipeline.repair_loop import (
    SelfRepairLoopEngine,
    RepairTurn,
    RepairSessionResult,
    MultiTurnRepairReport
)
from src.utils.formatting import PromptFormatter
from src.utils.visualization import Visualizer


def make_sample_problem(problem_id: str, tag: str = "math") -> DatasetItem:
    return DatasetItem(
        problem_id=problem_id,
        title=f"Repair Test Problem {problem_id}",
        ground_truth_tag=tag,
        rating=1100,
        statement="Given two integers A and B, compute their sum.",
        input_spec="Line 1: A B",
        output_spec="Single integer sum",
        teacher_rationale="Read integers and output A + B.",
        teacher_solution="import sys\n\ndef solve():\n    data = sys.stdin.read().split()\n    if not data: return\n    a, b = int(data[0]), int(data[1])\n    print(a + b)\n\nif __name__ == '__main__':\n    solve()\n",
        teacher_stated_tag=tag,
        sample_tests=[
            {"input": "3 5\n", "output": "8\n"},
            {"input": "10 20\n", "output": "30\n"}
        ]
    )


def test_repair_prompt_formatting():
    prompt = PromptFormatter.format_repair_prompt(
        problem_statement="Find maximum in array.",
        previous_code="def solve():\n    return max(arr)",
        error_feedback="NameError: name 'arr' is not defined",
        predicted_tag="greedy",
        failed_input="5\n1 2 3 4 5",
        expected_output="5",
        actual_output="",
        turn_index=1
    )

    assert "<|im_start|>system" in prompt
    assert "[ALGORITHM CATEGORY]: greedy" in prompt
    assert "NameError: name 'arr' is not defined" in prompt
    assert "Turn 0" in prompt
    assert "Failed Test Input:" in prompt
    assert "Expected Output:" in prompt
    assert "<|im_start|>assistant" in prompt


def test_feedback_extraction_from_judge():
    judge = SandboxJudge(default_timeout_s=2.0)
    engine = SelfRepairLoopEngine(judge=judge)

    # 1. Syntax Error (CE)
    syntax_error_code = "def solve()\n    print(1)"
    res_ce = judge.evaluate_python_solution(syntax_error_code, [{"input": "1", "output": "1"}])
    assert res_ce.verdict == Verdict.CE
    fb_ce, _, _, _ = engine.extract_feedback_from_judge(res_ce)
    assert "Syntax / Compilation Error" in fb_ce

    # 2. Runtime Exception (RE)
    runtime_error_code = "import sys\ndata = sys.stdin.read().split()\nprint(data[999])"
    res_re = judge.evaluate_python_solution(runtime_error_code, [{"input": "1 2", "output": "1"}])
    assert res_re.verdict == Verdict.RE
    fb_re, f_in, exp, act = engine.extract_feedback_from_judge(res_re)
    assert "Runtime Exception" in fb_re
    assert "IndexError" in fb_re
    assert f_in == "1 2"

    # 3. Wrong Answer (WA)
    wa_code = "import sys\nprint(999)"
    res_wa = judge.evaluate_python_solution(wa_code, [{"input": "1", "output": "42"}])
    assert res_wa.verdict == Verdict.WA
    fb_wa, _, exp_wa, act_wa = engine.extract_feedback_from_judge(res_wa)
    assert "Wrong Answer" in fb_wa
    assert exp_wa == "42"
    assert act_wa.strip() == "999"


def test_single_turn_self_repair_recovery():
    engine = SelfRepairLoopEngine()
    item = make_sample_problem("TEST_REPAIR_1")

    # Broken initial code that crashes with IndexError on empty or single-token input
    buggy_code = (
        "import sys\n\n"
        "def solve():\n"
        "    data = sys.stdin.read().split()\n"
        "    idx = 0\n"
        "    # Missing bounds check\n"
        "    a = int(data[idx])\n"
        "    b = int(data[idx+1])\n"
        "    print(a + b)\n\n"
        "if __name__ == '__main__':\n"
        "    solve()\n"
    )

    session: RepairSessionResult = engine.run_repair_session(
        problem_statement=item.statement,
        sample_tests=item.sample_tests,
        problem_id=item.problem_id,
        predicted_tag=item.ground_truth_tag,
        initial_code=buggy_code,
        ground_truth_solution=item.teacher_solution,
        max_turns=2
    )

    assert len(session.turns) >= 1
    # Turn 0 was evaluated
    assert session.turns[0].turn_index == 0
    # Final verdict should be AC
    assert session.is_resolved is True
    assert session.final_verdict == Verdict.AC.value
    assert session.total_time_ms > 0


def test_early_stopping_on_ac():
    engine = SelfRepairLoopEngine()
    item = make_sample_problem("TEST_EARLY_AC")

    # Already correct code
    session = engine.run_repair_session(
        problem_statement=item.statement,
        sample_tests=item.sample_tests,
        problem_id=item.problem_id,
        predicted_tag=item.ground_truth_tag,
        initial_code=item.teacher_solution,
        max_turns=3
    )

    # Should terminate immediately at Turn 0
    assert len(session.turns) == 1
    assert session.resolved_at_turn == 0
    assert session.is_resolved is True
    assert session.initial_verdict == Verdict.AC.value
    assert session.final_verdict == Verdict.AC.value


def test_multiturn_report_dataset_evaluation(tmp_path):
    engine = SelfRepairLoopEngine(max_turns=2)
    items = [
        make_sample_problem("REP_1", "math"),
        make_sample_problem("REP_2", "greedy"),
        make_sample_problem("REP_3", "dynamic programming")
    ]
    # Train fast corpus
    engine.stage2.train(items)

    report: MultiTurnRepairReport = engine.evaluate_dataset(
        test_items=items,
        max_turns=2,
        generation_mode="conditioned"
    )

    assert report.total_problems == 3
    assert 0 in report.pass_at_turn
    assert 1 in report.pass_at_turn
    assert 2 in report.pass_at_turn
    assert report.delta_pass_at_1 >= 0.0
    assert report.total_delta_pass >= 0.0
    assert len(report.session_results) == 3

    # Test visualization functions with report data
    viz = Visualizer(output_dir=str(tmp_path))
    p1_path = viz.plot_phase6_repair_delta(
        conditioned_turns=report.pass_at_turn,
        baseline_turns={0: 0.20, 1: 0.26, 2: 0.28}
    )
    assert os.path.exists(p1_path)

    p2_path = viz.plot_phase6_error_recovery(
        recovery_by_type=report.recovery_by_error_type,
        error_transitions=report.error_transitions
    )
    assert os.path.exists(p2_path)
