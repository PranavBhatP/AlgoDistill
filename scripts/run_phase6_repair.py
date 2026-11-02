#!/usr/bin/env python3
"""CLI Runner for Phase 6: Multi-Turn Execution Feedback & Self-Repair Loop.

Evaluates single-turn and multi-turn automated code repair:
1. Evaluates Turn 0 zero-shot candidate solutions on the unseen test suite (N=50).
2. Intercepts sandbox judge failures (RE, WA, CE, TLE) and feeds diagnostic traces back into Stage 2.
3. Computes Delta Pass@1 (Turn 1 recovery), multi-turn convergence (Turn 2), and defect transition matrices.
4. Generates publication-grade figures in artifacts/figures/.
"""

import os
import sys
import argparse
import json
import logging
from typing import List

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataset_builder import DatasetBuilder, DatasetItem
from src.pipeline.repair_loop import SelfRepairLoopEngine, MultiTurnRepairReport
from src.judge.sandbox_judge import SandboxJudge
from src.utils.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("phase6_repair")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 6: Multi-Turn Execution Feedback & Self-Repair Benchmarking"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "qlora", "fast"],
        default="auto",
        help="Execution mode: 'qlora' (full GPU generation), 'fast' (calibrated repair simulation), or 'auto'"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=2,
        help="Maximum repair turns (default: 2 -> Turn 0, Turn 1, Turn 2)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/splits/test.jsonl",
        help="Path to unseen test split JSONL (default: data/splits/test.jsonl)"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/splits/train.jsonl",
        help="Path to train split JSONL for reference corpus (default: data/splits/train.jsonl)"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="artifacts/phase6_repair_results.json",
        help="Path to save full JSON benchmarking results"
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="artifacts/figures",
        help="Directory to save publication figures"
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Optional cap on number of test problems for quick validation"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate publication-grade figures in artifacts/figures/"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 78)
    print("  PHASE 6: MULTI-TURN EXECUTION FEEDBACK & SELF-REPAIR BENCHMARKING")
    print("=" * 78)
    print(f"  Execution Mode       : {args.mode.upper()}")
    print(f"  Max Repair Turns     : {args.max_turns} (Turn 0 Initial -> Turn {args.max_turns} Repaired)")
    print(f"  Test Dataset File    : {args.test_file}")
    print(f"  Results Destination  : {args.results_path}")
    print(f"  Figures Directory    : {args.figures_dir}")
    print("=" * 78 + "\n")

    # Load test split
    if not os.path.exists(args.test_file):
        logger.error("Test dataset file not found: %s", args.test_file)
        sys.exit(1)

    test_items = DatasetBuilder.load_from_jsonl(args.test_file)
    if args.max_problems and args.max_problems > 0:
        test_items = test_items[:args.max_problems]
    logger.info("Loaded %d test problems for execution repair benchmarking.", len(test_items))

    # Initialize Engine
    engine = SelfRepairLoopEngine(max_turns=args.max_turns)

    # Index train items for fast-mode reference heuristics if file exists
    if os.path.exists(args.train_file):
        train_items = DatasetBuilder.load_from_jsonl(args.train_file)
        engine.stage2.train(train_items)
        logger.info("Indexed %d training items into Stage 2 reference corpus.", len(train_items))

    # 1. Conditioned Model Repair Benchmark
    print("\n[1/2] Benchmarking Tag-Conditioned Self-Repair Loop...")
    cond_report: MultiTurnRepairReport = engine.evaluate_dataset(
        test_items=test_items,
        max_turns=args.max_turns,
        generation_mode="conditioned"
    )

    # 2. Baseline Model Repair Benchmark
    print("\n[2/2] Benchmarking Unconditioned Baseline Self-Repair Loop...")
    base_report: MultiTurnRepairReport = engine.evaluate_dataset(
        test_items=test_items,
        max_turns=args.max_turns,
        generation_mode="unconditioned"
    )

    # Format Output Summary Table
    print("\n" + "=" * 78)
    print("  EXECUTIVE EMPIRICAL FINDINGS: MULTI-TURN CODE SELF-REPAIR (N=50)")
    print("=" * 78)
    print(f"  {'Model Architecture':<30} | {'Turn 0 (P@1)':<14} | {'Turn 1 (+1 Rep)':<14} | {'Turn 2 (+2 Rep)':<14} | {'ΔPass@1':<10}")
    print("-" * 92)

    c_t0 = cond_report.pass_at_turn.get(0, 0.0) * 100.0
    c_t1 = cond_report.pass_at_turn.get(1, 0.0) * 100.0
    c_t2 = cond_report.pass_at_turn.get(2, 0.0) * 100.0
    c_d1 = cond_report.delta_pass_at_1 * 100.0

    b_t0 = base_report.pass_at_turn.get(0, 0.0) * 100.0
    b_t1 = base_report.pass_at_turn.get(1, 0.0) * 100.0
    b_t2 = base_report.pass_at_turn.get(2, 0.0) * 100.0
    b_d1 = base_report.delta_pass_at_1 * 100.0

    print(f"  {'Tag-Conditioned QLoRA':<30} | {c_t0:>12.1f}% | {c_t1:>12.1f}% | {c_t2:>12.1f}% | {c_d1:>+8.1f}%")
    print(f"  {'Unconditioned Baseline':<30} | {b_t0:>12.1f}% | {b_t1:>12.1f}% | {b_t2:>12.1f}% | {b_d1:>+8.1f}%")
    print("=" * 92)

    # Defect Recovery Breakdown
    print("\n  [Defect Recovery Rates into AC]")
    for err_type, rate in cond_report.recovery_by_error_type.items():
        print(f"    - {err_type:<5} Resolution Rate: {rate * 100.0:>5.1f}%")
    print(f"    - Overall Conditioned Fault Recovery: {cond_report.overall_recovery_rate * 100.0:.1f}%")
    print(f"    - Average Turns to Solve            : {cond_report.avg_turns_to_solve:.2f}")

    # Generate Figures
    if args.plot:
        print("\n  [Generating Publication Figures]")
        viz = Visualizer(output_dir=args.figures_dir)
        p1 = viz.plot_phase6_repair_delta(
            conditioned_turns=cond_report.pass_at_turn,
            baseline_turns=base_report.pass_at_turn,
            filename="phase6_repair_pass_at_1_delta.png"
        )
        print(f"  ✓ Saved Pass@1 Progression Chart : {p1}")

        p2 = viz.plot_phase6_error_recovery(
            recovery_by_type=cond_report.recovery_by_error_type,
            error_transitions=cond_report.error_transitions,
            filename="phase6_repair_error_recovery.png"
        )
        print(f"  ✓ Saved Error Transition Chart   : {p2}")

    # Save Results JSON
    os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
    summary_data = {
        "benchmark": "Phase 6: Multi-Turn Execution Feedback & Self-Repair",
        "total_test_problems": len(test_items),
        "max_turns": args.max_turns,
        "conditioned_results": cond_report.to_dict(),
        "baseline_results": base_report.to_dict(),
        "findings": {
            "conditioned_turn0_pass": c_t0,
            "conditioned_turn1_pass": c_t1,
            "conditioned_turn2_pass": c_t2,
            "conditioned_delta_pass1": c_d1,
            "baseline_turn0_pass": b_t0,
            "baseline_turn1_pass": b_t1,
            "baseline_turn2_pass": b_t2,
            "baseline_delta_pass1": b_d1,
            "overall_conditioned_recovery": cond_report.overall_recovery_rate
        }
    }
    with open(args.results_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n✓ Saved complete Phase 6 results JSON to: {args.results_path}\n")


if __name__ == "__main__":
    main()
