#!/usr/bin/env python3
"""50% Evaluation Milestone Demonstration Script.

Runs end-to-end:
1. Benchmark Dataset & Quality Controls (Phase 1 & 2.1 recap)
2. Stage 1 QLoRA Algorithmic Tag Classifier (Phase 2.2)
3. Stage 2 Tag-Conditioned Code Generator & Baseline Ablation Model (Phase 2.3)
4. End-to-End Two-Stage Distillation Pipeline with Subprocess Sandbox Execution Judge (Phase 2.4)
5. Execution Pass@1 Benchmark comparing Conditioned vs. Baseline (Phase 2.5)
6. Scaling Sweep Partitions (10%, 25%, 50%, 100%) for 'Valley of Code Reasoning' (Phase 2.6)
7. Live Interactive Problem Inference & Execution Verification
"""

import os
import sys
import json
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.data.dataset_builder import DatasetBuilder
from src.filter.rationale_filter import RationaleFilter
from src.judge.sandbox_judge import SandboxJudge, Verdict
from src.models.stage1_classifier import Stage1TagClassifier, Stage1Config
from src.models.stage2_generator import Stage2CodeGenerator, Stage2Config
from src.pipeline.e2e_pipeline import EndToEndDistillationPipeline, PipelineResult
from src.eval.execution_eval import ExecutionEvaluator
from src.data.scaling_partitioner import ScalingPartitioner

console = Console()


def run_demo():
    console.print(Panel.fit(
        "[bold cyan]QLoRA Distillation for Algorithmic Code Classification and Generation[/bold cyan]\n"
        "[bold yellow]IT488 Course Project - 50% Evaluation Milestone Demonstration[/bold yellow]\n"
        "[white]Author: Pranav Bhat P (Roll No: 231IT049)[/white]\n"
        "[dim]Two-Stage Architecture: Decoupled Classification + Tag-Conditioned Code Synthesis[/dim]",
        border_style="cyan"
    ))

    # ----------------------------------------------------
    # Step 1: Dataset & Quality Filter
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Step 1: Benchmark Dataset & Rationale Filter Verification[/bold green]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="Loading curated TACO problems and teacher traces...", total=None)
        builder = DatasetBuilder()
        dataset = builder.build_raw_dataset()

    filt = RationaleFilter()
    retained, discarded, filt_summary = filt.filter_dataset(dataset)

    step1_panel = (
        f"[bold white]Total Curated Problems:[/bold white] {len(dataset)} across 10 algorithmic categories\n"
        f"[bold green]Retained Verified Traces:[/bold green] {filt_summary['retained_count']} ({filt_summary['retention_rate_pct']}% yield)\n"
        f"[bold red]Discarded Noisy / Contradictory Traces:[/bold red] {filt_summary['discarded_count']}\n"
        f"[bold cyan]Subprocess Sandbox Judge Limits:[/bold cyan] 2.0s CPU timeout, 512MB POSIX memory"
    )
    console.print(Panel(step1_panel, title="Phase 1 & 2.1 Quality Assurance Recap", border_style="blue"))

    train_split, val_split, test_split = builder.create_stratified_splits(retained)
    console.print(f"[white]Dataset Partitioning:[/white] Train = {len(train_split)}, Val = {len(val_split)}, Test = {len(test_split)}")

    # ----------------------------------------------------
    # Step 2: Stage 1 Classifier Training & Evaluation
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Step 2: Stage 1 Algorithmic Tag Classifier (Problem -> [TAG])[/bold green]")
    clf = Stage1TagClassifier(Stage1Config(mode="fast"))
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="Training Stage 1 Classifier...", total=None)
        clf.train(train_split, val_split)

    clf_report = clf.evaluate(test_split)
    clf_table = Table(title="Stage 1 Classifier Metrics (Test Set)", show_header=True, header_style="bold magenta")
    clf_table.add_column("Metric", style="cyan")
    clf_table.add_column("Value", justify="right", style="bold green")
    clf_table.add_row("Top-1 Classification Accuracy", f"{clf_report.accuracy * 100:.2f}%")
    clf_table.add_row("Top-3 Accuracy", f"{clf_report.top_3_accuracy * 100:.2f}%")
    clf_table.add_row("Macro F1-Score", f"{clf_report.macro_f1:.4f}")
    clf_table.add_row("Weighted F1-Score", f"{clf_report.weighted_f1:.4f}")
    console.print(clf_table)

    # ----------------------------------------------------
    # Step 3: Stage 2 Tag-Conditioned Code Generator
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Step 3: Stage 2 Tag-Conditioned Generator & Baseline Ablation Model[/bold green]")
    gen = Stage2CodeGenerator(Stage2Config(mode="fast"))
    gen.train(train_split, val_split)

    console.print("[dim]Stage 2 generator initialized with dual capability:[/dim]")
    console.print("  [cyan]1. Conditioned Mode:[/cyan]   [ALGORITHM CATEGORY] + Problem Statement -> Rationale + Code")
    console.print("  [yellow]2. Baseline Mode:[/yellow]      Problem Statement (No Category Tag) -> Code (Ablation Control)")

    # ----------------------------------------------------
    # Step 4: End-to-End Chained Pipeline & Sandbox Judge
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Step 4: End-to-End Chained Two-Stage Distillation Pipeline[/bold green]")
    judge = SandboxJudge(default_timeout_s=2.0, max_memory_mb=512)
    pipeline = EndToEndDistillationPipeline(clf, gen, judge)

    sample_test_item = test_split[0]
    res: PipelineResult = pipeline.run(
        problem_statement=sample_test_item.statement,
        problem_id=sample_test_item.problem_id,
        sample_tests=sample_test_item.sample_tests,
        input_spec=sample_test_item.input_spec,
        output_spec=sample_test_item.output_spec,
        run_baseline=True
    )

    e2e_table = Table(title=f"Chained Pipeline Evaluation on Test Problem: {sample_test_item.problem_id}", show_header=True, header_style="bold magenta")
    e2e_table.add_column("Pipeline Stage", style="cyan")
    e2e_table.add_column("Component Output", style="white")
    e2e_table.add_column("Execution Verdict", justify="center", style="bold green")

    e2e_table.add_row(
        "Stage 1 Classifier",
        f"Predicted Tag: [bold cyan]{res.predicted_tag.title()}[/bold cyan] (Confidence: {res.tag_confidence*100:.1f}%)",
        "[bold green]OK[/bold green]"
    )
    e2e_table.add_row(
        "Stage 2 Conditioned",
        f"Synthesized Python 3 Code ({len(res.conditioned_result.code.splitlines())} lines)",
        f"[bold green]{res.conditioned_result.verdict}[/bold green]" if res.conditioned_result.verdict == "AC" else f"[bold red]{res.conditioned_result.verdict}[/bold red]"
    )
    if res.baseline_result:
        e2e_table.add_row(
            "Baseline Unconditioned",
            f"Synthesized Python 3 Code ({len(res.baseline_result.code.splitlines())} lines)",
            f"[bold green]{res.baseline_result.verdict}[/bold green]" if res.baseline_result.verdict == "AC" else f"[bold red]{res.baseline_result.verdict}[/bold red]"
        )
    console.print(e2e_table)

    # ----------------------------------------------------
    # Step 5: Pass@1 Benchmark Evaluation
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Step 5: Automated Execution Pass@1 Benchmark (Conditioned vs. Baseline)[/bold green]")
    evaluator = ExecutionEvaluator(pipeline)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="Evaluating test problems through Sandboxed Judge...", total=None)
        comp_report = evaluator.evaluate_dataset(test_split)

    pass_table = Table(title="Execution Pass@1 Benchmark Results", show_header=True, header_style="bold magenta")
    pass_table.add_column("Model Configuration", style="cyan")
    pass_table.add_column("Pass@1 Rate", justify="right", style="bold green")
    pass_table.add_column("Verdict Breakdown", style="yellow")
    pass_table.add_column("Relative Gain", justify="right", style="bold white")

    cond_breakdown = ", ".join([f"{k}: {v}" for k, v in comp_report.conditioned_verdict_counts.items()])
    base_breakdown = ", ".join([f"{k}: {v}" for k, v in comp_report.baseline_verdict_counts.items()])

    pass_table.add_row(
        "Stage 2 Tag-Conditioned (Ours)",
        f"{comp_report.conditioned_pass_at_1 * 100:.1f}%",
        cond_breakdown,
        f"+{comp_report.relative_gain_pct:.1f}%"
    )
    pass_table.add_row(
        "Unconditioned Baseline (Control)",
        f"{comp_report.baseline_pass_at_1 * 100:.1f}%",
        base_breakdown,
        "Baseline"
    )
    console.print(pass_table)

    # ----------------------------------------------------
    # Step 6: 'Valley of Code Reasoning' Scaling Sweep Partitions
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Step 6: 'Valley of Code Reasoning' Scaling Sweep Setup[/bold green]")
    scaling_subsets = ScalingPartitioner.create_fractional_subsets(train_split, fractions=[0.10, 0.25, 0.50, 1.00])
    
    scale_table = Table(title="Empirical Scaling Sweep Data Fractions", show_header=True, header_style="bold magenta")
    scale_table.add_column("Scale Fraction", style="cyan")
    scale_table.add_column("Sample Count", justify="right", style="green")
    scale_table.add_column("Category Balance", style="white")
    scale_table.add_column("Target Analysis", style="yellow")

    for frac, subset in sorted(scaling_subsets.items()):
        cats = len(set(it.ground_truth_tag for it in subset))
        desc = "Initial few-shot reasoning" if frac == 0.10 else ("Valley transition dip hypothesis" if frac in (0.25, 0.50) else "Full training recovery")
        scale_table.add_row(f"{int(frac*100)}%", str(len(subset)), f"{cats}/10 categories covered", desc)
    console.print(scale_table)

    # ----------------------------------------------------
    # Step 7: Live Custom Problem Inference & Sandbox Judge
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Step 7: Live Custom Problem Synthesis & Sandboxed Execution[/bold green]")
    demo_problem = (
        "Given an array of integers, find the contiguous subarray with the largest sum "
        "and return its sum using Kadane's algorithm principles."
    )
    demo_tests = [
        {"input": "-2 1 -3 4 -1 2 1 -5 4\n", "output": "6\n"},
        {"input": "1\n", "output": "1\n"},
        {"input": "5 4 -1 7 8\n", "output": "23\n"}
    ]

    live_res = pipeline.run(
        problem_statement=demo_problem,
        problem_id="live_kadane_demo",
        sample_tests=demo_tests,
        input_spec="Line 1: space-separated integers",
        output_spec="Single integer denoting maximum subarray sum",
        run_baseline=False
    )

    console.print(Panel(
        f"[bold white]Problem:[/bold white] \"{demo_problem}\"\n\n"
        f"[bold cyan]Predicted Paradigm (Stage 1):[/bold cyan] [bold green]{live_res.predicted_tag.upper()}[/bold green] "
        f"(Confidence: {live_res.tag_confidence*100:.1f}%)\n"
        f"[bold yellow]Synthesized Rationale (Stage 2):[/bold yellow]\n{live_res.conditioned_result.rationale[:250]}...\n\n"
        f"[bold cyan]Sandboxed Judge Execution Verdict:[/bold cyan] [bold green]{live_res.conditioned_result.verdict}[/bold green] "
        f"({live_res.conditioned_result.tests_passed}/{live_res.conditioned_result.total_tests} tests passed, Max Runtime: {live_res.conditioned_result.max_runtime_ms:.1f} ms)",
        title="Live Interactive Two-Stage Synthesis & Execution",
        border_style="green"
    ))

    console.print(Panel.fit(
        "[bold green]✓ 50% EVALUATION MILESTONE SUCCESSFULLY IMPLEMENTED AND VERIFIED![/bold green]\n"
        "[white]Stage 1 Classifier + Stage 2 Conditioned Generator + Chained Pipeline + Sandbox Judge + Pass@1 Suite[/white]",
        border_style="green"
    ))


if __name__ == "__main__":
    run_demo()
