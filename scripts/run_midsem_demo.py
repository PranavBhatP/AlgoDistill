#!/usr/bin/env python3
"""Midsemester Evaluation Demonstration Script.

Runs end-to-end Phase 1 (Data Engine, Teacher Traces, Sandbox Judge) and
Phase 2 (Rationale-Consistency Filter, Stage 1 QLoRA Tag Classifier).
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.data.dataset_builder import DatasetBuilder
from src.filter.rationale_filter import RationaleFilter
from src.models.stage1_classifier import Stage1TagClassifier, Stage1Config
from src.judge.sandbox_judge import SandboxJudge, Verdict
from src.utils.visualization import Visualizer

console = Console()


def run_demo():
    console.print(Panel.fit(
        "[bold cyan]QLoRA Distillation for Algorithmic Code Classification and Generation[/bold cyan]\n"
        "[yellow]IT488 Course Project - Midsemester Evaluation (Phases 1 & 2)[/yellow]\n"
        "[white]Author: Pranav Bhat P (231IT049)[/white]",
        border_style="cyan"
    ))

    # ----------------------------------------------------
    # Phase 1: Problem Dataset Construction & Teacher Traces
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Phase 1.1: Dataset Construction & Teacher Trace Generation[/bold green]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="Loading curated Codeforces problems & teacher rationales...", total=None)
        builder = DatasetBuilder()
        dataset = builder.build_raw_dataset()

    cat_table = Table(title="Curated Algorithmic Problem Distribution (10 Taxonomy Categories)", show_header=True, header_style="bold magenta")
    cat_table.add_column("Index", justify="right", style="dim")
    cat_table.add_column("Algorithmic Category", style="cyan")
    cat_table.add_column("Problem Count", justify="right", style="green")
    cat_table.add_column("Example Problems", style="white")

    tag_probs = {}
    for item in dataset:
        tag_probs.setdefault(item.ground_truth_tag, []).append(item)

    for i, (tag, items) in enumerate(sorted(tag_probs.items()), start=1):
        examples = ", ".join([f"{it.problem_id} ({it.title})" for it in items[:2]])
        cat_table.add_row(str(i), tag.title(), str(len(items)), examples)
    console.print(cat_table)

    # ----------------------------------------------------
    # Phase 1: Sandboxed Subprocess Test Judge
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Phase 1.2: Sandboxed Subprocess Execution Judge[/bold green]")
    judge = SandboxJudge(default_timeout_s=2.0, max_memory_mb=512)
    
    judge_table = Table(title="Sandboxed Subprocess Execution Judge Verification", show_header=True, header_style="bold magenta")
    judge_table.add_column("Problem ID", style="cyan")
    judge_table.add_column("Category", style="yellow")
    judge_table.add_column("Verdict", style="bold green")
    judge_table.add_column("Tests Passed", justify="center")
    judge_table.add_column("Max Runtime", justify="right")

    for item in dataset[:4]:
        res = judge.evaluate_python_solution(item.teacher_solution, item.sample_tests)
        verdict_str = f"[bold green]{res.verdict.value}[/bold green]" if res.verdict == Verdict.AC else f"[bold red]{res.verdict.value}[/bold red]"
        judge_table.add_row(
            item.problem_id,
            item.ground_truth_tag.title(),
            verdict_str,
            f"{res.tests_passed}/{res.total_tests}",
            f"{res.max_runtime_ms:.1f} ms"
        )
    console.print(judge_table)

    # ----------------------------------------------------
    # Phase 2: Rationale-Consistency Filter
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Phase 2.1: Rationale-Consistency Semantic Filter[/bold green]")
    console.print("[dim]Evaluating teacher reasoning traces against ground-truth tags to eliminate spurious/hallucinatory supervision...[/dim]")
    
    filt = RationaleFilter()
    retained, discarded, summary = filt.filter_dataset(dataset)

    filt_panel = (
        f"[bold white]Total Traces Evaluated:[/bold white] {summary['total_evaluated']}\n"
        f"[bold green]Retained Verified Traces:[/bold green] {summary['retained_count']} ({summary['retention_rate_pct']}%)\n"
        f"[bold red]Discarded Flawed/Mismatched Traces:[/bold red] {summary['discarded_count']}\n"
        f"[bold yellow]Verdict Breakdown:[/bold yellow] {json.dumps(summary['verdict_breakdown'], indent=2)}"
    )
    console.print(Panel(filt_panel, title="Rationale-Consistency Filter Statistics", border_style="yellow"))

    # ----------------------------------------------------
    # Phase 2: Stage 1 QLoRA Algorithmic Tag Classifier
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Phase 2.2: Stage 1 Algorithmic Tag Classifier Training & Evaluation[/bold green]")
    train_split, val_split, test_split = builder.create_stratified_splits(retained)
    console.print(f"[white]Dataset Partitioning:[/white] Train = {len(train_split)}, Val = {len(val_split)}, Test = {len(test_split)}")

    clf = Stage1TagClassifier(Stage1Config(mode="fast"))
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(description="Training Stage 1 Algorithmic Classifier...", total=None)
        train_res = clf.train(train_split, val_split)

    report = clf.evaluate(test_split)

    metrics_table = Table(title="Stage 1 Algorithmic Tag Classification Results (Test Set)", show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Score", justify="right", style="bold green")
    metrics_table.add_row("Top-1 Classification Accuracy", f"{report.accuracy * 100:.2f}%")
    metrics_table.add_row("Top-3 Accuracy", f"{report.top_3_accuracy * 100:.2f}%")
    metrics_table.add_row("Macro-Averaged F1 Score", f"{report.macro_f1:.4f}")
    metrics_table.add_row("Weighted F1 Score", f"{report.weighted_f1:.4f}")
    metrics_table.add_row("Total Test Problems", str(report.total_samples))
    console.print(metrics_table)

    # ----------------------------------------------------
    # Visualization Artifacts Generation
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Generating Visualizations & Reports[/bold green]")
    vis = Visualizer(output_dir="artifacts/figures")
    cm_plot = vis.plot_confusion_matrix(report.confusion_mat, report.classes)
    f1_plot = vis.plot_stage1_metrics(report.per_class_f1, report.macro_f1, report.accuracy)
    yield_plot = vis.plot_rationale_filtering_breakdown(summary)
    
    console.print(f"[white]Plots generated:[/white]")
    console.print(f"  [cyan]• Confusion Matrix:[/cyan] {cm_plot}")
    console.print(f"  [cyan]• Per-Class F1 Chart:[/cyan] {f1_plot}")
    console.print(f"  [cyan]• Rationale Filter Yield:[/cyan] {yield_plot}")

    # ----------------------------------------------------
    # Live Inference Demonstration
    # ----------------------------------------------------
    console.print("\n[bold green]>>> Phase 2.3: Live Problem Inference Demonstration[/bold green]")
    demo_problem = (
        "Given a weighted graph with n vertices and m edges. Find the shortest path "
        "between source vertex 1 and destination vertex n using a priority queue."
    )
    pred_res = clf.predict(demo_problem)
    
    console.print(Panel(
        f"[bold white]Input Problem Statement:[/bold white]\n\"{demo_problem}\"\n\n"
        f"[bold green]Predicted Algorithmic Paradigm:[/bold green] [bold cyan]{pred_res['predicted_tag'].upper()}[/bold cyan] "
        f"(Confidence: {pred_res['confidence'] * 100:.1f}%)\n"
        f"[bold yellow]Top-3 Predicted Classes:[/bold yellow]\n" +
        "\n".join([f"  {i+1}. {item['tag'].title()} ({item['probability']*100:.1f}%)" for i, item in enumerate(pred_res['top_k'])]),
        title="Stage 1 Live Tag Prediction",
        border_style="green"
    ))

    console.print(Panel.fit(
        "[bold green]✓ MIDSEMESTER EVALUATION (PHASES 1 & 2) SUCCESSFULLY VERIFIED![/bold green]\n"
        "[dim]Ready for presentation and demonstration tomorrow.[/dim]",
        border_style="green"
    ))


if __name__ == "__main__":
    run_demo()
