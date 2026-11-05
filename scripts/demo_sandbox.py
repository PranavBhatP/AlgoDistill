#!/usr/bin/env python3
"""Interactive demonstration script for the Sandbox Judge execution engine.

Demonstrates all key capabilities highlighted in the system architecture:
1. Subprocess execution with isolated multi-tier containerization.
2. POSIX RLIMIT_AS memory capping (512 MB).
3. Asynchronous CPU timeout watchdog (2.0s) with SIGKILL.
4. Pre-compilation AST syntax guard (0.0ms instant CE detection).
5. Standard competitive programming verdict classification (AC, WA, TLE, RE, CE).
"""

import os
import sys
import textwrap

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.judge.sandbox_judge import SandboxJudge, Verdict

console = Console()


def run_sandbox_demo():
    console.print(Panel.fit(
        "[bold cyan]Subprocess Execution Sandbox Judge Demo[/bold cyan]\n"
        "[dim]POSIX Resource Limiting (RLIMIT_AS), CPU Watchdog & Pre-compilation AST Guard[/dim]",
        border_style="cyan"
    ))

    judge = SandboxJudge(default_timeout_s=2.0, max_memory_mb=512)

    scenarios = [
        {
            "name": "1. Correct Solution (Expected: AC)",
            "description": "Standard I/O solution reading stdin and printing sum.",
            "code": textwrap.dedent("""
                import sys
                def solve():
                    lines = sys.stdin.read().split()
                    if not lines:
                        return
                    a, b = int(lines[0]), int(lines[1])
                    print(a + b)
                if __name__ == '__main__':
                    solve()
            """),
            "tests": [
                {"input_data": "10 20\n", "output_data": "30\n"},
                {"input_data": "5 -3\n", "output_data": "2\n"}
            ],
            "timeout": 2.0
        },
        {
            "name": "2. Logic Discrepancy (Expected: WA)",
            "description": "Output does not match expected reference output.",
            "code": textwrap.dedent("""
                import sys
                lines = sys.stdin.read().split()
                print(int(lines[0]) * int(lines[1]))  # Multiplies instead of adds
            """),
            "tests": [
                {"input_data": "10 20\n", "output_data": "30\n"}
            ],
            "timeout": 2.0
        },
        {
            "name": "3. CPU Timeout Watchdog (Expected: TLE)",
            "description": "Infinite loop killed via SIGKILL after 2.0s limit.",
            "code": textwrap.dedent("""
                import time
                while True:
                    time.sleep(0.1)
            """),
            "tests": [
                {"input_data": "1\n", "output_data": "1\n"}
            ],
            "timeout": 2.0
        },
        {
            "name": "4. Runtime Exception (Expected: RE)",
            "description": "Uncaught ZeroDivisionError with stderr capture.",
            "code": textwrap.dedent("""
                x = 10 / 0
            """),
            "tests": [
                {"input_data": "1\n", "output_data": "1\n"}
            ],
            "timeout": 2.0
        },
        {
            "name": "5. Pre-compilation AST Guard (Expected: CE)",
            "description": "Broken syntax intercepted before subprocess execution (0.0ms).",
            "code": textwrap.dedent("""
                def broken_function(:
                    return 42
            """),
            "tests": [
                {"input_data": "1\n", "output_data": "42\n"}
            ],
            "timeout": 2.0
        },
        {
            "name": "6. Memory Limit Exceeded (Expected: RE / MemoryError)",
            "description": "Attempt to allocate array exceeding POSIX RLIMIT_AS (512 MB).",
            "code": textwrap.dedent("""
                # Attempt to allocate 800 MB virtual memory
                big_array = [0] * (100 * 1024 * 1024)
                print(len(big_array))
            """),
            "tests": [
                {"input_data": "1\n", "output_data": "OK\n"}
            ],
            "timeout": 2.0
        }
    ]

    table = Table(
        title="Sandbox Execution Results Across Failure Modes",
        box=box.ROUNDED,
        header_style="bold magenta",
        show_lines=True
    )
    table.add_column("Scenario / Test", style="cyan", width=34)
    table.add_column("Verdict", justify="center", width=12)
    table.add_column("Pass Rate", justify="center", width=12)
    table.add_column("Max Runtime", justify="right", width=14)
    table.add_column("Diagnostics / Error Message", style="yellow")

    for scenario in scenarios:
        res = judge.evaluate_python_solution(
            scenario["code"],
            scenario["tests"],
            timeout_s=scenario["timeout"]
        )

        verdict_color = {
            Verdict.AC: "bold green",
            Verdict.WA: "bold red",
            Verdict.TLE: "bold yellow",
            Verdict.RE: "bold magenta",
            Verdict.CE: "bold blue"
        }.get(res.verdict, "white")

        error_msg = res.error_message or "-"
        if len(error_msg) > 65:
            error_msg = error_msg[:62] + "..."

        table.add_row(
            f"[bold]{scenario['name']}[/bold]\n[dim]{scenario['description']}[/dim]",
            f"[{verdict_color}]{res.verdict.value}[/{verdict_color}]",
            f"{res.tests_passed}/{res.total_tests}",
            f"{res.max_runtime_ms:.1f} ms",
            error_msg
        )

    console.print(table)
    console.print("\n[bold green]✓ Sandbox Judge demonstration completed successfully![/bold green]\n")


if __name__ == "__main__":
    run_sandbox_demo()

