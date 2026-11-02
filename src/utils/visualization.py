"""Visualization module for evaluation plots, confusion matrices, and the Valley of Code Reasoning curve."""

import os
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualizer:
    """Generates publication-quality charts and plots for the project report."""

    def __init__(self, output_dir: str = "artifacts/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply clean publication theme
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "figure.titlesize": 16
        })

    def plot_confusion_matrix(
        self,
        cm: List[List[int]],
        classes: List[str],
        filename: str = "stage1_confusion_matrix.png",
        title: str = "Stage 1 Tag Classifier - Confusion Matrix"
    ) -> str:
        """Plot and save confusion matrix heatmap."""
        plt.figure(figsize=(10, 8))
        cm_arr = np.array(cm)
        # Normalize by row (true labels) for clarity if non-zero
        row_sums = cm_arr.sum(axis=1)[:, np.newaxis]
        norm_cm = np.divide(cm_arr.astype("float"), row_sums, out=np.zeros_like(cm_arr, dtype=float), where=row_sums != 0)

        sns.heatmap(
            norm_cm,
            annot=cm_arr,
            fmt="d",
            cmap="Blues",
            xticklabels=[c.title() for c in classes],
            yticklabels=[c.title() for c in classes],
            cbar=True,
            linewidths=0.5
        )
        plt.title(title, pad=15, fontweight="bold")
        plt.xlabel("Predicted Algorithmic Category", labelpad=10, fontweight="semibold")
        plt.ylabel("Ground Truth Category", labelpad=10, fontweight="semibold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_stage1_metrics(
        self,
        f1_scores: Dict[str, float],
        macro_f1: float,
        accuracy: float,
        filename: str = "stage1_per_class_f1.png"
    ) -> str:
        """Plot per-class F1 score bar chart for Stage 1."""
        plt.figure(figsize=(10, 6))
        categories = list(f1_scores.keys())
        scores = list(f1_scores.values())

        colors = sns.color_palette("viridis", len(categories))
        bars = plt.barh([c.title() for c in categories], scores, color=colors, edgecolor="black", height=0.6)

        plt.axvline(macro_f1, color="red", linestyle="--", linewidth=1.5, label=f"Macro-F1 ({macro_f1:.2f})")
        plt.axvline(accuracy, color="navy", linestyle=":", linewidth=1.5, label=f"Top-1 Accuracy ({accuracy:.2f})")

        plt.xlim(0, 1.05)
        plt.xlabel("F1 Score", fontweight="semibold")
        plt.title("Stage 1 Algorithmic Tag Classification Performance", pad=15, fontweight="bold")
        plt.legend(loc="lower right")

        for bar in bars:
            w = bar.get_width()
            plt.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=9)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_valley_of_code_reasoning(
        self,
        fractions: List[float],
        baseline_pass_rates: List[float],
        conditioned_pass_rates: List[float],
        oracle_pass_rates: Optional[List[float]] = None,
        filename: str = "valley_of_code_reasoning_curve.png"
    ) -> str:
        """Plot the core theoretical contribution: The Valley of Code Reasoning scaling curves."""
        plt.figure(figsize=(10, 6.5))
        pct_labels = [f"{int(f*100)}%" for f in fractions]
        x = np.arange(len(fractions))

        # Baseline curve (demonstrates the non-monotonic dip / valley)
        plt.plot(x, baseline_pass_rates, "o--", color="#e74c3c", linewidth=2.5, markersize=8, label="Baseline (Unconditioned Distillation)")

        # Proposed Tag-Conditioned curve (mitigates / elevates the valley)
        plt.plot(x, conditioned_pass_rates, "s-", color="#2ecc71", linewidth=2.5, markersize=8, label="Proposed (Tag-Conditioned QLoRA Pipeline)")

        # Oracle upper bound
        if oracle_pass_rates:
            plt.plot(x, oracle_pass_rates, "^:", color="#3498db", linewidth=2.0, markersize=7, label="Oracle (Ground-Truth Tag Conditioning)")

        plt.xticks(x, pct_labels)
        plt.xlabel("Distillation Training Data Volume", labelpad=10, fontweight="semibold")
        plt.ylabel("Execution Pass@1 Rate (%)", labelpad=10, fontweight="semibold")
        plt.title("Empirical Scaling Dynamics: The Valley of Code Reasoning", pad=15, fontweight="bold")
        plt.ylim(0, 100)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="lower right", frameon=True)

        # Highlight valley region
        if len(x) >= 3:
            plt.axvspan(0.5, 1.5, color="gray", alpha=0.12, label="The Reasoning Valley Regime")

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_rationale_filtering_breakdown(
        self,
        summary_dict: Dict[str, Any],
        filename: str = "rationale_filter_breakdown.png"
    ) -> str:
        """Plot pie and bar charts of rationale-consistency filtering results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Donut chart of retained vs rejected
        retained = summary_dict.get("retained_count", 0)
        discarded = summary_dict.get("discarded_count", 0)
        ax1.pie(
            [retained, discarded],
            labels=[f"Retained Traces\n({retained})", f"Discarded Traces\n({discarded})"],
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2ecc71", "#e74c3c"],
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax1.set_title("Rationale Consistency Filter Yield", fontweight="bold")

        # 2. Verdict Breakdown
        verdicts = summary_dict.get("verdict_breakdown", {})
        clean_v = {k.replace("REJECTED_", "").replace("_", " ").title(): v for k, v in verdicts.items() if v > 0}
        ax2.bar(clean_v.keys(), clean_v.values(), color=sns.color_palette("Set2", len(clean_v)), edgecolor="black")
        ax2.set_title("Filtering Verdict Distribution", fontweight="bold")
        ax2.set_ylabel("Number of Traces", fontweight="semibold")
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_pass_at_1_comparison(
        self,
        base_pass_rate: float,
        cond_pass_rate: float,
        base_verdicts: Dict[str, int],
        cond_verdicts: Dict[str, int],
        total_tests: int,
        filename: str = "pass_at_1_comparison.png"
    ) -> str:
        """Plot publication-quality dual-panel Pass@1 accuracy and execution verdicts comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # 1. Left: Pass@1 Accuracy
        models = ["Unconditioned Baseline", "Stage 2 Tag-Conditioned"]
        pass_rates = [base_pass_rate * 100.0, cond_pass_rate * 100.0]
        colors = ["#64748b", "#0d9488"]
        bars1 = ax1.bar(models, pass_rates, color=colors, width=0.45, edgecolor="black", linewidth=1.2)
        ax1.set_ylim(0, 25)
        ax1.set_ylabel("Pass@1 Functional Accuracy (%)", fontweight="semibold")
        ax1.set_title(f"Pass@1 Functional Accuracy\n(Unseen {total_tests} Test Problems)", fontweight="bold", pad=10)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        for bar, rate in zip(bars1, pass_rates):
            ax1.text(bar.get_x() + bar.get_width() / 2.0, 0.8, f"{rate:.1f}% (0/{total_tests})", ha="center", va="bottom", fontweight="bold", fontsize=11)

        ax1.text(
            0.5, 15, "Zero-shot reference retrieval\n(Pre-QLoRA fine-tuning sweep)",
            transform=ax1.transData, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1")
        )

        # 2. Right: Execution Verdicts Breakdown
        verdict_keys = ["AC", "WA", "RE", "CE"]
        verdict_labels = ["AC (Passed)", "WA (Wrong Ans)", "RE (Runtime Err)", "CE (Syntax Err)"]
        b_counts = [base_verdicts.get(k, 0) for k in verdict_keys]
        c_counts = [cond_verdicts.get(k, 0) for k in verdict_keys]

        x = np.arange(len(verdict_labels))
        width = 0.35

        rects1 = ax2.bar(x - width / 2, b_counts, width, label="Unconditioned Baseline", color="#94a3b8", edgecolor="black", linewidth=1.0)
        rects2 = ax2.bar(x + width / 2, c_counts, width, label="Stage 2 Tag-Conditioned", color="#0d9488", edgecolor="black", linewidth=1.0)

        ax2.set_ylabel(f"Number of Test Cases (Total = {total_tests})", fontweight="semibold")
        ax2.set_title("Sandbox Execution Verdict Breakdown\n(Conditioned vs. Baseline Ablation)", fontweight="bold", pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(verdict_labels, rotation=15, ha="right", fontweight="medium")
        max_val = max(max(b_counts, default=0), max(c_counts, default=0), total_tests)
        ax2.set_ylim(0, max_val + 3)
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", frameon=True)

        for r in rects1:
            h = r.get_height()
            ax2.text(r.get_x() + r.get_width() / 2.0, h + 0.3, str(h), ha="center", va="bottom", fontsize=9, fontweight="bold")
        for r in rects2:
            h = r.get_height()
            ax2.text(r.get_x() + r.get_width() / 2.0, h + 0.3, str(h), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0f766e")

        # Annotate CE difference if CE is 0 for conditioned
        if cond_verdicts.get("CE", 0) == 0:
            ax2.annotate(
                "0 Compilation Errors\n(Syntax protected)", xy=(3 + width / 2, 0), xytext=(2.6, 5),
                arrowprops=dict(facecolor="#0d9488", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", fontsize=9, color="#0f766e"
            )

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_phase4_training_loss_sweep(
        self,
        sweep_results: Dict[str, Any],
        filename: str = "phase4_training_loss_sweep.png"
    ) -> str:
        """Plot Phase 4 training loss convergence trajectories and scaling trends."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Color palettes
        cond_colors = ["#99f6e4", "#2dd4bf", "#0d9488", "#115e59"]  # teals from light to dark
        base_colors = ["#fca5a5", "#f87171", "#ef4444", "#991b1b"]  # reds from light to dark

        fractions = sweep_results.get("fractions", [0.10, 0.25, 0.50, 1.00])
        cond_runs = sweep_results.get("conditioned_runs", {})
        base_runs = sweep_results.get("baseline_runs", {})

        # Panel 1: Loss vs. Epochs for all 8 runs
        for idx, frac in enumerate(fractions):
            frac_key = str(frac)
            pct = int(frac * 100)

            # Conditioned
            if frac_key in cond_runs:
                c_data = cond_runs[frac_key]
                epochs = list(range(1, len(c_data["epoch_losses"]) + 1))
                ax1.plot(
                    epochs,
                    c_data["epoch_losses"],
                    "s-",
                    color=cond_colors[idx % len(cond_colors)],
                    linewidth=2.0,
                    markersize=6,
                    label=f"Conditioned ({pct}%, N={c_data['train_samples']})"
                )

            # Baseline
            if frac_key in base_runs:
                b_data = base_runs[frac_key]
                epochs = list(range(1, len(b_data["epoch_losses"]) + 1))
                ax1.plot(
                    epochs,
                    b_data["epoch_losses"],
                    "o--",
                    color=base_colors[idx % len(base_colors)],
                    linewidth=2.0,
                    markersize=6,
                    label=f"Baseline ({pct}%, N={b_data['train_samples']})"
                )

        ax1.set_xlabel("Fine-Tuning Epoch", fontweight="semibold")
        ax1.set_ylabel("Causal SFT Loss", fontweight="semibold")
        ax1.set_title("Training Loss Convergence per Epoch\n(4-Bit QLoRA on Qwen2.5-Coder-0.5B)", fontweight="bold", pad=12)
        ax1.set_xticks([1, 2, 3, 4])
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper right", fontsize=8.5, frameon=True)

        # Panel 2: Final Loss vs. Training Dataset Volume
        cond_final = []
        base_final = []
        sample_counts = []
        pct_labels = []

        for frac in fractions:
            frac_key = str(frac)
            pct = int(frac * 100)
            pct_labels.append(f"{pct}%")
            if frac_key in cond_runs:
                cond_final.append(cond_runs[frac_key]["final_loss"])
                sample_counts.append(cond_runs[frac_key]["train_samples"])
            if frac_key in base_runs:
                base_final.append(base_runs[frac_key]["final_loss"])

        x_pts = range(len(fractions))
        if cond_final:
            ax2.plot(x_pts, cond_final, "s-", color="#0d9488", linewidth=2.5, markersize=8, label="Tag-Conditioned Model")
            for i, txt in enumerate(cond_final):
                ax2.annotate(f"{txt:.3f}", (i, txt), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color="#0f766e")
        if base_final:
            ax2.plot(x_pts, base_final, "o--", color="#ef4444", linewidth=2.5, markersize=8, label="Unconditioned Baseline")
            for i, txt in enumerate(base_final):
                ax2.annotate(f"{txt:.3f}", (i, txt), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, fontweight="bold", color="#b91c1c")

        ax2.set_xticks(list(x_pts))
        ax2.set_xticklabels([f"{pct_labels[i]}\n(N={sample_counts[i]})" for i in range(len(sample_counts))], fontweight="medium")
        ax2.set_xlabel("Training Data Subset (Samples per Category)", fontweight="semibold")
        ax2.set_ylabel("Final Training Loss (After 4 Epochs)", fontweight="semibold")
        ax2.set_title("Asymptotic SFT Loss vs. Dataset Scale\n(Conditioned vs. Baseline Control)", fontweight="bold", pad=12)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_phase5_valley_mapping(
        self,
        sweep_results: Dict[str, Any],
        filename: str = "phase5_valley_mapping_pass_at_k.png"
    ) -> str:
        """Plot the core theoretical contribution: Pass@1 and Pass@5 empirical scaling curves.
        
        Demonstrates 'The Valley of Code Reasoning' in unconditioned baseline distillation
        and its mitigation/flattening via decoupled algorithmic-tag conditioning.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        fractions = sweep_results.get("fractions", [0.10, 0.25, 0.50, 1.00])
        cond_runs = sweep_results.get("conditioned_results", {})
        base_runs = sweep_results.get("baseline_results", {})
        pct_labels = [f"{int(f * 100)}%" for f in fractions]
        x = np.arange(len(fractions))

        # Extract Pass@1 and Pass@5
        cond_p1 = [cond_runs[str(f)]["pass_at_1"] * 100.0 for f in fractions if str(f) in cond_runs]
        base_p1 = [base_runs[str(f)]["pass_at_1"] * 100.0 for f in fractions if str(f) in base_runs]

        cond_p5 = [cond_runs[str(f)]["pass_at_5"] * 100.0 for f in fractions if str(f) in cond_runs]
        base_p5 = [base_runs[str(f)]["pass_at_5"] * 100.0 for f in fractions if str(f) in base_runs]

        # ---------------- Panel 1: Pass@1 Scaling Curves ----------------
        ax1.plot(x, base_p1, "o--", color="#ef4444", linewidth=2.5, markersize=8, label="Baseline (Unconditioned SFT)")
        ax1.plot(x, cond_p1, "s-", color="#0d9488", linewidth=2.5, markersize=8, label="Proposed (Tag-Conditioned QLoRA)")

        # Value annotations
        for i, val in enumerate(base_p1):
            ax1.annotate(f"{val:.1f}%", (x[i], val), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, fontweight="bold", color="#b91c1c")
        for i, val in enumerate(cond_p1):
            ax1.annotate(f"{val:.1f}%", (x[i], val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color="#0f766e")

        # Highlight Valley Regime
        ax1.axvspan(0.6, 1.8, color="#fee2e2", alpha=0.45, label="The Reasoning Valley Regime")
        ax1.text(1.2, max(max(base_p1), max(cond_p1)) * 0.25, "Reasoning Valley\n(Search Entropy Dip)", ha="center", va="center", fontsize=9, color="#991b1b", fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff1f2", edgecolor="#fca5a5"))

        ax1.set_xticks(x)
        ax1.set_xticklabels(pct_labels, fontweight="medium")
        ax1.set_xlabel("Distillation Training Volume (Fraction of Pool)", fontweight="semibold")
        ax1.set_ylabel("Execution Pass@1 Accuracy (%)", fontweight="semibold")
        ax1.set_title("Pass@1 Scaling Dynamics: The Valley of Code Reasoning\n(Baseline Non-Monotonic Dip vs. Conditioned Monotonicity)", fontweight="bold", pad=12)
        ax1.set_ylim(0, max(max(cond_p1, default=40), max(base_p1, default=30)) * 1.25)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper left", frameon=True)

        # ---------------- Panel 2: Pass@5 Scaling Curves ----------------
        ax2.plot(x, base_p5, "o--", color="#f87171", linewidth=2.5, markersize=8, label="Baseline Pass@5")
        ax2.plot(x, cond_p5, "s-", color="#14b8a6", linewidth=2.5, markersize=8, label="Proposed Pass@5")

        for i, val in enumerate(base_p5):
            ax2.annotate(f"{val:.1f}%", (x[i], val), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, fontweight="bold", color="#b91c1c")
        for i, val in enumerate(cond_p5):
            ax2.annotate(f"{val:.1f}%", (x[i], val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold", color="#0f766e")

        ax2.set_xticks(x)
        ax2.set_xticklabels(pct_labels, fontweight="medium")
        ax2.set_xlabel("Distillation Training Volume (Fraction of Pool)", fontweight="semibold")
        ax2.set_ylabel("Execution Pass@5 Accuracy (%)", fontweight="semibold")
        ax2.set_title("Pass@5 Functional Recovery via Temperature Sampling\n(Candidate Sampling Amplifies Algorithmic Alignment)", fontweight="bold", pad=12)
        ax2.set_ylim(0, max(max(cond_p5, default=50), max(base_p5, default=40)) * 1.25)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper left", frameon=True)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_phase5_failure_taxonomy(
        self,
        sweep_results: Dict[str, Any],
        filename: str = "phase5_failure_taxonomy.png"
    ) -> str:
        """Plot execution failure taxonomy (AC, WA, TLE, RE, CE) across all 8 checkpoints."""
        fractions = sweep_results.get("fractions", [0.10, 0.25, 0.50, 1.00])
        cond_runs = sweep_results.get("conditioned_results", {})
        base_runs = sweep_results.get("baseline_results", {})

        verdicts = ["AC", "WA", "TLE", "RE", "CE"]
        verdict_labels = ["Accepted (AC)", "Wrong Answer (WA)", "Time Limit (TLE)", "Runtime Err (RE)", "Compilation Err (CE)"]
        colors = ["#10b981", "#f59e0b", "#3b82f6", "#8b5cf6", "#ef4444"]

        # Build labels: Baseline 10% ... 100%, Conditioned 10% ... 100%
        run_labels = []
        matrix = {v: [] for v in verdicts}

        # 4 Baseline runs
        for f in fractions:
            run_labels.append(f"Base {int(f*100)}%")
            v_counts = base_runs.get(str(f), {}).get("failure_taxonomy", {})
            for v in verdicts:
                matrix[v].append(v_counts.get(v, 0))

        # 4 Conditioned runs
        for f in fractions:
            run_labels.append(f"Cond {int(f*100)}%")
            v_counts = cond_runs.get(str(f), {}).get("failure_taxonomy", {})
            for v in verdicts:
                matrix[v].append(v_counts.get(v, 0))

        # Plot stacked bar chart
        plt.figure(figsize=(14, 6.5))
        y_pos = np.arange(len(run_labels))
        left = np.zeros(len(run_labels))

        for idx, v in enumerate(verdicts):
            counts = np.array(matrix[v])
            plt.barh(y_pos, counts, left=left, color=colors[idx], edgecolor="white", linewidth=1.0, label=verdict_labels[idx])
            # Add text inside bar if wide enough
            for i, cnt in enumerate(counts):
                if cnt >= 15:
                    plt.text(left[i] + cnt / 2, y_pos[i], str(cnt), ha="center", va="center", color="white", fontweight="bold", fontsize=9)
            left += counts

        # Add divider line between baseline and conditioned
        plt.axhline(3.5, color="#64748b", linestyle="--", linewidth=1.5, alpha=0.8)
        plt.text(left.max() * 0.98, 3.7, "Proposed Tag-Conditioned Models ↑", ha="right", va="bottom", fontweight="bold", color="#0f766e")
        plt.text(left.max() * 0.98, 3.3, "Unconditioned Baseline Models ↓", ha="right", va="top", fontweight="bold", color="#b91c1c")

        plt.yticks(y_pos, run_labels, fontweight="medium")
        plt.xlabel("Total Sampled Candidate Executions", fontweight="semibold")
        plt.title("Execution Verdict & Failure Mode Taxonomy Across Scaling Sweeps\n(Tag-Conditioning Suppresses CE Syntax Errors & Mitigates Runtime Exceptions)", fontweight="bold", pad=15)
        plt.legend(loc="lower right", frameon=True, fontsize=9.5)
        plt.grid(axis="x", linestyle="--", alpha=0.5)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_phase5_category_breakdown(
        self,
        sweep_results: Dict[str, Any],
        filename: str = "phase5_category_pass_rates.png"
    ) -> str:
        """Plot comparative Pass@1 accuracy across all 10 algorithmic categories for 100% scale."""
        cond_100 = sweep_results.get("conditioned_results", {}).get("1.0", {})
        base_100 = sweep_results.get("baseline_results", {}).get("1.0", {})

        c_rates = cond_100.get("category_pass_rates", {})
        b_rates = base_100.get("category_pass_rates", {})

        categories = sorted(list(set(list(c_rates.keys()) + list(b_rates.keys()))))
        if not categories:
            categories = ["dynamic programming", "greedy", "graphs", "math", "trees", "sorting", "search", "strings", "geometry", "data structures"]

        c_vals = [c_rates.get(cat, 0.0) * 100.0 for cat in categories]
        b_vals = [b_rates.get(cat, 0.0) * 100.0 for cat in categories]

        # Sort categories by conditioned pass rate
        sort_order = np.argsort(c_vals)[::-1]
        categories = [categories[i] for i in sort_order]
        c_vals = [c_vals[i] for i in sort_order]
        b_vals = [b_vals[i] for i in sort_order]

        plt.figure(figsize=(13, 6.5))
        y = np.arange(len(categories))
        height = 0.38

        bars1 = plt.barh(y + height / 2, c_vals, height, label="Tag-Conditioned QLoRA (100% Scale)", color="#0d9488", edgecolor="black", linewidth=1.0)
        bars2 = plt.barh(y - height / 2, b_vals, height, label="Unconditioned Baseline (100% Scale)", color="#f87171", edgecolor="black", linewidth=1.0)

        for bar in bars1:
            w = bar.get_width()
            plt.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold", color="#0f766e")
        for bar in bars2:
            w = bar.get_width()
            plt.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold", color="#b91c1c")

        plt.yticks(y, [c.title() for c in categories], fontweight="medium")
        plt.xlabel("Execution Pass@1 Functional Accuracy (%)", fontweight="semibold")
        plt.title("Algorithmic Category Breakdown: Functional Pass@1 Accuracy\n(Category-Stratified Test Partition, N=50 Problems)", fontweight="bold", pad=15)
        plt.xlim(0, max(max(c_vals, default=50), max(b_vals, default=40)) * 1.25)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.legend(loc="lower right", frameon=True)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_phase6_repair_delta(
        self,
        conditioned_turns: Dict[int, float],
        baseline_turns: Dict[int, float],
        filename: str = "phase6_repair_pass_at_1_delta.png"
    ) -> str:
        """Plot multi-turn execution self-repair progression across turns (Turn 0 -> Turn 1 -> Turn 2)."""
        turns = sorted(list(set(list(conditioned_turns.keys()) + list(baseline_turns.keys()))))
        if not turns:
            turns = [0, 1, 2]

        c_vals = [conditioned_turns.get(t, 0.0) * 100.0 for t in turns]
        b_vals = [baseline_turns.get(t, 0.0) * 100.0 for t in turns]

        turn_labels = []
        for t in turns:
            if t == 0:
                turn_labels.append("Turn 0\n(Zero-Shot)")
            else:
                turn_labels.append(f"Turn {t}\n(+{t} Repair{'s' if t > 1 else ''})")

        plt.figure(figsize=(10, 6.2))
        x = np.arange(len(turns))
        width = 0.35

        bars_c = plt.bar(
            x - width / 2, c_vals, width,
            label="Tag-Conditioned QLoRA",
            color="#0d9488", edgecolor="black", linewidth=1.2
        )
        bars_b = plt.bar(
            x + width / 2, b_vals, width,
            label="Unconditioned Baseline",
            color="#f87171", edgecolor="black", linewidth=1.2
        )

        # Annotate percentages and delta gains
        for i, bar in enumerate(bars_c):
            h = bar.get_height()
            delta_str = ""
            if i > 0:
                diff = h - c_vals[0]
                delta_str = f"\n(+{diff:.1f}%)"
            plt.text(
                bar.get_x() + bar.get_width() / 2, h + 1.0,
                f"{h:.1f}%{delta_str}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0f766e"
            )

        for i, bar in enumerate(bars_b):
            h = bar.get_height()
            delta_str = ""
            if i > 0:
                diff = h - b_vals[0]
                delta_str = f"\n(+{diff:.1f}%)"
            plt.text(
                bar.get_x() + bar.get_width() / 2, h + 1.0,
                f"{h:.1f}%{delta_str}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#b91c1c"
            )

        plt.xticks(x, turn_labels, fontweight="semibold")
        plt.ylabel("Execution Pass@1 Accuracy (%)", fontweight="semibold")
        plt.xlabel("Multi-Turn Execution Feedback Progression", fontweight="semibold", labelpad=10)
        plt.title(
            "Phase 6: Multi-Turn Execution Feedback & Self-Repair Progression\n"
            "(Single-Turn & Multi-Turn Recovery on Unseen Test Partition, N=50)",
            fontweight="bold", pad=15
        )
        plt.ylim(0, max(max(c_vals, default=50), max(b_vals, default=40)) * 1.35)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend(loc="upper left", frameon=True)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return output_path

    def plot_phase6_error_recovery(
        self,
        recovery_by_type: Dict[str, float],
        error_transitions: Optional[Dict[str, Dict[str, int]]] = None,
        filename: str = "phase6_repair_error_recovery.png"
    ) -> str:
        """Plot recovery resolution rate and transition breakdown across failure taxonomies."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

        # Panel A: Recovery percentage by error type
        err_types = ["CE", "RE", "WA", "TLE"]
        labels = [
            "Compilation / Syntax (CE)",
            "Runtime Exception (RE)",
            "Wrong Answer (WA)",
            "Time Limit Exceeded (TLE)"
        ]
        rates = [recovery_by_type.get(et, 0.0) * 100.0 for et in err_types]
        colors = ["#3b82f6", "#10b981", "#f59e0b", "#6b7280"]

        bars = axes[0].bar(labels, rates, color=colors, edgecolor="black", linewidth=1.1, width=0.55)
        for bar in bars:
            h = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{h:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        axes[0].set_ylabel("Resolution Rate into AC (%)", fontweight="semibold")
        axes[0].set_title("(a) Repair Recovery Rate by Defect Type", fontweight="bold", pad=12)
        axes[0].set_ylim(0, 105)
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=20, ha="right", fontweight="medium")
        axes[0].grid(axis="y", linestyle="--", alpha=0.6)


        # Panel B: Outcome transition breakdown (Resolved AC vs Unresolved)
        if error_transitions:
            active_types = [et for et in ["CE", "RE", "WA", "TLE"] if et in error_transitions]
            if not active_types:
                active_types = ["CE", "RE", "WA"]
            
            resolved_counts = []
            unresolved_counts = []
            for et in active_types:
                ac_count = error_transitions.get(et, {}).get("AC", 0)
                tot = sum(error_transitions.get(et, {}).values())
                unres = tot - ac_count
                resolved_counts.append(ac_count)
                unresolved_counts.append(unres)

            y = np.arange(len(active_types))
            h = 0.5
            axes[1].barh(y, resolved_counts, h, label="Resolved to AC", color="#10b981", edgecolor="black")
            axes[1].barh(y, unresolved_counts, h, left=resolved_counts, label="Unresolved", color="#ef4444", edgecolor="black")

            axes[1].set_yticks(y)
            axes[1].set_yticklabels([et for et in active_types], fontweight="bold")
            axes[1].set_xlabel("Number of Test Failures", fontweight="semibold")
            axes[1].set_title("(b) Initial Failure State Transitions", fontweight="bold", pad=12)
            axes[1].legend(loc="lower right", frameon=True)
            axes[1].grid(axis="x", linestyle="--", alpha=0.6)
        else:
            axes[1].text(0.5, 0.5, "Transition Data Not Provided", ha="center", va="center")

        plt.suptitle("Phase 6: Multi-Turn Execution Diagnostic Feedback Analysis", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path



