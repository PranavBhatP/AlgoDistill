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
