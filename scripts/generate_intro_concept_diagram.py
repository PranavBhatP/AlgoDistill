#!/usr/bin/env python3
"""
Generates publication-quality Figure 1 for the IEEE Access research paper:
Contrasting monolithic token entanglement (Valley of Code Reasoning) with
the decoupled two-stage conditioning and closed-loop execution self-repair framework.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def generate_intro_diagram(output_path="ieee_access_report/intro_concept_diagram.png"):
    fig, ax = plt.subplots(figsize=(11.0, 4.6), dpi=300)
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 4.8)
    ax.axis('off')

    # =========================================================================
    # LEFT PANEL: The Research Problem (Token Entanglement & Valley Dip)
    # =========================================================================
    p_box = patches.FancyBboxPatch(
        (0.2, 0.2), 5.0, 4.4,
        boxstyle='round,pad=0.12',
        facecolor='#FDFBF9',
        edgecolor='#D35400',
        linewidth=1.4
    )
    ax.add_patch(p_box)
    
    # Header badge
    ax.text(
        2.7, 4.35,
        'The Problem: Token Entanglement & Valley Dip',
        fontsize=10.5, fontweight='bold', color='#A04000', ha='center', va='center'
    )

    # Monolithic Student Model
    m_box = patches.FancyBboxPatch(
        (0.6, 3.48), 4.2, 0.65,
        boxstyle='round,pad=0.08',
        facecolor='#FDEDEC',
        edgecolor='#E74C3C',
        lw=1.2
    )
    ax.add_patch(m_box)
    ax.text(2.7, 3.86, 'Monolithic Student Model (≤ 1.5B Parameters)', fontsize=9, fontweight='bold', color='#78281F', ha='center')
    ax.text(2.7, 3.63, 'Joint next-token generation on distilled Chain-of-Thought (CoT) traces', fontsize=7.4, color='#922B21', ha='center')

    # Down arrow to Entangled Dual-Objective
    ax.annotate(
        '', xy=(2.7, 3.12), xytext=(2.7, 3.48),
        arrowprops=dict(arrowstyle='->', lw=1.4, color='#C0392B')
    )

    # Entangled Box
    e_box = patches.FancyBboxPatch(
        (0.5, 2.15), 4.4, 0.95,
        boxstyle='round,pad=0.08',
        facecolor='#FFFFFF',
        edgecolor='#BDC3C7',
        lw=1.0
    )
    ax.add_patch(e_box)
    ax.text(2.7, 2.84, 'Entangled Dual-Objective Optimization:',
            fontsize=8.0, ha='center', va='center', color='#2C3E50', fontweight='bold')
    ax.text(2.7, 2.62, 'Abstract System-2 Algorithmic Planning vs. Concrete Syntax Emission',
            fontsize=7.5, ha='center', va='center', color='#2C3E50', fontweight='bold')
    ax.text(2.7, 2.36, 'Competing token gradients cause representation collapse at intermediate training scales',
            fontsize=7.0, style='italic', color='#7F8C8D', ha='center', va='center')

    # Valley curve illustration mini-graph
    curve_x = [1.0, 2.1, 3.2, 4.4]
    curve_y = [1.32, 0.94, 1.02, 1.48]
    
    # Draw reference dashed trend
    ax.plot([1.0, 4.4], [1.32, 1.48], color='#BDC3C7', linestyle='--', lw=1.0, zorder=2)
    # Draw actual valley curve
    ax.plot(curve_x, curve_y, color='#C0392B', lw=2.4, marker='o', markersize=5.5, zorder=3)
    
    # Data labels with verified factual numbers from benchmarking runs
    ax.text(1.0, 1.44, '10%\n(13.2%)', fontsize=6.8, ha='center', va='bottom', color='#2C3E50', fontweight='bold')
    ax.text(2.1, 0.76, '25% (11.2%)\nValley Dip (-2.0%)', fontsize=6.8, ha='center', va='top', color='#C0392B', fontweight='bold')
    ax.text(3.2, 1.14, '50%\n(11.6%)', fontsize=6.8, ha='center', va='bottom', color='#2C3E50', fontweight='bold')
    ax.text(4.4, 1.60, '100%\n(17.6%)', fontsize=6.8, ha='center', va='bottom', color='#2C3E50', fontweight='bold')

    ax.text(2.7, 0.42, 'Unconditioned Baseline: Non-Monotonic Scaling Collapse',
            fontsize=8.0, color='#922B21', ha='center', fontweight='bold')

    # =========================================================================
    # RIGHT PANEL: The Proposed Decoupled & Self-Repair Architecture
    # =========================================================================
    s_box = patches.FancyBboxPatch(
        (5.5, 0.2), 5.3, 4.4,
        boxstyle='round,pad=0.12',
        facecolor='#F8FBFD',
        edgecolor='#2471A3',
        linewidth=1.4
    )
    ax.add_patch(s_box)
    
    # Header badge
    ax.text(
        8.15, 4.35,
        'Proposed Solution: Decoupling + Closed-Loop Self-Repair',
        fontsize=10.5, fontweight='bold', color='#154360', ha='center', va='center'
    )

    # Core modules are positioned from x=7.0 to x=10.5 (width=3.5, center=8.75)
    # This leaves [5.5 to 7.0] (1.5 units) completely free as a dedicated clearance lane!
    
    # Stage 1: Classifier
    st1_box = patches.FancyBboxPatch(
        (7.0, 3.48), 3.5, 0.65,
        boxstyle='round,pad=0.08',
        facecolor='#E8F4F8',
        edgecolor='#2980B9',
        lw=1.2
    )
    ax.add_patch(st1_box)
    ax.text(8.75, 3.86, 'Stage 1: Algorithmic Classifier  P(τ | X)', fontsize=8.5, fontweight='bold', color='#154360', ha='center')
    ax.text(8.75, 3.63, 'Commits to Paradigm Tag τ (10-Class Taxonomy)', fontsize=7.2, color='#1B4F72', ha='center')

    # Arrow Stage 1 -> Stage 2
    ax.annotate(
        '', xy=(8.75, 2.97), xytext=(8.75, 3.48),
        arrowprops=dict(arrowstyle='->', lw=1.4, color='#2980B9')
    )
    # Tag annotation alongside arrow
    ax.text(8.9, 3.22, 'Algorithmic Tag τ', fontsize=7.2, color='#1A5276', fontweight='bold', ha='left', va='center')

    # Stage 2: Generator
    st2_box = patches.FancyBboxPatch(
        (7.0, 2.22), 3.5, 0.75,
        boxstyle='round,pad=0.08',
        facecolor='#EAFAF1',
        edgecolor='#27AE60',
        lw=1.2
    )
    ax.add_patch(st2_box)
    ax.text(8.75, 2.72, 'Stage 2: Tag-Conditioned Generator  P(C, R | X, τ)', fontsize=8.5, fontweight='bold', color='#145A32', ha='center')
    ax.text(8.75, 2.48, 'Synthesizes rationale R & code C conditioned on tag τ', fontsize=7.2, color='#196F3D', ha='center')
    ax.text(8.75, 2.31, '(Narrows hypothesis space; eliminates strategy drift)', fontsize=6.8, style='italic', color='#229954', ha='center')

    # Arrow Stage 2 -> Sandbox
    ax.annotate(
        '', xy=(8.75, 1.66), xytext=(8.75, 2.22),
        arrowprops=dict(arrowstyle='->', lw=1.4, color='#27AE60')
    )
    # Code annotation alongside arrow
    ax.text(8.9, 1.94, 'Candidate Code C', fontsize=7.2, color='#196F3D', fontweight='bold', ha='left', va='center')

    # POSIX Sandbox Judge
    sb_box = patches.FancyBboxPatch(
        (7.0, 0.98), 3.5, 0.68,
        boxstyle='round,pad=0.08',
        facecolor='#FEF9E7',
        edgecolor='#F39C12',
        lw=1.2
    )
    ax.add_patch(sb_box)
    ax.text(8.75, 1.45, 'POSIX Sandbox Judge (2.0s CPU / 512MB RAM)', fontsize=8.2, fontweight='bold', color='#7D6608', ha='center')
    ax.text(8.75, 1.25, 'Isolated subprocess execution against benchmark test suite', fontsize=7.0, color='#9A7D0A', ha='center')
    ax.text(8.75, 1.10, 'Emits Verdict: AC (Success) or WA / TLE / RE / CE', fontsize=6.8, style='italic', color='#7D6608', ha='center')

    # =========================================================================
    # REPAIR FEEDBACK LOOP (Cleanly routed in the dedicated clearance lane [5.5 to 7.0])
    # =========================================================================
    lane_x = 6.18
    y_start = 1.32
    y_end = 2.60

    # Draw step lines
    ax.plot([7.0, lane_x, lane_x, 7.0], [y_start, y_start, y_end, y_end],
            color='#C0392B', lw=1.5, zorder=3)
    # Arrow head entering Stage 2 cleanly
    ax.annotate(
        '', xy=(7.0, y_end), xytext=(6.75, y_end),
        arrowprops=dict(arrowstyle='->', lw=1.5, color='#C0392B')
    )

    # Feedback badge background placed squarely inside lane [5.65 to 6.71]
    fb_rect = patches.FancyBboxPatch(
        (5.65, 1.62), 1.06, 0.68,
        boxstyle='round,pad=0.04',
        facecolor='#FDEDEC',
        edgecolor='#E74C3C',
        lw=0.9,
        zorder=4
    )
    ax.add_patch(fb_rect)
    
    ax.text(
        6.18, 2.08, 'Verdict ≠ AC',
        fontsize=7.0, fontweight='bold', color='#78281F', ha='center', va='center', zorder=5
    )
    ax.text(
        6.18, 1.88, 'Diagnostic\nFeedback E',
        fontsize=6.4, fontweight='bold', color='#922B21', ha='center', va='center', linespacing=1.1, zorder=5
    )
    ax.text(
        6.18, 1.69, '(Max T=2 turns)',
        fontsize=5.8, style='italic', color='#78281F', ha='center', va='center', zorder=5
    )

    # Bottom summary comparison text
    ax.text(
        8.15, 0.42,
        'Conditioned: Strictly Monotonic (16.0% → 44.4%) | 4× Repair Gain (4.0% → 16.0%)',
        fontsize=7.8, color='#154360', ha='center', fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated clean concept diagram at {output_path}")

if __name__ == "__main__":
    generate_intro_diagram()
