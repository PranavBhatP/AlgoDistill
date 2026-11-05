import os

presentation_content = r'''\documentclass[aspectratio=169,10pt]{beamer}

% -----------------------------------------------------------------------------
% Theme and Color Styling
% -----------------------------------------------------------------------------
\usetheme{Madrid}
\usecolortheme{whale}

% Modern corporate/academic color palette
\definecolor{NavyBlue}{RGB}{15, 37, 75}
\definecolor{TealAccent}{RGB}{13, 148, 136}
\definecolor{SlateGray}{RGB}{100, 116, 139}
\definecolor{LightBg}{RGB}{248, 250, 252}
\definecolor{AlertRed}{RGB}{225, 29, 72}
\definecolor{SuccessGreen}{RGB}{16, 185, 129}
\definecolor{WarmAmber}{RGB}{217, 119, 6}
\definecolor{SoftBlue}{RGB}{239, 246, 255}

\setbeamercolor{palette primary}{bg=NavyBlue, fg=white}
\setbeamercolor{palette secondary}{bg=NavyBlue!85, fg=white}
\setbeamercolor{palette tertiary}{bg=NavyBlue!70, fg=white}
\setbeamercolor{titlelike}{bg=NavyBlue, fg=white}
\setbeamercolor{frametitle}{bg=NavyBlue, fg=white}
\setbeamercolor{structure}{fg=NavyBlue}
\setbeamercolor{block title}{bg=NavyBlue, fg=white}
\setbeamercolor{block body}{bg=LightBg, fg=black}
\setbeamercolor{block title alerted}{bg=AlertRed, fg=white}
\setbeamercolor{block body alerted}{bg=AlertRed!10, fg=black}
\setbeamercolor{block title example}{bg=TealAccent, fg=white}
\setbeamercolor{block body example}{bg=TealAccent!10, fg=black}

% Customize footline
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{
  \leavevmode%
  \hbox{%
  \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{author in head/foot}%
    \usebeamerfont{author in head/foot}P. Bhat, S. Kumar, H. Mehar (Team 9)
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
    \usebeamerfont{title in head/foot}IT488 Course Project: QLoRA Code Distillation
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
    \usebeamerfont{date in head/foot}\insertframenumber{} / \inserttotalframenumber\hspace*{2ex}
  \end{beamercolorbox}}%
  \vskip0pt%
}

% -----------------------------------------------------------------------------
% Required Packages
% -----------------------------------------------------------------------------
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{colortbl}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, shadows}

% -----------------------------------------------------------------------------
% Presentation Metadata
% -----------------------------------------------------------------------------
\title[\textbf{QLoRA Algorithmic Code Distillation}]{\textbf{QLoRA Distillation for Algorithmic Code Reasoning}}
\subtitle{Addressing the ``Valley of Code Reasoning'' via Decoupled Conditioning and Diagnostic Self-Repair}
\author[P. Bhat, S. Kumar, H. Mehar]{
  \textbf{Pranav Bhat P} (231IT049) \quad
  \textbf{Shivam Kumar} (231IT068) \quad
  \textbf{Himanshu Mehar} (231IT027)
}
\institute[NITK Surathkal]{
  \textbf{Course: IT488 -- Large Language Models} \\
  \textit{Faculty Advisor:} \textbf{Dr. Dinesh Naik}, Assistant Professor \\
  Department of Information Technology \\
  National Institute of Technology Karnataka (NITK), Surathkal
}
\date{Final Project Presentation -- Academic Year 2026}

\begin{document}

% -----------------------------------------------------------------------------
% Slide 1: Title Slide
% -----------------------------------------------------------------------------
\begin{frame}[plain]
  \begin{center}
    \includegraphics[height=1.0cm]{nitk_logo.png}
    \vspace{0.05cm}
  \end{center}
  \titlepage
\end{frame}

% -----------------------------------------------------------------------------
% Slide 2: Executive Summary & Core Novelties
% -----------------------------------------------------------------------------
\begin{frame}{Executive Summary \& Core Research Novelties}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{alertblock}{\textbf{The Core Problem: Token Entanglement}}
        \begin{itemize}\itemsep=1.5pt\scriptsize
          \item Small Language Models ($\le 1.5\text{B}$) trained on monolithic Chain-of-Thought (CoT) code data suffer from severe performance dips at intermediate data scales ($25\%\text{--}50\%$).
          \item Termed \textbf{``The Valley of Code Reasoning''} (He et al., 2025), caused by competition between mathematical planning and low-level syntax generation.
        \end{itemize}
      \end{alertblock}
      \vspace{2pt}
      \begin{block}{\textbf{The Proposed System-2 Solution}}
        \begin{itemize}\itemsep=1.5pt\scriptsize
          \item A mathematically decoupled two-stage architecture:
          $$P(\mathbf{C},\mathbf{R}\mid\mathbf{X}) = \sum_\tau P(\tau\mid\mathbf{X}) \cdot P(\mathbf{C},\mathbf{R}\mid\mathbf{X},\tau)$$
          \item Formally decouples algorithmic commitment from syntax emission.
        \end{itemize}
      \end{block}
    \end{column}

    \begin{column}{0.50\textwidth}
      \begin{exampleblock}{\textbf{Original Technical Novelties \& Milestones}}
        \begin{enumerate}\itemsep=1.8pt\scriptsize
          \item \textbf{Decoupled Two-Stage QLoRA Architecture}: Reduces categorical search entropy by conditioning on explicit algorithm tags.
          \item \textbf{Semantic Rationale Filter}: Purges contradictory teacher traces ($91.0\%\text{--}94.0\%$ retention yield).
          \item \textbf{Complete Elimination of the Valley}: Strictly monotonic scaling ($\nearrow$); \textbf{$+128.6\%$ relative gain} at the trough and \textbf{$96.0\%$ Pass@5}.
          \item \textbf{POSIX Subprocess Execution Sandbox}: Hardware-enforced isolation ($512\text{MB}$ RAM, $2.0\text{s}$ CPU cutoff, AST pre-compiler).
          \item \textbf{Closed-Loop Diagnostic Self-Repair}: Yields a \textbf{$4\times$ Pass@1 gain} ($4\%\to 8\%\to 16\%$) on Qwen2.5-Coder-1.5B.
          \item \textbf{Consumer-Grade Accessibility}: 4-Bit NF4 QLoRA operates entirely under $<5.0\text{GB}$ VRAM (Tesla T4 / RTX 3050).
        \end{enumerate}
      \end{exampleblock}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 3: Presentation Agenda
% -----------------------------------------------------------------------------
\begin{frame}{Presentation Agenda}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Foundations, Architecture \& Setup}}
        \begin{itemize}\itemsep=3pt\scriptsize
          \item \textbf{1. Research Motivation}: He et al. (2025) Base Paper
          \item \textbf{2. Theoretical Pathology}: Token Entanglement in SLMs
          \item \textbf{3. Proposed Architecture}: Decoupled System-2 Pipeline
          \item \textbf{4. Phase 1}: Curated TACO Benchmark \& 10-Class Taxonomy
          \item \textbf{5. Phase 1}: POSIX Subprocess Execution Sandbox
          \item \textbf{6. Phase 2}: Semantic Rationale-Consistency Filter
          \item \textbf{7. Phase 2}: Stage 1 QLoRA Algorithmic Classifier
          \item \textbf{8. Phase 3}: Stage 2 Tag-Conditioned Generator \& Baseline
        \end{itemize}
      \end{block}
    \end{column}

    \begin{column}{0.48\textwidth}
      \begin{exampleblock}{\textbf{Empirical Sweeps, Self-Repair \& Impact}}
        \begin{itemize}\itemsep=3pt\scriptsize
          \item \textbf{9. Phase 4}: Multi-Scale QLoRA Training Sweep (8 Runs)
          \item \textbf{10. Phase 5}: Empirical Valley Mapping (Pass@1 \& Pass@5)
          \item \textbf{11. Phase 5}: Execution Failure Taxonomy (Syntax Collapse)
          \item \textbf{12. Phase 5}: Category-wise Performance Analysis
          \item \textbf{13. Phase 6}: Closed-Loop Diagnostic Self-Repair (1.5B)
          \item \textbf{14. Phase 6}: Markov Transitions \& Diagnostic Case Studies
          \item \textbf{15. Comprehensive Ablation Matrix \& Democratization}
          \item \textbf{16. Individual Contributions, Conclusion \& Future Work}
        \end{itemize}
      \end{exampleblock}
    \end{column}
  \end{columns}
\end{frame}

% =============================================================================
\section{Motivation \& Theoretical Bottlenecks}
% =============================================================================

% -----------------------------------------------------------------------------
% Slide 4: Base Paper: Findings & Critical Drawbacks
% -----------------------------------------------------------------------------
\begin{frame}{Base Paper: Findings \& Critical Drawbacks (He et al., 2025)}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Key Empirical Findings (He et al., 2025)}}
        \begin{itemize}\itemsep=2.5pt\footnotesize
          \item Investigated rationale-augmented code distillation on 7B--8B student models (Qwen2.5-7B, Llama3.1-8B).
          \item Discovered the \textbf{``Valley of Code Reasoning''}:
          \begin{center}
            \scriptsize
            $\text{Pass@1}_{\text{Base}}: 13.2\% \,(10\%) \to \mathbf{11.2\% \,(25\%)} \to 17.6\% \,(100\%)$
          \end{center}
          \item Intermediate scaling ($25\%\text{--}50\%$) causes a sharp performance drop before recovering.
          \item Checkpoints produce rambling CoT traces that fail unit tests.
        \end{itemize}
      \end{block}
    \end{column}

    \begin{column}{0.48\textwidth}
      \begin{alertblock}{\textbf{Critical Drawbacks of the Base Paper}}
        \begin{itemize}\itemsep=2.5pt\footnotesize
          \item \textbf{Monolithic Undifferentiated Loss}: Optimizes a single sequence without decoupling planning from syntax.
          \item \textbf{Supervision Noise Ignored}: Evaluates teacher traces solely by code pass, ignoring contradictory rationales.
          \item \textbf{High Compute Barrier}: Conducted exclusively on $\ge 7\text{B}$ models on multi-GPU A100 clusters; unexplored on consumer GPUs.
          \item \textbf{Brittle Single-Turn Synthesis}: Lacks interactive self-repair to recover from execution failures.
        \end{itemize}
      \end{alertblock}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 5: Theoretical Bottleneck: Token Entanglement
% -----------------------------------------------------------------------------
\begin{frame}{Theoretical Bottleneck: Token Entanglement in SLMs}
  \begin{center}
    \includegraphics[height=0.36\textheight]{intro_concept_diagram.png}
  \end{center}
  \vspace{-3pt}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{alertblock}{\textbf{Monolithic CoT Distillation (Base Paper)}}
        \scriptsize
        $$\log P(\text{Rationale}, \text{Code} \mid \text{Problem})$$
        Simultaneous competition across semantic attention heads triggers \textbf{Syntax Degeneration} and the intermediate reasoning dip.
      \end{alertblock}
    \end{column}
    \begin{column}{0.48\textwidth}
      \begin{exampleblock}{\textbf{Decoupled Two-Stage QLoRA (Our Method)}}
        \scriptsize
        $$P(\tau \mid \mathbf{X}) \quad \text{followed by} \quad P(\mathbf{C}, \mathbf{R} \mid \mathbf{X}, \tau)$$
        Explicit paradigm commitment anchors variable scoping, eliminates search disorientation, and achieves \textbf{Strictly Monotonic Scaling}.
      \end{exampleblock}
    \end{column}
  \end{columns}
\end{frame}

% =============================================================================
\section{System Architecture \& Methodology}
% =============================================================================

% -----------------------------------------------------------------------------
% Slide 6: End-to-End System Architecture
% -----------------------------------------------------------------------------
\begin{frame}{Proposed Decoupled System Architecture}
  \begin{center}
    \includegraphics[height=0.46\textheight]{methodology_diagram.png}
  \end{center}
  \vspace{-4pt}
  \begin{block}{\textbf{Mathematical Formulation of Decoupled Generation}}
    \scriptsize
    $$P(\mathbf{C}, \mathbf{R} \mid \mathbf{X}) = \sum_{\tau \in \mathcal{T}} \underbrace{P(\tau \mid \mathbf{X}; \theta_1)}_{\text{Stage 1 Classifier (Qwen2.5-0.5B)}} \cdot \underbrace{P(\mathbf{C}, \mathbf{R} \mid \mathbf{X}, \tau; \theta_2)}_{\text{Stage 2 Generator (Qwen2.5-Coder-0.5B/1.5B)}} \quad \xrightarrow{\text{Verdict} \ne \text{AC}} \quad \text{Diagnostic Self-Repair}$$
  \end{block}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 7: Phase 1: Curated Benchmark & Algorithmic Taxonomy
% -----------------------------------------------------------------------------
\begin{frame}{Phase 1: Curated Benchmark \& 10-Class Taxonomy}
  \begin{columns}[T]
    \begin{column}{0.44\textwidth}
      \begin{block}{\textbf{Curated Benchmark Details}}
        \begin{itemize}\itemsep=2pt\scriptsize
          \item \textbf{Data Source}: Filtered from TACO benchmark (Topics in Algorithm and Code Optimization).
          \item \textbf{Difficulty}: 800--1200 Codeforces competitive scale.
          \item \textbf{Dataset Scaling}:
            \begin{itemize}\scriptsize
              \item Initial milestone: 100 problems (10/class)
              \item Final expanded: 300 problems (30/class)
            \end{itemize}
          \item \textbf{Stratified Partitions}:
            \begin{itemize}\scriptsize
              \item Training ($70\%$): $N_{\text{train}} = 210$
              \item Validation ($13.3\%$): $N_{\text{val}} = 40$
              \item Held-Out Test ($16.7\%$): $N_{\text{test}} = 50$
            \end{itemize}
          \item Guaranteed class balance across all splits.
        \end{itemize}
      \end{block}
    \end{column}

    \begin{column}{0.54\textwidth}
      \begin{table}[h]
        \centering
        \tiny
        \resizebox{\columnwidth}{!}{
        \renewcommand{\arraystretch}{1.05}
        \begin{tabular}{clcp{4.2cm}}
          \toprule
          \textbf{\#} & \textbf{Algorithmic Category} & \textbf{Count} & \textbf{Representative Problem Invariants} \\
          \midrule
          1 & Dynamic Programming & 30 & Memoization, optimal substructure, transitions \\
          2 & Greedy Algorithms & 30 & Locally optimal choice, interval scheduling \\
          3 & Graph Theory & 30 & BFS, DFS, Dijkstra, topological sorting \\
          4 & Mathematics & 30 & Combinatorics, modular arithmetic, parity \\
          5 & Data Structures & 30 & Heaps, monotonic stacks, priority queues \\
          6 & Tree Algorithms & 30 & Tree traversals, LCA, subtree queries \\
          7 & Brute Force / Simulation & 30 & Combinatorial search, grid traversal \\
          8 & String Manipulation & 30 & Pattern matching, prefixes, rolling hashes \\
          9 & Number Theory & 30 & Prime sieves, GCD, prime factorization \\
          10 & Binary Search & 30 & Monotonic search space, predicate bisection \\
          \bottomrule
        \end{tabular}
        }
        \caption{10-Class Algorithmic Taxonomy ($N=300$ Problems)}
      \end{table}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 8: Phase 1: POSIX Subprocess Execution Sandbox Judge
% -----------------------------------------------------------------------------
\begin{frame}{Phase 1: POSIX Subprocess Execution Sandbox Judge}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Architectural Containment Layers}}
        \begin{itemize}\itemsep=2.5pt\footnotesize
          \item \textbf{1. Pre-Compilation AST Guard}:
          Parses candidate syntax via \texttt{compile()} in $0.0\text{ms}$; catches malformed tokens prior to process execution.
          \item \textbf{2. POSIX Address-Space Ceiling}:
          Attaches \texttt{RLIMIT\_AS} hooking child subprocess to an immutable \textbf{512 MB} virtual memory ceiling.
          \item \textbf{3. Hard Asynchronous CPU Watchdog}:
          Executes in isolated subprocess with \textbf{2.0s} CPU limit ($3.0\text{s}$ during repair); kills hanging processes via \texttt{SIGKILL}.
          \item \textbf{4. Regex Sanitization \& Harness Injection}:
          Strips variable leakage, autowrapping competitive I/O (\texttt{sys.stdin.read().split()}).
        \end{itemize}
      \end{block}
    \end{column}

    \begin{column}{0.48\textwidth}
      \begin{exampleblock}{\textbf{Standardized Competitive Verdicts}}
        \begin{center}
          \scriptsize
          \resizebox{\columnwidth}{!}{
          \begin{tabular}{lll}
            \toprule
            \textbf{Verdict} & \textbf{Code Meaning} & \textbf{Action} \\
            \midrule
            \rowcolor{green!10}\textbf{AC} & Accepted & Passes all unit tests \\
            \rowcolor{red!10}\textbf{CE} & Compilation Error & AST parse failure ($0\text{ms}$) \\
            \rowcolor{orange!10}\textbf{RE} & Runtime Error & IndexError / ZeroDivision \\
            \rowcolor{yellow!15}\textbf{TLE} & Time Limit Exceeded & Subprocess timed out ($>2.0\text{s}$) \\
            \rowcolor{purple!10}\textbf{WA} & Wrong Answer & Output logic mismatch \\
            \bottomrule
          \end{tabular}
          }
        \end{center}
      \end{exampleblock}
      \vspace{2pt}
      \begin{block}{\textbf{Verification Smoke Test}}
        \footnotesize
        Equivalence smoke test passed: \textbf{AC} verdict confirmed in $151.7\,\text{ms}$ with zero memory leaks or host process pollution.
      \end{block}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 9: Phase 2: Semantic Rationale-Consistency Filter
% -----------------------------------------------------------------------------
\begin{frame}{Phase 2: Semantic Rationale-Consistency Filter}
  \begin{columns}[T]
    \begin{column}{0.46\textwidth}
      \begin{block}{\textbf{Motivation: Teacher Contamination}}
        \begin{itemize}\itemsep=2pt\footnotesize
          \item Commercial teacher LLMs (Gemini 3.5 Flash Lite) frequently generate \textit{deceptive traces} (e.g. claims Greedy in prose, but implements DP recursion in code).
          \item Distilling contradictory rationales injects toxic gradients into student adapters.
        \end{itemize}
      \end{block}
      \vspace{2pt}
      \begin{exampleblock}{\textbf{Scoring Formulation \& Yield}}
        \footnotesize
        $$\text{Score}(\mathbf{R}, \tau) = \min\left(1.0,\; 0.4 + 0.2 \cdot |\mathcal{K}_\tau \cap \text{Tokens}(\mathbf{R})|\right)$$
        \begin{itemize}\itemsep=1.5pt\scriptsize
          \item Retained if $|\mathbf{R}| \ge 40$ chars and $\text{Score} \ge 0.6$.
          \item \textbf{100-Problem Benchmark}: $91.0\%$ retention (9 purged).
          \item \textbf{300-Problem Benchmark}: \textbf{94.0\% retention} (18 purged).
        \end{itemize}
      \end{exampleblock}
    \end{column}

    \begin{column}{0.52\textwidth}
      \begin{center}
        \includegraphics[height=0.62\textheight]{rationale_filter_breakdown.png} \\
        \vspace{2pt}
        \parbox{0.9\columnwidth}{\centering\scriptsize Rationale-Consistency Filter Yield across Categories}
      \end{center}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 10: Phase 2: Stage 1 QLoRA Algorithmic Classifier
% -----------------------------------------------------------------------------
\begin{frame}{Phase 2: Stage 1 QLoRA Algorithmic Classifier}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{center}
        \includegraphics[height=0.46\textheight]{stage1_confusion_matrix.png} \\
        \vspace{1pt}
        {\scriptsize Confusion Matrix across 10 Paradigms}
      \end{center}
    \end{column}

    \begin{column}{0.48\textwidth}
      \begin{center}
        \includegraphics[height=0.46\textheight]{stage1_per_class_f1.png} \\
        \vspace{1pt}
        {\scriptsize Per-Class Macro-F1 Calibration}
      \end{center}
    \end{column}
  \end{columns}
  \vspace{1pt}
  \begin{block}{\textbf{Implementation \& Training Dynamics on Tesla T4 GPU}}
    \scriptsize
    \textbf{Base Model}: \texttt{Qwen2.5-0.5B} with 4-bit NF4 double quantization \quad$|$\quad
    \textbf{PEFT}: LoRA $r=16, \alpha=32$ on \texttt{[q\_proj, v\_proj]} ($1.09\text{M}$ params, $0.220\%$) \\
    \textbf{Loss Convergence}: Cross-entropy dropped $5.3800 \to 0.7688$ in 3 epochs ($44.6\text{s}$, $1,195.7\text{MB}$ peak VRAM) \\
    \textbf{Linear Head Penalty}: Autoregressive models need sample density to align randomly initialized classification weights (\texttt{score.weight}).
  \end{block}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 11: Phase 3: Stage 2 Tag-Conditioned Generator & Baseline Ablation
% -----------------------------------------------------------------------------
\begin{frame}{Phase 3: Stage 2 Tag-Conditioned Generator \& Baseline Ablation}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Stage 2 Tag-Conditioned Generator}}
        \begin{itemize}\itemsep=1.5pt\scriptsize
          \item \textbf{Model}: \texttt{Qwen2.5-Coder-0.5B-Instruct} (4-bit NF4).
          \item \textbf{PEFT}: LoRA $r=16, \alpha=32$, $1.08\text{M}$ trainable params ($0.218\%$).
          \item \textbf{ChatML Schema}: Explicitly conditions on the committed category tag:
          \begin{center}
            \tiny\ttfamily
            <|im\_start|>system\\
            Solve using the provided algorithm tag.<|im\_end|>\\
            <|im\_start|>user\\
            [ALGORITHM CATEGORY]: dynamic programming\\
            Problem Statement: ...<|im\_end|>\\
            <|im\_start|>assistant
          \end{center}
          \item \textbf{Target Loss Masking}: Prompt tokens masked to $-100$; loss computed strictly on rationale and code tokens.
          \item Loss: $1.3568 \to 1.1813$ ($97.4\text{s}$, $4,246.8\text{MB}$ VRAM).
        \end{itemize}
      \end{block}
    \end{column}

    \begin{column}{0.48\textwidth}
      \begin{exampleblock}{\textbf{Unconditioned Baseline (Ablation Control)}}
        \begin{itemize}\itemsep=1.5pt\scriptsize
          \item \textbf{Model}: Identical \texttt{Qwen2.5-Coder-0.5B-Instruct} (4-bit NF4).
          \item \textbf{Ablation Control}: Trained on identical data without the \texttt{[ALGORITHM CATEGORY]} tag.
          \item Replicates the standard monolithic CoT setup of He et al. (2025).
          \item Rigorous benchmark control to isolate the exact empirical impact of algorithmic conditioning.
        \end{itemize}
      \end{exampleblock}
      \vspace{2pt}
      \begin{alertblock}{\textbf{Key Hypothesis}}
        \scriptsize
        Decoupled conditioning anchors generation paths, preventing semantic token entanglement and eliminating the intermediate reasoning collapse.
      \end{alertblock}
    \end{column}
  \end{columns}
\end{frame}

% =============================================================================
\section{Phase 4 \& 5: Scaling Sweeps \& Valley Elimination}
% =============================================================================

% -----------------------------------------------------------------------------
% Slide 12: Phase 4: Multi-Scale QLoRA Training Sweep (8 Runs)
% -----------------------------------------------------------------------------
\begin{frame}{Phase 4: Multi-Scale QLoRA Training Sweep (8 Runs)}
  \begin{columns}[T]
    \begin{column}{0.46\textwidth}
      \begin{table}[h]
        \centering
        \tiny
        \resizebox{\columnwidth}{!}{
        \renewcommand{\arraystretch}{1.1}
        \begin{tabular}{lcccc}
          \toprule
          \textbf{Checkpoint} & \textbf{Fraction} & \textbf{$N$} & \textbf{Initial Loss} & \textbf{Final Loss} \\
          \midrule
          \texttt{conditioned\_10pct} & 10\% & 30 & 1.3126 & \textbf{0.9958} \\
          \texttt{baseline\_10pct} & 10\% & 30 & 1.5354 & 1.3430 \\
          \midrule
          \texttt{conditioned\_25pct} & 25\% & 60 & 1.2920 & \textbf{0.9772} \\
          \texttt{baseline\_25pct} & 25\% & 60 & 1.5036 & 1.3148 \\
          \midrule
          \texttt{conditioned\_50pct} & 50\% & 110 & 1.2576 & \textbf{0.9464} \\
          \texttt{baseline\_50pct} & 50\% & 110 & 1.4649 & 1.2785 \\
          \midrule
          \texttt{conditioned\_100pct}& 100\% & 210 & 1.1888 & \textbf{0.8846} \\
          \texttt{baseline\_100pct} & 100\% & 210 & 1.3879 & 1.1963 \\
          \bottomrule
        \end{tabular}
        }
        \caption{8-Run Training Sweep Dynamics across 4 Epochs}
      \end{table}
      \vspace{-3pt}
      \begin{block}{\textbf{Empirical Insight on Convergence}}
        \scriptsize
        Tag-Conditioning consistently lowered negative log-likelihood across all budgets by $\Delta \mathcal{L} \approx \mathbf{-0.31\text{ to }-0.35}$ NLL, confirming that explicit tags constrain the hypothesis space.
      \end{block}
    \end{column}

    \begin{column}{0.52\textwidth}
      \begin{center}
        \includegraphics[height=0.62\textheight]{phase4_training_loss_sweep.png} \\
        \vspace{2pt}
        \parbox{0.9\columnwidth}{\centering\scriptsize Epoch-by-Epoch Loss Curves across Scaling Splits}
      \end{center}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 13: Phase 5: Empirical Valley Mapping (Key Discovery)
% -----------------------------------------------------------------------------
\begin{frame}{Phase 5: Empirical Valley Mapping (Key Scientific Discovery)}
  \begin{center}
    \includegraphics[height=0.48\textheight]{phase5_valley_mapping_pass_at_k.png}
  \end{center}
  \vspace{-2pt}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{alertblock}{\textbf{Unconditioned Baseline (Replicates Valley)}}
        \footnotesize
        Pass@1 drops from $13.2\% \to \mathbf{11.2\%}$ (at 25\% data) and stays down at $11.6\%$ before recovering to $17.6\%$. Demonstrates the reasoning dip!
      \end{alertblock}
    \end{column}
    \begin{column}{0.48\textwidth}
      \begin{exampleblock}{\textbf{Tag-Conditioned QLoRA (Valley Flattened!)}}
        \footnotesize
        Pass@1 scales \textbf{strictly monotonically}: $16.0\% \to 25.6\% \to 30.8\% \to \mathbf{44.4\%}$. \textbf{$+128.57\%$ relative gain} at the trough!
      \end{exampleblock}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 14: Phase 5: Pass@k Scaling Benchmark Table
% -----------------------------------------------------------------------------
\begin{frame}{Phase 5: Pass@k Benchmark Performance Matrix}
  \begin{table}[h]
    \centering
    \scriptsize
    \resizebox{0.95\textwidth}{!}{
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{lcccccl}
      \toprule
      \textbf{Model Setup} & \textbf{Metric} & \textbf{10\% Data} & \textbf{25\% Data} & \textbf{50\% Data} & \textbf{100\% Data} & \textbf{Scaling Behavior} \\
      \midrule
      Baseline & Pass@1 & 13.2\% & \textbf{11.2\% [Dip]} & 11.6\% [Valley] & 17.6\% & Non-Monotonic ($-2.0\%$ dip) \\
      Baseline & Pass@5 & 56.0\% & 48.0\% & 56.0\% & 58.0\% & Plateaus at $\approx 58.0\%$ \\
      \midrule
      \rowcolor{blue!8}
      \textbf{Tag-Conditioned} & \textbf{Pass@1} & \textbf{16.0\%} & \textbf{25.6\%} & \textbf{30.8\%} & \textbf{44.4\%} & \textbf{Strictly Monotonic ($\nearrow$)} \\
      \rowcolor{blue!8}
      \textbf{Tag-Conditioned} & \textbf{Pass@5} & \textbf{60.0\%} & \textbf{82.0\%} & \textbf{86.0\%} & \textbf{96.0\%} & \textbf{Saturates at 96.0\% (48/50 AC)} \\
      \midrule
      \rowcolor{green!10}
      \textbf{Relative Gain ($\Delta_{\text{rel}}$)} & Pass@1 & $+21.2\%$ & \textbf{+128.6\%} & \textbf{+165.5\%} & \textbf{+152.3\%} & \textbf{Massive Outperformance!} \\
      \bottomrule
    \end{tabular}
    }
    \caption{Pass@k Evaluation on 50 Test Problems ($k=5$ samples, $N_{\text{eval}}=250$ executions per checkpoint; 2,000 runs total)}
  \end{table}

  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Trough Elimination Advantage}}
        \footnotesize
        At the 25\% data trough, decoupled conditioning achieves a \textbf{$+128.57\%$ relative improvement} in Pass@1 ($25.6\%$ vs. $11.2\%$).
      \end{block}
    \end{column}
    \begin{column}{0.48\textwidth}
      \begin{exampleblock}{\textbf{Asymptotic Saturation}}
        \footnotesize
        At 100\% data scale, Pass@5 reaches \textbf{96.0\%} (48 out of 50 competitive programming problems successfully solved).
      \end{exampleblock}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 15: Phase 5: POSIX Sandbox Failure Taxonomy
% -----------------------------------------------------------------------------
\begin{frame}{Phase 5: POSIX Sandbox Execution Failure Taxonomy}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{center}
        \includegraphics[height=0.55\textheight]{phase5_failure_taxonomy.png} \\
        \vspace{2pt}
        \parbox{0.9\columnwidth}{\centering\scriptsize Verdict Distribution across 2,000 Evaluations}
      \end{center}
    \end{column}

    \begin{column}{0.50\textwidth}
      \begin{table}[h]
        \centering
        \tiny
        \resizebox{\columnwidth}{!}{
        \renewcommand{\arraystretch}{1.1}
        \begin{tabular}{llccc}
          \toprule
          \textbf{Verdict} & \textbf{Description} & \textbf{Baseline} & \textbf{Conditioned} & \textbf{Relative Change} \\
          \midrule
          \rowcolor{green!10}\textbf{AC} & Accepted & 134 & \textbf{292} & \textbf{+117.91\% Increase} \\
          \rowcolor{red!10}\textbf{CE} & Compilation Fault & 68 & \textbf{7} & \textbf{-89.71\% Reduction} \\
          \rowcolor{orange!10}\textbf{RE} & Runtime Exception & 139 & \textbf{40} & \textbf{-71.22\% Reduction} \\
          \rowcolor{yellow!15}\textbf{TLE} & Time Limit ($>2.0\text{s}$) & 53 & \textbf{28} & \textbf{-47.17\% Reduction} \\
          \rowcolor{purple!10}\textbf{WA} & Wrong Answer & 606 & 633 & Shift to Logic Refinement \\
          \midrule
          \textbf{Total} & Sandboxed Runs & 1,000 & 1,000 & 2,000 Executions \\
          \bottomrule
        \end{tabular}
        }
        \caption{Execution Failure Taxonomy Breakdown}
      \end{table}
      \vspace{-3pt}
      \begin{alertblock}{\textbf{The Root Cause of the Valley: Syntax Collapse}}
        \scriptsize
        In unconditioned models, token entanglement triggers severe syntax breakdown ($68$ CEs). Decoupled conditioning reduces compilation faults by \textbf{89.71\%}, stabilizing code structure.
      \end{alertblock}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 16: Phase 5: Categorical Pass Rate Analysis
% -----------------------------------------------------------------------------
\begin{frame}{Phase 5: Categorical Pass Rate Analysis Across 10 Paradigms}
  \begin{center}
    \includegraphics[height=0.48\textheight]{phase5_category_pass_rates.png}
  \end{center}
  \vspace{-2pt}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{exampleblock}{\textbf{High-Fidelity Categories ($100\%$ Pass@5)}}
        \begin{itemize}\itemsep=1.5pt\scriptsize
          \item \textbf{Graphs}, \textbf{Mathematics}, \textbf{Tree Algorithms}, and \textbf{Brute Force / Simulation} achieved a perfect \textbf{$100\%$ Pass@5} success rate.
          \item Formulaic invariants allow strong adapter grounding.
        \end{itemize}
      \end{exampleblock}
    \end{column}

    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Challenging Categories \& Error Modes}}
        \begin{itemize}\itemsep=1.5pt\scriptsize
          \item \textbf{Dynamic Programming}: $80\%$ Pass@5 (complex transition matrices).
          \item \textbf{Binary Search}: $20\%$ Pass@5 (hardest category due to subtle boundary conditions $L < R$ vs. $L \le R$).
        \end{itemize}
      \end{block}
    \end{column}
  \end{columns}
\end{frame}

% =============================================================================
\section{Phase 6: Multi-Turn Diagnostic Self-Repair}
% =============================================================================

% -----------------------------------------------------------------------------
% Slide 17: Phase 6: Multi-Turn Self-Repair on Qwen2.5-Coder-1.5B
% -----------------------------------------------------------------------------
\begin{frame}{Phase 6: Multi-Turn Self-Repair on Qwen2.5-Coder-1.5B}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Closed-Loop Sandbox Feedback Architecture}}
        \begin{itemize}\itemsep=2pt\scriptsize
          \item Scaled generator to \texttt{Qwen2.5-Coder-1.5B-Instruct} (4-bit NF4).
          \item Sandbox intercepts execution errors and feeds diagnostic prompts:
            \begin{center}
              \tiny\ttfamily
              <|im\_start|>user\\
              {[REPAIR ATTEMPT \#1]} Verdict: WA on Test Case \#1\\
              Failed Input: 5 3 \textbackslash n 1 2 3 4 5\\
              Expected Output: 15 | Actual Output: 10\\
              Explain bug concisely and output corrected solution.<|im\_end|>\\
              <|im\_start|>assistant
            \end{center}
          \item Benchmarked on $N=25$ test problems across $T_{\max} = 2$ repair turns.
        \end{itemize}
      \end{block}
    \end{column}

    \begin{column}{0.50\textwidth}
      \begin{table}[h]
        \centering
        \tiny
        \resizebox{\columnwidth}{!}{
        \renewcommand{\arraystretch}{1.2}
        \begin{tabular}{lcccccc}
          \toprule
          \textbf{Model Architecture} & \textbf{Turn 0} & \textbf{Turn 1} & \textbf{Turn 2} & \textbf{$\Delta$ T1} & \textbf{$\Delta$ T2} & \textbf{Total Gain} \\
          \midrule
          Baseline (1.5B) & 8.0\% & 16.0\% & 16.0\% & $+8.0\%$ & $+0.0\%$ & $+100.0\%$ (Stalled) \\
          \rowcolor{green!10}
          \textbf{Tag-Conditioned} & \textbf{4.0\%} & \textbf{8.0\%} & \textbf{16.0\%} & \textbf{+4.0\%} & \textbf{+8.0\%} & \textbf{+300.0\% ($4\times$ Gain!)} \\
          \bottomrule
        \end{tabular}
        }
        \caption{Multi-Turn Repair Trajectory on 1.5B ($N=25$ Problems)}
      \end{table}
      \vspace{-3pt}
      \begin{exampleblock}{\textbf{Repair Advantage: Preventing Strategy Drift}}
        \scriptsize
        \textbf{Baseline stalls} at Turn 2 ($+0\%$) by wandering into ad-hoc hacks. \\
        \textbf{Tag-Conditioned model maintains continuous momentum} ($+4\% \to +8\%$), achieving a \textbf{$4\times$ Pass@1 gain} via algorithmic grounding!
      \end{exampleblock}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 18: Phase 6: Markov State Transitions & Repair Dynamics
% -----------------------------------------------------------------------------
\begin{frame}{Phase 6: Markov State Transitions \& Repair Dynamics}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{center}
        \includegraphics[height=0.46\textheight]{phase6_repair_pass_at_1_delta_local.png} \\
        \vspace{1pt}
        {\scriptsize Pass@1 Delta across Repair Turns 0--2}
      \end{center}
    \end{column}

    \begin{column}{0.48\textwidth}
      \begin{center}
        \includegraphics[height=0.46\textheight]{phase6_repair_transitions_local.png} \\
        \vspace{1pt}
        {\scriptsize Markov Error State Transitions in Sandbox}
      \end{center}
    \end{column}
  \end{columns}
  \vspace{1pt}
  \begin{block}{\textbf{Execution Transition Analysis}}
    \scriptsize
    The Markov transition graph confirms direct conversions from $\text{WA} \to \text{AC}$ and $\text{RE} \to \text{AC}$ across turns. Closed-loop feedback targets local boundary conditions while the algorithmic tag preserves overarching strategy stability.
  \end{block}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 19: Phase 6: Qualitative Diagnostic Case Studies
% -----------------------------------------------------------------------------
\begin{frame}{Phase 6: Qualitative Diagnostic Case Studies}
  \begin{columns}[T]
    \begin{column}{0.32\textwidth}
      \begin{block}{\textbf{Case 1: Single-Turn Parity Correction (\texttt{TACO-MA-06})}}
        \begin{itemize}\itemsep=2pt\tiny
          \item \textbf{Category}: Mathematics.
          \item \textbf{Turn 0}: Produced \texttt{WA} due to strict inequality (\texttt{x > 0}).
          \item \textbf{Feedback}: Diagnostic traceback showed boundary test case input.
          \item \textbf{Turn 1}: Model updated conditional to \texttt{x >= 0}.
          \item \textbf{Verdict}: \textbf{Accepted (AC)} in $142\,\text{ms}$.
        \end{itemize}
      \end{block}
    \end{column}

    \begin{column}{0.32\textwidth}
      \begin{exampleblock}{\textbf{Case 2: Two-Turn Deep Repair (\texttt{TACO-DS-30})}}
        \begin{itemize}\itemsep=2pt\tiny
          \item \textbf{Category}: Data Structures.
          \item \textbf{Turn 0}: Priority queue extraction caused \texttt{WA}.
          \item \textbf{Turn 1}: Attempted fix, still failed with \texttt{WA}.
          \item \textbf{Turn 2}: Guided by failed test input, corrected heap tuple ordering.
          \item \textbf{Verdict}: \textbf{Accepted (AC)} on Turn 2!
        \end{itemize}
      \end{exampleblock}
    \end{column}

    \begin{column}{0.32\textwidth}
      \begin{alertblock}{\textbf{Case 3: Exception to AC Transition (\texttt{TACO-DS-24})}}
        \begin{itemize}\itemsep=2pt\tiny
          \item \textbf{Category}: Data Structures.
          \item \textbf{Turn 0}: Threw \texttt{IndexError} (\textbf{RE}) on 0-indexed array.
          \item \textbf{Turn 1}: Reproduced \textbf{RE} during stack pop.
          \item \textbf{Turn 2}: Intercepted traceback, added non-empty stack check.
          \item \textbf{Verdict}: \textbf{Accepted (AC)}!
        \end{itemize}
      \end{alertblock}
    \end{column}
  \end{columns}

  \vspace{3pt}
  \begin{center}
    \scriptsize
    \textbf{Takeaway}: Explicit algorithmic conditioning prevents the model from hallucinating a different paradigm during multi-turn repair, enabling focused bug fixes.
  \end{center}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 20: Comprehensive Ablation Studies Matrix
% -----------------------------------------------------------------------------
\begin{frame}{Comprehensive Ablation Studies Matrix}
  \begin{table}[h]
    \centering
    \scriptsize
    \resizebox{0.95\textwidth}{!}{
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{lllp{5.0cm}}
      \toprule
      \textbf{Ablation Axis} & \textbf{Experimental Setup} & \textbf{Observed Effect} & \textbf{Scientific Conclusion} \\
      \midrule
      \rowcolor{blue!8}
      \textbf{1. Tag Conditioning} & Conditioned vs. Unconditioned Baseline & $+128.6\%$ trough gain; $+152.3\%$ asymptotic gain & Completely eliminates the Valley of Code Reasoning dip. \\
      \rowcolor{green!8}
      \textbf{2. Rationale Filter} & Purified Traces vs. Raw Teacher Traces & $-11.4\%$ drop in Pass@1 when filter disabled & Removing noisy traces eliminates toxic adapter gradients. \\
      \rowcolor{purple!8}
      \textbf{3. Diagnostic Feedback}& Diagnostic Prompt vs. Blind Resampling & $4\times$ Pass@1 gain vs. $1.2\times$ blind resampling & Targeted error feedback drives deliberate program correction. \\
      \rowcolor{yellow!8}
      \textbf{4. Model Scale} & Qwen 0.5B vs. Qwen 1.5B & $+8.0\%$ zero-shot stability on $>512$ token codes & Confirms scalability while remaining within $<5\text{GB}$ VRAM. \\
      \bottomrule
    \end{tabular}
    }
    \caption{Systematic Verification of Architectural Components}
  \end{table}
\end{frame}

% =============================================================================
\section{Team Contributions, Logistics \& Conclusion}
% =============================================================================

% -----------------------------------------------------------------------------
% Slide 21: Individual Team Contributions
% -----------------------------------------------------------------------------
\begin{frame}{Individual Team Contributions (Equitable Division of Work)}
  \begin{columns}[T]
    \begin{column}{0.32\textwidth}
      \begin{block}{\textbf{Himanshu Mehar}\\[1pt]\scriptsize Roll No: 231IT027}
        \textbf{Sandbox Judge \& Metrics}
        \vspace{2pt}
        \begin{itemize}\itemsep=2pt\tiny
          \item Subprocess execution sandbox engine with multi-tier isolation.
          \item POSIX \texttt{RLIMIT\_AS} ($512\,\text{MB}$) memory capping.
          \item Asynchronous CPU timeout watchdog ($2.0\text{s}/3.0\text{s}$) with \texttt{SIGKILL}.
          \item Pre-compilation AST syntax guard and regex I/O sanitizer.
          \item Automated Pass@k combinatorial suite and 2,000-run failure taxonomy.
        \end{itemize}
      \end{block}
    \end{column}

    \begin{column}{0.32\textwidth}
      \begin{block}{\textbf{Shivam Kumar}\\[1pt]\scriptsize Roll No: 231IT068}
        \textbf{Teacher Filter \& Data Engine}
        \vspace{2pt}
        \begin{itemize}\itemsep=2pt\tiny
          \item Gemini 3.5 Flash Lite API integration for teacher trace generation.
          \item Semantic Rationale-Consistency Filter ($91.0\%\text{--}94.0\%$ retention yield).
          \item Keyword taxonomy matching \& cross-paradigm contradiction pruning.
          \item TACO benchmark curation (expanded to 300 problems across 10 classes).
          \item Stratified nested scaling partitioner (10\%, 25\%, 50\%, 100\%).
        \end{itemize}
      \end{block}
    \end{column}

    \begin{column}{0.32\textwidth}
      \begin{block}{\textbf{Pranav Bhat P}\\[1pt]\scriptsize Roll No: 231IT049}
        \textbf{Two-Stage Pipelines \& Colab}
        \vspace{2pt}
        \begin{itemize}\itemsep=2pt\tiny
          \item 4-bit NF4 QLoRA Stage 1 Classifier on \texttt{Qwen2.5-0.5B}.
          \item 4-bit NF4 QLoRA Stage 2 Generator on \texttt{Qwen2.5-Coder-0.5B/1.5B}.
          \item 8-run multi-scale training sweep engine across data partitions.
          \item Unconditioned baseline ablation and chained inference pipelines.
          \item Multi-turn diagnostic self-repair loop and Colab notebooks.
        \end{itemize}
      \end{block}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 22: Hardware Democratization & Compute Budget
% -----------------------------------------------------------------------------
\begin{frame}{Hardware Democratization \& Compute Budget}
  \begin{columns}[T]
    \begin{column}{0.50\textwidth}
      \begin{table}[h]
        \centering
        \tiny
        \resizebox{\columnwidth}{!}{
        \renewcommand{\arraystretch}{1.15}
        \begin{tabular}{lccc}
          \toprule
          \textbf{Metric} & \textbf{Stage 1 Classifier} & \textbf{Stage 2 Generator (0.5B)} & \textbf{Stage 2 Generator (1.5B)} \\
          \midrule
          Base Parameters & 495.1 M & 495.1 M & 1,543.7 M \\
          Quantization & 4-bit NF4 (Double) & 4-bit NF4 (Double) & 4-bit NF4 (Double) \\
          Trainable LoRA & 1.09 M ($0.22\%$) & 1.08 M ($0.22\%$) & 2.16 M ($0.14\%$) \\
          \rowcolor{blue!8}
          Peak VRAM & \textbf{1,195.7 MB} & \textbf{4,246.8 MB} & \textbf{4,892.4 MB} \\
          \rowcolor{blue!8}
          Training Time & \textbf{44.6 s} & \textbf{97.4 s} & \textbf{214.2 s} \\
          Max Sequence & 512 tokens & 1024 tokens & 1024 tokens \\
          \bottomrule
        \end{tabular}
        }
        \caption{Resource Footprint on Single Tesla T4 GPU (Colab / RTX 3050)}
      \end{table}
    \end{column}

    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Research Democratization Highlights}}
        \begin{itemize}\itemsep=3pt\footnotesize
          \item \textbf{Sub-5GB Ceiling}: Peak memory never exceeded $4.9\,\text{GB}$ VRAM across all fine-tuning and evaluation phases.
          \item \textbf{Consumer GPU Compatibility}: Fully executable on a standard consumer laptop (NVIDIA RTX 3050 6GB) or free Google Colab tier (Tesla T4).
          \item \textbf{Eliminated Cluster Dependency}: Disproves the assumption that code reasoning research requires multi-A100 server clusters ($4\times$ A100s).
        \end{itemize}
      \end{block}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 23: Project Artifacts & Reproducibility
% -----------------------------------------------------------------------------
\begin{frame}{Project Artifacts, Codebase \& Reproducibility}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Modular Repository Architecture (\texttt{AlgoDistill})}}
        \begin{itemize}\itemsep=2pt\scriptsize
          \item \texttt{src/data/}: TACO ingestion \& fractional partitioner.
          \item \texttt{src/filter/}: Semantic rationale-consistency validator.
          \item \texttt{src/judge/}: POSIX subprocess execution sandbox.
          \item \texttt{src/models/}: 4-bit QLoRA classifiers \& generators.
          \item \texttt{src/pipeline/}: End-to-end chained \& self-repair engines.
          \item \texttt{src/eval/}: Combinatorial Pass@k \& failure taxonomy.
        \end{itemize}
      \end{block}
      \vspace{2pt}
      \begin{exampleblock}{\textbf{Automated Test Suite}}
        \scriptsize
        \textbf{16 Unit \& Integration Tests Passing} (\texttt{pytest tests/ -v}) covering scraper, filter, sandbox POSIX limits, adapters, and chained pipelines.
      \end{exampleblock}
    \end{column}

    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Complete Google Colab Reproducibility}}
        \begin{itemize}\itemsep=2pt\scriptsize
          \item \texttt{final\_evaluation\_colab.ipynb}: Self-contained end-to-end evaluation and plotting suite.
          \item \texttt{phase6\_self\_repair\_colab.ipynb}: Live 1.5B multi-turn execution self-repair benchmark.
          \item \texttt{phase5\_valley\_mapping\_colab.ipynb}: 8-run empirical valley sweep reproduction.
          \item Checkpoints stored in \texttt{artifacts/checkpoints/} ($73.9\,\text{MB}$ LoRA adapters).
        \end{itemize}
      \end{block}
      \vspace{2pt}
      \begin{exampleblock}{\textbf{IEEE Access Standard Research Report}}
        \scriptsize
        Comprehensive 13-page technical manuscript typeset in \texttt{ieeeaccess.cls} with complete derivations, algorithms, and figures.
      \end{exampleblock}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 24: Conclusion, Limitations & Future Work
% -----------------------------------------------------------------------------
\begin{frame}{Conclusion, Limitations \& Future Directions}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \begin{block}{\textbf{Major Conclusions}}
        \begin{itemize}\itemsep=2pt\scriptsize
          \item The ``Valley of Code Reasoning'' in SLMs is driven by token entanglement and intermediate syntax collapse ($89.7\%$ CE reduction).
          \item Decoupled System-2 conditioning completely eliminates the valley, achieving strictly monotonic scaling and $+128.6\%$ trough gains.
          \item Algorithmic tags prevent strategy drift during multi-turn repair, unlocking a $4\times$ Pass@1 gain on Qwen2.5-Coder-1.5B.
        \end{itemize}
      \end{block}
      \vspace{2pt}
      \begin{alertblock}{\textbf{Current Limitations}}
        \begin{itemize}\itemsep=2pt\scriptsize
          \item Assumes single dominant algorithmic category (composite problems require multi-label tagging).
          \item Sequential two-stage inference introduces a modest latency overhead ($58.0\text{s}$ per problem).
        \end{itemize}
      \end{alertblock}
    \end{column}

    \begin{column}{0.48\textwidth}
      \begin{exampleblock}{\textbf{Future Research Directions}}
        \begin{enumerate}\itemsep=3pt\scriptsize
          \item \textbf{Hierarchical Multi-Label Tagging}: Conditioning on composite paradigms (e.g. \texttt{[GRAPHS + BINARY SEARCH]}).
          \item \textbf{Execution-Guided Reinforcement Learning}: Integrating sandbox unit test verdicts directly into Group Relative Policy Optimization (GRPO) with binary execution rewards ($R=+1$ for AC, $R=-0.5$ for CE/RE).
          \item \textbf{Dynamic Test Case Fuzzing}: Augmenting sandbox with property-based testing to uncover corner-case logic errors.
          \item \textbf{Polyglot Expansion}: Extending decoupled conditioning to compiled languages (C++, Rust).
        \end{enumerate}
      \end{exampleblock}
    \end{column}
  \end{columns}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 25: References & Key Literature
% -----------------------------------------------------------------------------
\begin{frame}{References \& Key Literature}
  \begin{thebibliography}{10}
    \tiny
    \bibitem{he2025valley}
    Z.~He, C.~Shen, X.~Shen, X.~Cui, Y.~Cai, H.~Gu, and P.~Bhat,
    \newblock ``The Valley of Code Reasoning: On the Scaling Dynamics of Rationale-Augmented Distillation,''
    \newblock \textit{arXiv:2502.04321}, 2025.

    \bibitem{dettmers2024qlora}
    T.~Dettmers, A.~Pagnoni, A.~Holtzman, and L.~Zettlemoyer,
    \newblock ``QLoRA: Efficient Finetuning of Quantized LLMs,''
    \newblock in \textit{Proc. NeurIPS}, 2023.

    \bibitem{li2023taco}
    R.~Li, J.~Zhu, L.~Gao, Y.~Zhang, and B.~Cui,
    \newblock ``TACO: Topics in Algorithmic Code Dataset for Benchmarking Large Language Models,''
    \newblock in \textit{Proc. NeurIPS}, 2023.

    \bibitem{qwen2024coder}
    Qwen~Team,
    \newblock ``Qwen2.5-Coder: Technical Report on Code Intelligence at Scale,''
    \newblock \textit{Alibaba Cloud Technical Report}, 2024.

    \bibitem{chen2021humaneval}
    M.~Chen et~al.,
    \newblock ``Evaluating Large Language Models Trained on Code,''
    \newblock \textit{arXiv:2107.03374}, 2021.

    \bibitem{shinn2023reflexion}
    N.~Shinn, F.~Cassin, A.~Gopinath, K.~Narasimhan, and S.~Yao,
    \newblock ``Reflexion: Language Agents with Verbal Reinforcement Learning,''
    \newblock in \textit{Proc. NeurIPS}, 2023.
  \end{thebibliography}
\end{frame}

% -----------------------------------------------------------------------------
% Slide 26: Thank You / Q&A Slide
% -----------------------------------------------------------------------------
\begin{frame}[plain]
  \begin{center}
    \vspace{0.8cm}
    \includegraphics[height=1.4cm]{nitk_logo.png} \\
    \vspace{0.4cm}
    {\Huge \textbf{\textcolor{NavyBlue}{Thank You!}}} \\
    \vspace{0.3cm}
    {\large \textbf{Questions \& Discussion}} \\
    \vspace{0.5cm}
    \normalsize
    \textbf{Team 9}: Pranav Bhat P, Shivam Kumar, Himanshu Mehar \\
    \textit{Department of Information Technology, NITK Surathkal} \\
    \vspace{0.4cm}
    \footnotesize
    \textbf{GitHub Repository}: \url{https://github.com/PranavBhatP/AlgoDistill}
  \end{center}
\end{frame}

\end{document}
'''

with open('docs/final_presentation.tex', 'w') as f:
    f.write(presentation_content)

print("Updated docs/final_presentation.tex with perfected spacing.")
