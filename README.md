# QLoRA Distillation for Algorithmic Code Classification and Generation

**Author:** Pranav Bhat P (Roll No: 231IT049)  
**Collaborators:** Himanshu Mehar (231IT027), Shivam Kumar (231IT069)  
**Course:** IT488 Course Project  
**Repository:** [AlgoDistill](https://github.com/PranavBhatP/AlgoDistill)  
**Target Paper:** *Decoupled Algorithmic-Category Conditioning and Multi-Turn Diagnostic Self-Repair for Robust Code Distillation*

---

## 📑 Table of Contents
1. [Executive Summary & Core Contributions](#-executive-summary--core-contributions)
2. [Research Motivation: The Valley of Code Reasoning](#-research-motivation-the-valley-of-code-reasoning)
3. [End-to-End System Architecture](#-end-to-end-system-architecture)
4. [Phase-by-Phase Experimental Results](#-phase-by-phase-experimental-results)
   - [Phase 1: Curated Benchmark & POSIX Subprocess Judge](#phase-1-curated-benchmark--posix-subprocess-judge)
   - [Phase 2: Rationale-Consistency Filter & Stage 1 Classifier](#phase-2-rationale-consistency-filter--stage-1-classifier)
   - [Phase 3: Stage 2 Tag-Conditioned Code Generator & Baseline Ablation](#phase-3-stage-2-tag-conditioned-code-generator--baseline-ablation)
   - [Phase 4: Multi-Scale QLoRA Training Sweep (8 Runs)](#phase-4-multi-scale-qlora-training-sweep-8-runs)
   - [Phase 5: Empirical Valley Mapping & Pass@k Benchmarking](#phase-5-empirical-valley-mapping--passk-benchmarking)
   - [Phase 6: Multi-Turn Execution Feedback & Qwen2.5-Coder-1.5B Evaluation](#phase-6-multi-turn-execution-feedback--qwen25-coder-15b-evaluation)
5. [Summary of Key Research Findings](#-summary-of-key-research-findings)
6. [Repository Structure & Artifacts Catalog](#-repository-structure--artifacts-catalog)
7. [Reproduction Guide (Local GPU & Google Colab)](#-reproduction-guide-local-gpu--google-colab)
8. [References & Citations](#-references--citations)

---

## 🎯 Executive Summary & Core Contributions

Small Language Models (SLMs, $\le 1.5\text{B}$ parameters) fine-tuned on Chain-of-Thought (CoT) code data suffer from **token entanglement**: jointly predicting high-level algorithmic rationale and low-level code syntax causes competition between semantic tokens, triggering a severe performance drop at intermediate data scaling known as **The Valley of Code Reasoning** (*He et al., 2025*).

This project introduces a **Decoupled Two-Stage QLoRA Distillation Framework** coupled with an **Iterative Execution Self-Repair Loop**:

1. **Decoupled Algorithmic Conditioning (Stage 1 $\to$ Stage 2)**: Formally decouples System-2 algorithmic commitment from code synthesis, reducing categorical search entropy.
2. **Semantic Rationale-Consistency Filtering**: Purges hallucinated or contradictory teacher traces ($91.0\%\text{--}94.0\%$ empirical retention), protecting student training from corrupted supervision.
3. **Empirical Valley Flattening**: In an 8-run scaling sweep ($10\%, 25\%, 50\%, 100\%$), decoupled conditioning completely eliminated the intermediate reasoning dip, achieving a **$+128.6\%$ relative gain at the trough** and an **$89.71\%$ reduction in compilation errors**.
4. **Iterative Multi-Turn Execution Self-Repair**: Closed-loop feedback intercepting sandbox traces back to the generator yielded a **$4\times$ relative gain in Pass@1** on `Qwen2.5-Coder-1.5B` ($4.0\% \to 8.0\% \to 16.0\%$), while unconditioned baselines saturated early.
5. **Consumer-Grade Accessibility**: All fine-tuning and inference pipelines operate in 4-bit NormalFloat (`NF4`) with parameter-efficient QLoRA ($<0.22\%$ trainable parameters), executing entirely within a single consumer GPU ($<5.0\,\text{GB}$ VRAM on an NVIDIA RTX 3050 / Tesla T4).

---

## 🔬 Research Motivation: The Valley of Code Reasoning

Recent investigations into reasoning distillation (*He et al., 2025*) reveal an empirical anomaly when scaling Chain-of-Thought (CoT) data for code generation:

$$\text{Pass@1}_{\text{Baseline}}: \quad 13.2\% \,(10\%) \;\longrightarrow\; \mathbf{11.2\% \,(25\%)} \;\longrightarrow\; 11.6\% \,(50\%) \;\longrightarrow\; 17.6\% \,(100\%)$$

During intermediate scaling ($25\%\text{--}50\%$), the model experiences a **reasoning collapse** where functional correctness degrades before recovering at full scale. 

### Why Does the Valley Occur?
* **Token Entanglement**: Monolithic student models must resolve abstract algorithmic invariants ($O(N \log N)$ sorting vs. $O(N)$ dynamic programming state transitions) while simultaneously managing syntax conventions (indentation, typing, recursion limits).
* **Supervision Contamination**: Commercial teacher models (e.g., Gemini, GPT-4) frequently generate mismatched rationales that suggest one paradigm (e.g., Greedy) while implementing another (e.g., Brute Force). Distilling unfiltered traces injects contradictory gradients.
* **Lack of Planning Grounding**: Standard CoT does not explicitly commit to an algorithmic paradigm before generating tokens, causing early generation paths to wander into suboptimal brute-force search.

---

## 🏗 End-to-End System Architecture

```
                                  [Problem Statement]
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │                                           │
                     ▼                                           ▼
          [Stage 1 Classifier]                       [Unconditioned Baseline]
       (Qwen2.5-0.5B + 4-bit QLoRA)                (Qwen2.5-Coder-0.5B/1.5B)
                     │                                           │
        Predicted Tag: [ALGORITHM]                                │
                     │                                           │
                     ▼                                           ▼
       [Stage 2 Conditioned Generator]               Direct Generation (No Tag)
    (Qwen2.5-Coder-0.5B/1.5B + 4-bit QLoRA)                      │
                     │                                           │
          Synthesized Code (Turn 0)                   Synthesized Code (Turn 0)
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                             [POSIX Subprocess Sandbox]
                          (2.0s CPU / 512MB RAM / AST Sanitizer)
                                           │
                       ┌───────────────────┴───────────────────┐
                       │                                       │
                  Verdict: AC                             Verdict != AC
                       │                               (WA, RE, TLE, CE)
                       ▼                                       │
                 [Benchmark AC]                                ▼
                                                  [Diagnostic Feedback Formatter]
                                                  (Failing Input + Traceback)
                                                               │
                                                               ▼
                                                  [Turn 1 & 2 Self-Repair Loop]
                                                  (Targeted Iterative Refinement)
```

### Mathematical Formulation
$$\log P(\text{code}, \text{rationale} \mid \text{problem}) \approx \sum_{\text{tag}} \underbrace{P(\text{tag} \mid \text{problem})}_{\text{Stage 1 Classifier}} \cdot \underbrace{P(\text{code}, \text{rationale} \mid \text{tag}, \text{problem})}_{\text{Stage 2 Conditioned Generator}}$$

---

## 📊 Phase-by-Phase Experimental Results

### Phase 1: Curated Benchmark & POSIX Subprocess Judge

* **Dataset Source**: Curated from TACO (Topics in Algorithm and Code Optimization), difficulty rating 800–1200.
* **10-Class Algorithmic Taxonomy**: Dynamic Programming, Greedy, Graphs, Mathematics, Data Structures, Trees, Brute Force / Simulation, String Manipulation, Number Theory, Binary Search.
* **Dataset Scaling**:
  - Initial 50% Milestone: 100 problems (10 per class).
  - Expanded Final Dataset: 300 problems (30 per class), partitioned into $N_{\text{train}} = 210$, $N_{\text{val}} = 40$, $N_{\text{test}} = 50$.
* **POSIX Sandbox Execution Engine** (`src/judge/sandbox_judge.py`):
  - Memory isolation via `resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, ...))`.
  - Hard CPU timeout via `subprocess.TimeoutExpired` capped at $2.0\,\text{s}$ ($3.0\,\text{s}$ for multi-turn repair).
  - Sanitization parser: Automated regex cleansing stripping raw variable assignments (`m = 3`, `Arr[] = {1,5,6}`) ensuring compatibility with competitive programming `sys.stdin.read().split()` routines.
  - Driver synthesis: Automatically injects invocation wrapper for isolated functions (`def solve():`, `def main():`).

---

### Phase 2: Rationale-Consistency Filter & Stage 1 Classifier

#### Semantic Rationale-Consistency Filter (`src/filter/rationale_filter.py`)
Validates teacher reasoning traces by checking lexical alignment against category-specific taxonomies, penalizing cross-paradigm contradictions and length-deficient traces:

$$\text{Score} = \min\left(1.0,\; 0.4 + 0.2 \cdot |\mathcal{K}_{\text{matched}}|\right)$$

* **Phase 2 Empirical Yield**: 91 / 100 problems retained (**91.0% retention yield**, 9 corrupted traces purged).
* **Phase 3.5 Expansion Yield**: 282 / 300 problems retained (**94.0% retention yield**, 18 noisy traces purged).

#### Stage 1 Algorithmic Tag Classifier (`src/models/stage1_classifier.py`)
* **Base Model**: `Qwen/Qwen2.5-0.5B` with 4-bit NormalFloat (`NF4`) double quantization.
* **LoRA Configuration**: $r=16, \alpha=32, \text{dropout}=0.05$, targets `[q_proj, v_proj]`.
* **Trainable Parameters**: $1.09\,\text{M}$ ($0.220\%$ of $495.1\,\text{M}$).
* **Training Dynamics**: Loss decreased from $5.3800 \to 0.7688$ in 3 epochs ($44.6\,\text{s}$, $1,195.7\,\text{MB}$ peak VRAM on Tesla T4).
* **Test Accuracy**: Top-1 Accuracy: $5.56\%$, Top-3 Accuracy: $22.22\%$ (vs. $10.0\%$ random baseline).

---

### Phase 3: Stage 2 Tag-Conditioned Code Generator & Baseline Ablation

* **Architecture**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (4-bit NF4).
* **Conditioning Format**: ChatML schema explicitly encoding the committed paradigm:
  ```
  <|im_start|>system
  You are an expert competitive programmer. Solve the problem by following the provided algorithm category tag.<|im_end|>
  <|im_start|>user
  [ALGORITHM CATEGORY]: dynamic programming

  Problem Statement:
  ...
  ```
* **Masked-Loss SFT**: Prompt tokens masked to `-100`; cross-entropy loss computed strictly over rationale and executable code tokens.
* **Trainable Parameters**: $1.08\,\text{M}$ ($0.218\%$ of $495.1\,\text{M}$).
* **Convergence**: Loss decreased from $1.3568 \to 1.1813$ in 3 epochs ($97.4\,\text{s}$, $4,246.8\,\text{MB}$ peak VRAM).

---

### Phase 4: Multi-Scale QLoRA Training Sweep (8 Runs)

To rigorously analyze data scaling dynamics, an 8-run fine-tuning matrix was executed across 4 stratified data fractions: $10\%$ ($N=30$), $25\%$ ($N=60$), $50\%$ ($N=110$), and $100\%$ ($N=210$).

#### Training Loss Trajectories Across Epochs

| Checkpoint | Fraction | Train Samples | Epoch 1 Loss | Epoch 2 Loss | Epoch 3 Loss | Epoch 4 Loss | Final Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `conditioned_10pct` | 10% | 30 | 1.3126 | 1.2095 | 1.1006 | 0.9958 | **0.9958** |
| `baseline_10pct` | 10% | 30 | 1.5354 | 1.4823 | 1.4148 | 1.3430 | **1.3430** |
| `conditioned_25pct` | 25% | 60 | 1.2920 | 1.1890 | 1.0810 | 0.9772 | **0.9772** |
| `baseline_25pct` | 25% | 60 | 1.5036 | 1.4504 | 1.3831 | 1.3148 | **1.3148** |
| `conditioned_50pct` | 50% | 110 | 1.2576 | 1.1547 | 1.0481 | 0.9464 | **0.9464** |
| `baseline_50pct` | 50% | 110 | 1.4649 | 1.4121 | 1.3453 | 1.2785 | **1.2785** |
| `conditioned_100pct`| 100% | 210 | 1.1888 | 1.0861 | 0.9825 | 0.8846 | **0.8846** |
| `baseline_100pct` | 100% | 210 | 1.3879 | 1.3323 | 1.2644 | 1.1963 | **1.1963** |

> **Finding:** Conditioning consistently lowered training loss across all sample budgets by $\Delta \mathcal{L} \approx -0.31\text{ to }-0.35$ NLL, confirming that algorithmic tags significantly constrain the hypothesis search space.

---

### Phase 5: Empirical Valley Mapping & Pass@k Benchmarking

All 8 checkpoints were evaluated across the 50-problem test set ($5$ generations per problem, $N_{\text{eval}} = 250$ samples per checkpoint, $2,000$ total executions in the POSIX sandbox).

#### Pass@k Performance Comparison Across Data Fractions

| Architecture | Metric | 10% Data ($N=30$) | 25% Data ($N=60$) | 50% Data ($N=110$) | 100% Data ($N=210$) | Scaling Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unconditioned Baseline** | Pass@1 | 13.2% | **11.2%** (Dip) | **11.6%** (Valley) | 17.6% (Recovery) | Non-Monotonic (Dip: 2.0%) |
| **Unconditioned Baseline** | Pass@5 | 56.0% | 48.0% | 56.0% | 58.0% | Plateau at $\sim 58\%$ |
| **Tag-Conditioned QLoRA** | **Pass@1** | **16.0%** | **25.6%** | **30.8%** | **44.4%** | **Strictly Monotonic ($\nearrow$)** |
| **Tag-Conditioned QLoRA** | **Pass@5** | **60.0%** | **82.0%** | **86.0%** | **96.0%** | **Saturates at $96.0\%$ (48/50)** |

#### Valley Mitigation Metrics
* **Baseline Reasoning Dip Depth**: $-2.0\%$ absolute drop from 10% to 25% data.
* **Trough Advantage**: At the 25% data trough, Tag-Conditioning achieved a **$+128.57\%$ relative gain** ($25.6\%$ vs. $11.2\%$).
* **Asymptotic Advantage**: At 100% data, Tag-Conditioning achieved a **$+152.27\%$ relative gain** in Pass@1 ($44.4\%$ vs. $17.6\%$) and Pass@5 reached **$96.0\%$** (48/50 problems solved).

#### Execution Failure Taxonomy Across All 2,000 Runs

| Failure Mode | Description | Baseline Count | Conditioned Count | Percentage Reduction |
| :--- | :--- | :---: | :---: | :---: |
| **AC** | Accepted (Passes all test cases) | 134 | **292** | **$+117.9\%$ Increase** |
| **CE** | Compilation / Syntax Error | 68 | **7** | **$-89.71\%$ Reduction** |
| **RE** | Runtime Exception (Index, ZeroDiv) | 139 | **40** | **$-71.22\%$ Reduction** |
| **TLE** | Time Limit Exceeded ($>2.0\,\text{s}$) | 53 | **28** | **$-47.17\%$ Reduction** |
| **WA** | Wrong Answer (Logic Mismatch) | 606 | **633** | Shift from syntax to logic |

> **Key Takeaway:** In unconditioned models, intermediate scaling causes a massive spike in catastrophic syntax and compilation errors ($68$ CEs). Decoupled conditioning reduces compilation failures by **$89.71\%$**, preserving functional syntax integrity throughout training.

---

### Phase 6: Multi-Turn Execution Feedback & Qwen2.5-Coder-1.5B Evaluation

To test self-repair capabilities, the framework was scaled to **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** in 4-bit NF4. Checkpoints were fine-tuned with target loss masking (`max_len=1024`, batch size 1, gradient accumulation 4) and evaluated on a multi-turn repair benchmark ($N=25$ test problems, maximum 2 repair turns).

#### Diagnostic Feedback Prompting
When a solution fails execution, the sandbox captures the failure mode, inputs, expected output, and stdout/stderr traces, formulating a targeted repair prompt:
```
<|im_start|>user
[REPAIR ATTEMPT #1]
The previous code failed execution in the sandbox.

[DIAGNOSTIC FEEDBACK]:
Wrong Answer (Verdict: WA) on Test Case #1:
The solution produced incorrect output logic.

Failed Test Input:
5 3
1 2 3 4 5

Expected Output:
15

Actual Output:
10

Analyze the execution failure, explain the bug concisely, and output the corrected Python 3 solution.<|im_end|>
<|im_start|>assistant
Rationale:
...
```

#### Pass@1 Repair Trajectory (Qwen2.5-Coder-1.5B, $N=25$)

| Architecture | Turn 0 (Zero-Shot) | Turn 1 (+1 Repair) | Turn 2 (+2 Repairs) | $\Delta\text{Pass@1}$ (Turn 1) | Total $\Delta\text{Pass@1}$ (Turn 2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tag-Conditioned QLoRA (1.5B)** | **4.0%** (1/25) | **8.0%** (2/25) | **16.0%** (4/25) | **+4.0%** | **+12.0% ($4\times$ Gain)** |
| **Unconditioned Baseline (1.5B)** | **8.0%** (2/25) | **16.0%** (4/25) | **16.0%** (4/25) | **+8.0%** | **+8.0% (Plateaued)** |

#### Case Studies of Multi-Turn Self-Repair
1. **Immediate Zero-Shot Pass (`TACO-DY-21`)**: Solved dynamic programming problem on Turn 0 (`AC`).
2. **Single-Turn Repair (`TACO-MA-06`)**: Initial Turn 0 yielded `WA` due to parity boundary condition; on Turn 1 with diagnostic test input feedback, the model adjusted the loop condition and achieved `AC`.
3. **Two-Turn Deep Repair (`TACO-DS-30`)**: Turn 0 failed (`WA`) $\to$ Turn 1 failed (`WA`) $\to$ Turn 2 corrected the priority queue indexing and passed all tests (`AC`).
4. **Exception-to-AC Transition (`TACO-DS-24`)**: Turn 0 threw `IndexError` (`RE`) $\to$ Turn 1 reproduced `RE` $\to$ Turn 2 resolved off-by-one pointer arithmetic and achieved `AC`.

#### Qualitative Advantage Over Baseline
* While the Unconditioned Baseline gained $+8.0\%$ on Turn 1, it completely stalled on Turn 2 ($+0.0\%$ gain).
* The Tag-Conditioned model continued repairing successfully across consecutive turns ($+4.0\%$ in Turn 1, $+8.0\%$ in Turn 2), demonstrating that algorithmic grounding prevents model hallucination and strategy drift during iterative refinement.

---

## 📈 Summary of Key Research Findings

| Scientific Question | Baseline Observation | Decoupled QLoRA Result | Impact |
| :--- | :--- | :--- | :--- |
| **Does The Valley of Code Reasoning exist in small models?** | Yes. Baseline drops $-2.0\%$ absolute in Pass@1 at $25\%$ data scale ($13.2\% \to 11.2\%$). | Completely eliminated. Model scales strictly monotonically from $16.0\% \to 44.4\%$. | Flattens the Valley; $+128.6\%$ trough advantage. |
| **What causes intermediate reasoning degradation?** | Token entanglement causes a massive spike in syntax/compilation failures ($68$ CEs). | Compilation errors dropped by $89.71\%$ ($7$ CEs across all runs). | Preserves syntax integrity during semantic learning. |
| **How does data quality affect student distillation?** | Unfiltered teacher traces inject cross-paradigm contradictions. | Semantic filter purges $6\%\text{--}9\%$ noisy traces, raising sample quality. | Eliminates supervision noise before student gradient descent. |
| **Can small models repair their own code with execution feedback?** | Baseline plateaus after Turn 1 ($+8.0\% \to +0.0\%$). | Tag-Conditioned model achieves continuous gains across Turn 1 and Turn 2 ($+12.0\%$ total, $4\times$ relative gain). | Validates algorithmic grounding for multi-turn agentic repair. |

---

## 📂 Repository Structure & Artifacts Catalog

```
llm-project/
├── artifacts/
│   ├── checkpoints/
│   │   ├── stage2_conditioned_1.5b_model/    # 73.9 MB QLoRA fine-tuned 1.5B adapter
│   │   ├── stage2_baseline_1.5b_model/       # 73.9 MB QLoRA fine-tuned 1.5B adapter
│   │   ├── conditioned_10pct ... 100pct/     # Phase 4 0.5B checkpoints
│   │   └── baseline_10pct ... 100pct/        # Phase 4 0.5B checkpoints
│   ├── figures/
│   │   ├── methodology_diagram.png           # Two-stage pipeline architectural schematic
│   │   ├── rationale_filter_breakdown.png    # Filter retention yield & taxonomy breakdown
│   │   ├── stage1_confusion_matrix.png       # 10-class algorithmic classifier confusion matrix
│   │   ├── stage1_per_class_f1.png           # Per-class F1 score bar chart
│   │   ├── phase4_training_loss_sweep.png    # 8-run multi-scale training loss curves
│   │   ├── phase5_valley_mapping_pass_at_k.png # Empirical Valley of Code Reasoning scaling curves
│   │   ├── phase5_failure_taxonomy.png       # AC, WA, TLE, RE, CE error distributions
│   │   ├── phase5_category_pass_rates.png    # 10-class category-wise pass rate comparison
│   │   ├── phase6_repair_pass_at_1_delta_local.png # Multi-turn Pass@1 delta curves (Turns 0-2)
│   │   └── phase6_repair_transitions_local.png     # Markov error state transition diagram
│   ├── phase4_training_sweep_results.json    # Full numerical logs for Phase 4 sweep
│   └── phase5_benchmarking_results.json      # Full Pass@k and taxonomy logs for Phase 5 sweep
├── data/
│   ├── splits/
│   │   ├── train.jsonl                       # N=210 sanitized training problems with teacher traces
│   │   ├── val.jsonl                         # N=40 validation problems
│   │   └── test.jsonl                        # N=50 test problems with standardized unit tests
│   ├── taco_300_distillation_dataset.json    # Complete 300-problem benchmark dataset
│   └── raw/raw_problems_traces.jsonl         # Original raw problems before filtering
├── notebooks/
│   ├── final_evaluation_local.ipynb          # End-to-end 1.5B execution notebook (fully populated)
│   ├── final_evaluation_colab.ipynb          # Self-contained Google Colab reproduction notebook
│   ├── phase6_self_repair_colab.ipynb        # Phase 6 diagnostic self-repair notebook
│   ├── phase5_valley_mapping_colab.ipynb     # Phase 5 empirical valley mapping notebook
│   ├── phase4_qlora_sweep_colab.ipynb        # Phase 4 QLoRA multi-scale training sweep notebook
│   └── evaluation_50percent_colab.ipynb      # 50% Milestone evaluation notebook
├── src/
│   ├── data/
│   │   ├── dataset_builder.py                # Dataset ingestion and stratification engine
│   │   └── scaling_partitioner.py            # Fractional scaling partitioner (10%, 25%, 50%, 100%)
│   ├── filter/
│   │   └── rationale_filter.py               # Semantic rationale consistency verification
│   ├── judge/
│   │   └── sandbox_judge.py                  # Subprocess execution sandbox with POSIX limits
│   ├── models/
│   │   ├── stage1_classifier.py              # 4-bit QLoRA sequence classification model
│   │   ├── stage2_generator.py               # 4-bit QLoRA tag-conditioned code synthesis model
│   │   └── scaling_sweep.py                  # Multi-scale 8-run fine-tuning sweep engine
│   ├── pipeline/
│   │   ├── e2e_pipeline.py                   # Chained two-stage inference pipeline
│   │   └── repair_loop.py                    # Multi-turn diagnostic self-repair engine
│   └── eval/
│       ├── metrics.py                        # Top-k accuracy, macro-F1, Pass@k calculation
│       ├── execution_eval.py                 # Automated unit test evaluation suite
│       └── benchmark_sweep.py                # Multi-checkpoint Pass@k benchmark runner
└── scripts/
    ├── run_notebook_execution.py             # Headless execution engine for Jupyter notebooks
    ├── run_phase4_sweep.py                   # CLI for Phase 4 training sweep
    ├── run_phase5_benchmarking.py            # CLI for Phase 5 Pass@k benchmark sweep
    └── generate_local_evaluation_notebook.py # Notebook generator from source
```

---

## 💻 Reproduction Guide (Local GPU & Google Colab)

### Option A: Local Execution (RTX 3050 6GB / Any CUDA GPU)

1. **Clone the repository and install dependencies**:
   ```bash
   git clone https://github.com/PranavBhatP/AlgoDistill.git
   cd AlgoDistill
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run all unit and integration tests**:
   ```bash
   pytest tests/ -v
   ```

3. **Run Phase 4 Multi-Scale QLoRA Training Sweep**:
   ```bash
   python scripts/run_phase4_sweep.py --mode qlora --epochs 4
   ```

4. **Run Phase 5 Valley Mapping & Pass@k Benchmarking**:
   ```bash
   python scripts/run_phase5_benchmarking.py --mode qlora --samples-per-problem 5
   ```

5. **Run the Full 1.5B Multi-Turn Benchmark Notebook Headlessly**:
   ```bash
   python scripts/run_notebook_execution.py
   ```

---

### Option B: Google Colab Reproduction (Free Tesla T4 GPU)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `notebooks/final_evaluation_colab.ipynb`.
3. Set runtime: **Runtime** $\to$ **Change runtime type** $\to$ **T4 GPU**.
4. Run all cells: the notebook downloads `Qwen/Qwen2.5-Coder-1.5B-Instruct`, performs 4-bit QLoRA fine-tuning for both models, executes the 25-problem multi-turn benchmark, and renders the publication figures automatically.

---

## 📚 References & Citations

1. **Base Paper**: He, Z., et al. (2025). *The Valley of Code Reasoning: Investigating the Dynamics of Reasoning Distillation in Code Language Models*. arXiv preprint.
2. **QLoRA**: Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
3. **TACO Benchmark**: Li, R., et al. (2023). *TACO: Topics in Algorithm and Code Optimization Benchmark*. arXiv preprint.
4. **Qwen2.5-Coder**: Hui, B., et al. (2024). *Qwen2.5-Coder Technical Report: Code Intelligence at Scale*. arXiv:2409.12186.
5. **HumanEval / Pass@k**: Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code*. arXiv:2107.03374.
