# QLoRA Distillation for Algorithmic Code Classification and Generation

**Author:** Pranav Bhat P (Roll No: 231IT049)  
**Course:** IT488 Course Project  
**Milestone:** Midsemester Evaluation (Phases 1 & 2 - 40% Scope)

---

## 🎯 Overview
This project implements the research proposed in **"QLoRA Distillation for Algorithmic Code Classification and Generation"**. The system addresses the limitations of *The Valley of Code Reasoning* (He et al., 2025) by introducing:
1. **Decoupled Algorithmic-Category Conditioning (Stage 1 Classifier)**: Predicts the underlying algorithm category from problem statements before synthesizing code.
2. **Rationale-Consistency Filtering**: Semantic validation step that eliminates hallucinatory or mismatched teacher reasoning traces before student supervision.
3. **Consumer-GPU QLoRA Pipeline**: 4-bit NormalFloat (NF4) quantization + LoRA adapters optimized for 6GB consumer GPUs and Google Colab / Kaggle T4 GPUs.
4. **Sandboxed Subprocess Execution Judge**: Enforces CPU timeouts (2.0s) and memory bounds (512MB) for automated competitive programming testing.

---

## 📁 Repository Structure
```
llm-project/
├── configs/
│   ├── stage1_classifier.yaml            # Stage 1 model & LoRA hyperparameter configuration
│   ├── stage2_generator.yaml             # Stage 2 conditioned generator & scaling sweep settings
│   └── hardware_profile.yaml             # GPU (6GB RTX 3050 / T4) and CPU profiles
├── data/
│   ├── raw/                              # Curated problems & teacher rationales
│   └── splits/                           # Category-stratified train.jsonl, val.jsonl, test.jsonl
├── notebooks/
│   └── midsemester_evaluation_colab.ipynb# Self-contained Google Colab Notebook (Run on T4 GPU)
├── scripts/
│   ├── run_midsem_demo.py                # Standalone CLI for Midsemester evaluation demo
│   └── generate_colab_notebook.py        # Re-generates Colab notebook from source
├── src/
│   ├── data/
│   │   ├── codeforces_scraper.py         # Codeforces scraper & 10-category problem curator
│   │   ├── teacher_generator.py          # Teacher rationale & code solution generator
│   │   └── dataset_builder.py            # Stratified dataset builder (Train/Val/Test)
│   ├── filter/
│   │   └── rationale_filter.py           # Semantic rationale-consistency filter
│   ├── judge/
│   │   └── sandbox_judge.py              # Isolated subprocess execution sandbox
│   ├── models/
│   │   └── stage1_classifier.py          # 4-bit QLoRA Stage 1 Algorithmic Tag Classifier
│   ├── eval/
│   │   └── metrics.py                    # Top-1 Acc, Top-3 Acc, Macro-F1, Confusion Matrix
│   └── utils/
│       ├── formatting.py                 # Prompt templates ([TAG] + [PROBLEM] -> [CODE])
│       └── visualization.py              # Matplotlib/Seaborn plot generators
├── tests/
│   ├── test_scraper.py                   # Tests for problem curator & taxonomy
│   ├── test_judge.py                     # Tests for sandbox judge (AC, WA, TLE, RE, CE)
│   ├── test_filter.py                    # Tests for rationale consistency filter
│   └── test_stage1.py                    # Tests for Stage 1 Classifier training & inference
├── artifacts/
│   └── figures/                          # Confusion matrices, F1 bar charts, filter yield
├── requirements.txt                      # Project dependencies
└── pyproject.toml                        # uv / pip configuration
```

---

## 🚀 Quick Start (Local Execution)

### 1. Run the End-to-End Midsemester Demo CLI
```bash
python scripts/run_midsem_demo.py
```
This executes:
- Phase 1.1: TACO 100-Problem Dataset loading across 10 algorithmic categories
- Phase 1.2: Sandboxed subprocess judge verification with timeout & memory bounds
- Phase 2.1: Rationale-consistency semantic filter (90% retention yield)
- Phase 2.2: Stage 1 Algorithmic Tag Classifier training & evaluation
- Phase 2.3: Live interactive problem inference on custom problem statements

### 2. Run Automated Unit and Integration Tests
```bash
pytest tests/ -v
```

---

## 🌐 Running on Google Colab (Recommended for GPU)
The self-contained notebook is available at:
`notebooks/midsemester_evaluation_colab.ipynb`

### How to Run on Colab:
1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload `notebooks/midsemester_evaluation_colab.ipynb`.
3. Select **Runtime** → **Change runtime type** → **T4 GPU**.
4. Run all cells to execute the full Phase 1 & 2 pipeline with GPU acceleration!
