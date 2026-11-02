#!/usr/bin/env python3
"""CLI Runner for Phase 4: Multi-Scale QLoRA Training Sweep (8 Runs).

Executes the empirical distillation sweep across 4 training data scale fractions
(10%, 25%, 50%, 100%) for both the Tag-Conditioned Generator and Unconditioned
Baseline on Qwen2.5-Coder-0.5B using 4-bit NormalFloat (NF4) quantization.
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
from src.models.scaling_sweep import ScalingSweepManager, SweepConfig
from src.utils.visualization import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("phase4_sweep")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: Multi-Scale QLoRA Training Sweep (8 Runs)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "qlora", "fast"],
        default="auto",
        help="Execution mode: 'qlora' (full GPU training), 'fast' (fast calibrated profiling), or 'auto' (GPU if available)"
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.10,0.25,0.50,1.00",
        help="Comma-separated scale fractions (e.g., '0.10,0.25,0.50,1.00')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of fine-tuning epochs per run (default: 4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device (default: 2)"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.5e-4,
        help="Peak learning rate for AdamW (default: 1.5e-4)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B",
        help="Hugging Face base student model (default: Qwen/Qwen2.5-Coder-0.5B)"
    )
    parser.add_argument(
        "--splits-file",
        type=str,
        default="data/splits/train.jsonl",
        help="Path to training split JSONL (default: data/splits/train.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/checkpoints",
        help="Directory to save checkpoint weights/metadata"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="artifacts/phase4_training_sweep_results.json",
        help="Path to save full JSON results"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate comparison figure artifacts/figures/phase4_training_loss_sweep.png"
    )
    return parser.parse_args()


def load_or_create_train_split(splits_file: str) -> List[DatasetItem]:
    """Loads training split items or constructs them from TACO 300 dataset."""
    if os.path.exists(splits_file):
        items = DatasetBuilder.load_jsonl(splits_file)
        if items:
            logger.info("Loaded %d training items from %s", len(items), splits_file)
            return items

    logger.warning("Splits file not found at %s. Building from raw dataset...", splits_file)
    builder = DatasetBuilder()
    raw = builder.build_raw_dataset()
    train_split, _, _ = builder.create_stratified_splits(raw)
    return train_split


def main():
    args = parse_args()

    # Determine execution mode
    import torch
    if args.mode == "auto":
        mode = "qlora" if torch.cuda.is_available() else "fast"
    else:
        mode = args.mode

    fractions = [float(f.strip()) for f in args.fractions.split(",") if f.strip()]

    # Load dataset
    train_items = load_or_create_train_split(args.splits_file)

    # Print experiment banner
    print("\n" + "=" * 80)
    print("       ALGODISTILL: PHASE 4 MULTI-SCALE QLoRA TRAINING SWEEP (8 RUNS)")
    print("=" * 80)
    print(f"  Base Model       : {args.base_model}")
    print(f"  Execution Mode   : {mode.upper()} (device: {'cuda' if torch.cuda.is_available() else 'cpu'})")
    print(f"  Scale Fractions  : {[f'{int(f*100)}%' for f in fractions]}")
    print(f"  Total Pool Size  : {len(train_items)} training problems")
    print(f"  Epochs per Run   : {args.epochs}")
    print(f"  Batch Size       : {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"  Output Directory : {args.output_dir}")
    print(f"  Results JSON     : {args.results_path}")
    print("=" * 80 + "\n")

    config = SweepConfig(
        base_model_name=args.base_model,
        fractions=fractions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        mode=mode,
        output_dir=args.output_dir,
        results_path=args.results_path
    )

    manager = ScalingSweepManager(config)
    results = manager.run_sweep(train_items=train_items, fractions=fractions)

    # Generate publication plot
    if args.plot:
        vis = Visualizer(output_dir="artifacts/figures")
        plot_path = vis.plot_phase4_training_loss_sweep(
            sweep_results=results,
            filename="phase4_training_loss_sweep.png"
        )
        print(f"\n✓ Generated Phase 4 loss comparison figure: {plot_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Run ID':<22} | {'Type':<12} | {'Scale':<6} | {'Samples':<8} | {'Final Loss':<11} | {'Time (s)':<9} | {'Peak VRAM'}")
    print("-" * 80)
    for r in results["runs_list"]:
        vram_str = f"{r['peak_vram_mb']:.1f} MB" if r.get("peak_vram_mb", 0) > 0 else "N/A"
        print(f"{r['run_id']:<22} | {r['model_type']:<12} | {int(r['fraction']*100):>3}%   | {r['train_samples']:<8} | {r['final_loss']:<11.4f} | {r['training_time_s']:<9.2f} | {vram_str}")
    print("=" * 80)
    print(f"Total Sweep Duration: {results.get('total_sweep_time_s', 0):.2f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
