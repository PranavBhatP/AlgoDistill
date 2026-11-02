#!/usr/bin/env python3
"""Executes notebooks/final_evaluation_local.ipynb cell by cell with live logging and incremental checkpointing.
"""

import os
import sys
import time
import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = "notebooks/final_evaluation_local.ipynb"

def main():
    print("=" * 75)
    print("  LAUNCHING FULL NOTEBOOK EXECUTION ENGINE: Qwen2.5-Coder-1.5B")
    print(f"  Notebook Target: {NOTEBOOK_PATH}")
    print("=" * 75)

    if not os.path.exists(NOTEBOOK_PATH):
        print(f"ERROR: Notebook {NOTEBOOK_PATH} not found!")
        sys.exit(1)

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    print(f"✓ Loaded notebook with {len(nb.cells)} total cells.")

    client = NotebookClient(
        nb,
        timeout=7200,  # 2 hours per cell timeout
        kernel_name="python3",
        allow_errors=False
    )

    t_start_total = time.time()
    print("✓ Initializing Jupyter kernel...")
    with client.setup_kernel():
        for idx, cell in enumerate(nb.cells):
            cell_type = cell.get("cell_type")
            if cell_type != "code":
                continue

            src_snippet = "".join(cell.get("source", []))[:80].replace("\n", " ")
            print(f"\n[{idx:02d}/{len(nb.cells)-1:02d}] Executing Cell {idx}: {src_snippet}...")
            t0 = time.time()

            try:
                client.execute_cell(cell, idx)
                elapsed = time.time() - t0
                print(f"✓ Cell {idx} completed successfully in {elapsed:.1f}s")
                
                # Print any stream text output from the cell
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "stream":
                        print("  " + "".join(out.get("text", [])).strip()[:400])
                    elif out.get("output_type") == "error":
                        print("  [ERROR]:", out.get("ename"), out.get("evalue"))

                # Incremental write-back to ensure progress is saved on disk
                with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
                    nbformat.write(nb, f)

            except Exception as e:
                elapsed = time.time() - t0
                print(f"❌ Error in Cell {idx} after {elapsed:.1f}s: {e}")
                # Save state even on error
                with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
                    nbformat.write(nb, f)
                raise

    total_time = time.time() - t_start_total
    print("\n" + "=" * 75)
    print(f"✓ ALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY IN {total_time/60:.2f} MINUTES!")
    print(f"✓ Final populated notebook saved to: {NOTEBOOK_PATH}")
    print("=" * 75)

if __name__ == "__main__":
    main()
