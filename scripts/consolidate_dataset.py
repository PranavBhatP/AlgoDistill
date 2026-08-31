"""Consolidates the 100 TACO problems and their Gemini teacher reasoning traces
into a single clean, standalone JSON dataset file for Google Colab.
"""

import json
import os

taco_path = "data/taco_100_problems.json"
traces_path = "data/teacher_traces_100.json"
output_path = "data/taco_100_distillation_dataset.json"

with open(taco_path, "r", encoding="utf-8") as f:
    problems = json.load(f)

with open(traces_path, "r", encoding="utf-8") as f:
    traces = json.load(f)

trace_map = {t["problem_id"]: t for t in traces}

consolidated = []
for p in problems:
    t = trace_map.get(p["problem_id"], {})
    consolidated.append({
        "problem_id": p["problem_id"],
        "title": p["title"],
        "ground_truth_tag": p["ground_truth_tag"],
        "rating": p.get("rating", 1000),
        "statement": p["statement"],
        "input_spec": p.get("input_spec", "Standard competitive programming input format."),
        "output_spec": p.get("output_spec", "Standard competitive programming output format."),
        "teacher_rationale": t.get("rationale", ""),
        "teacher_solution": t.get("solution_code", p.get("raw_solution", "")),
        "teacher_stated_tag": t.get("stated_algorithmic_strategy", p["ground_truth_tag"]),
        "sample_tests": p.get("sample_tests", []),
        "time_limit_ms": p.get("time_limit_ms", 2000),
        "memory_limit_mb": p.get("memory_limit_mb", 512),
        "is_consistency_verified": False
    })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(consolidated, f, indent=2)

print(f"Consolidated {len(consolidated)} problems & teacher traces into {output_path}")
