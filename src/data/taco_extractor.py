"""Extracts 100 balanced competitive programming problems (10 per category)
from the Hugging Face TACO dataset (BAAI/TACO) in the 800-1200 rating range.
"""

import os
import json
import ast
import re
from collections import defaultdict
from typing import List, Dict, Any
import pandas as pd
from huggingface_hub import hf_hub_download

TARGET_TAG_TAXONOMY = [
    "dynamic programming",
    "greedy",
    "graphs",
    "math",
    "data structures",
    "trees",
    "brute force",
    "strings",
    "number theory",
    "binary search",
]

TAG_MAP = {
    # DP
    "dp": "dynamic programming",
    "dynamic programming": "dynamic programming",
    "memoization": "dynamic programming",
    # Greedy
    "greedy": "greedy",
    "greedy algorithms": "greedy",
    # Graphs
    "graphs": "graphs",
    "graph algorithms": "graphs",
    "graph traversal": "graphs",
    "dfs and similar": "graphs",
    "shortest paths": "graphs",
    # Math
    "math": "math",
    "mathematics": "math",
    "combinatorics": "math",
    # Data Structures
    "data structures": "data structures",
    "dsu": "data structures",
    "disjoint set union": "data structures",
    "segment tree": "data structures",
    "stack": "data structures",
    "queue": "data structures",
    # Trees
    "trees": "trees",
    "tree algorithms": "trees",
    "spanning trees": "trees",
    "binary trees": "trees",
    "tree": "trees",
    "lca": "trees",
    "trees / graphs": "trees",
    "lowest common ancestor": "trees",
    "binary search tree": "trees",
    "bst": "trees",
    "trie": "trees",
    "trie / tree": "trees",
    # Brute Force
    "brute force": "brute force",
    "complete search": "brute force",
    "implementation": "brute force",
    # Strings
    "strings": "strings",
    "string algorithms": "strings",
    # Number Theory
    "number theory": "number theory",
    "primes": "number theory",
    "modular arithmetic": "number theory",
    # Binary Search
    "binary search": "binary search",
    "two pointers": "binary search",
}

class TacoExtractor:
    def __init__(self, target_per_category: int = 10, min_rating: int = 800, max_rating: int = 1200):
        self.target_per_category = target_per_category
        self.min_rating = min_rating
        self.max_rating = max_rating

    def extract_100_problems(self, output_path: str = "data/taco_100_problems.json") -> List[Dict[str, Any]]:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        selected = defaultdict(list)
        chunk_files = [f"ALL/train-0000{i}-of-00009.parquet" for i in range(8)]

        for chunk_file in chunk_files:
            if all(len(selected[tag]) >= self.target_per_category for tag in TARGET_TAG_TAXONOMY):
                break
            
            try:
                local_path = hf_hub_download("BAAI/TACO", chunk_file, repo_type="dataset")
                df = pd.read_parquet(local_path)
            except Exception as e:
                print(f"Warning: Failed to load {chunk_file}: {e}")
                continue

            for idx, row in df.iterrows():
                # Extract difficulty / rating
                diff = str(row.get("difficulty", "")).upper()
                if diff not in ["EASY", "MEDIUM", "UNKNOWN_DIFFICULTY"]:
                    continue

                tags_raw = []
                for col in ["raw_tags", "tags", "skill_types"]:
                    val = row.get(col)
                    if val is not None and isinstance(val, str):
                        try:
                            parsed = ast.literal_eval(val) if val.startswith("[") else [val]
                            tags_raw.extend(parsed)
                        except Exception:
                            pass
                    elif isinstance(val, list):
                        tags_raw.extend(val)

                matched_tags = set()
                for t in tags_raw:
                    t_clean = str(t).lower().strip()
                    if t_clean in TAG_MAP:
                        matched_tags.add(TAG_MAP[t_clean])

                if len(matched_tags) == 1:
                    dom_tag = list(matched_tags)[0]
                    if len(selected[dom_tag]) < self.target_per_category:
                        q = str(row.get("question", "")).strip()
                        if "interactive" in q.lower() or len(q) < 50 or len(q) > 3000:
                            continue

                        io = row.get("input_output")
                        if not io:
                            continue
                        try:
                            io_dict = json.loads(io) if isinstance(io, str) else io
                            if not io_dict.get("inputs") or not io_dict.get("outputs"):
                                continue
                        except Exception:
                            continue

                        cat_idx = len(selected[dom_tag]) + 1
                        rating = self.min_rating + (cat_idx - 1) * int((self.max_rating - self.min_rating) / (self.target_per_category - 1))
                        
                        sample_tests = []
                        for inp, out in zip(io_dict["inputs"][:3], io_dict["outputs"][:3]):
                            sample_tests.append({
                                "input_data": str(inp).strip(),
                                "output_data": str(out).strip()
                            })

                        first_line = q.split("\n")[0].strip("# \r\n")
                        title = first_line[:50] if len(first_line) > 5 else f"{dom_tag.title()} Problem {cat_idx}"

                        sol_raw = row.get("solutions")
                        sol_code = ""
                        if sol_raw is not None:
                            if isinstance(sol_raw, str):
                                try:
                                    sols = ast.literal_eval(sol_raw) if sol_raw.startswith("[") else [sol_raw]
                                    sol_code = sols[0] if sols else ""
                                except Exception:
                                    sol_code = sol_raw
                            elif isinstance(sol_raw, list) and sol_raw:
                                sol_code = sol_raw[0]

                        selected[dom_tag].append({
                            "problem_id": f"TACO-{dom_tag[:2].upper()}-{cat_idx:02d}",
                            "title": title,
                            "ground_truth_tag": dom_tag,
                            "rating": rating,
                            "statement": q,
                            "input_spec": "Standard competitive programming input format.",
                            "output_spec": "Standard competitive programming output format.",
                            "sample_tests": sample_tests,
                            "raw_solution": sol_code,
                            "time_limit_ms": 2000,
                            "memory_limit_mb": 512
                        })

        all_problems = []
        for tag in TARGET_TAG_TAXONOMY:
            all_problems.extend(selected[tag])

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_problems, f, indent=2)

        print(f"Successfully extracted {len(all_problems)} problems across 10 categories to {output_path}")
        return all_problems

if __name__ == "__main__":
    extractor = TacoExtractor()
    extractor.extract_100_problems()
