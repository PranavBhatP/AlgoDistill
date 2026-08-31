"""Prompt formatting templates for Stage 1 (Classification) and Stage 2 (Generation)."""

import re
from typing import Dict, Any, Optional
from src.data.dataset_builder import DatasetItem


class PromptFormatter:
    """Standardized prompt templates for reasoning distillation."""

    # Stage 1: Algorithmic Tag Classification Prompts
    STAGE1_INSTRUCTION = (
        "You are an expert competitive programmer and algorithm specialist. "
        "Analyze the following problem statement and classify its primary algorithmic category "
        "from the following taxonomy:\n"
        "- dynamic programming\n"
        "- greedy\n"
        "- graphs\n"
        "- math\n"
        "- data structures\n"
        "- trees\n"
        "- brute force\n"
        "- strings\n"
        "- number theory\n"
        "- binary search\n"
    )

    @classmethod
    def format_stage1_prompt(cls, problem_statement: str, input_spec: str = "", output_spec: str = "") -> str:
        """Format input for Stage 1 Tag Classifier."""
        prompt = (
            f"<|im_start|>system\n{cls.STAGE1_INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Problem Statement:\n{problem_statement}\n\n"
        )
        if input_spec:
            prompt += f"Input Specification:\n{input_spec}\n\n"
        if output_spec:
            prompt += f"Output Specification:\n{output_spec}\n\n"
        prompt += "Predict the single dominant algorithm tag for this problem.<|im_end|>\n"
        prompt += "<|im_start|>assistant\nTag: "
        return prompt

    # Stage 2: Conditioned Code Generation Prompts ([TAG] + [PROBLEM] -> [RATIONALE] + [CODE])
    @classmethod
    def format_stage2_conditioned_prompt(
        cls,
        predicted_tag: str,
        problem_statement: str,
        input_spec: str = "",
        output_spec: str = ""
    ) -> str:
        """Format input for Stage 2 Tag-Conditioned Generator."""
        prompt = (
            f"<|im_start|>system\n"
            f"You are an elite competitive programmer. You are given a problem statement along with "
            f"its algorithmic paradigm. First, explain your chain-of-thought rationale utilizing this paradigm, "
            f"and then write an optimal, complete Python 3 solution.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"[ALGORITHM CATEGORY]: {predicted_tag}\n\n"
            f"Problem Statement:\n{problem_statement}\n\n"
        )
        if input_spec:
            prompt += f"Input Specification:\n{input_spec}\n\n"
        if output_spec:
            prompt += f"Output Specification:\n{output_spec}\n\n"
        prompt += "Provide step-by-step reasoning followed by the executable Python solution.<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    # Baseline: Unconditioned Code Generation Prompts ([PROBLEM] -> [RATIONALE] + [CODE])
    @classmethod
    def format_baseline_unconditioned_prompt(
        cls,
        problem_statement: str,
        input_spec: str = "",
        output_spec: str = ""
    ) -> str:
        """Format input for Baseline Unconditioned Generator (no tag signal)."""
        prompt = (
            f"<|im_start|>system\n"
            f"You are an elite competitive programmer. First, explain your chain-of-thought rationale, "
            f"and then write an optimal, complete Python 3 solution.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Problem Statement:\n{problem_statement}\n\n"
        )
        if input_spec:
            prompt += f"Input Specification:\n{input_spec}\n\n"
        if output_spec:
            prompt += f"Output Specification:\n{output_spec}\n\n"
        prompt += "Provide step-by-step reasoning followed by the executable Python solution.<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    @classmethod
    def format_stage2_target(cls, rationale: str, code: str) -> str:
        """Format target response (Rationale + Code Block)."""
        clean_code = code.strip()
        if not clean_code.startswith("```"):
            clean_code = f"```python\n{clean_code}\n```"
        return f"Rationale:\n{rationale.strip()}\n\nSolution:\n{clean_code}<|im_end|>"

    @classmethod
    def extract_code_from_response(cls, response_text: str) -> str:
        """Extract clean executable Python code from model output."""
        # Look for markdown code blocks ```python ... ```
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", response_text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()

        # Fallback: if response contains standard python headers
        lines = response_text.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("def ") or line.strip().startswith("sys."):
                in_code = True
            if in_code:
                code_lines.append(line)

        return "\n".join(code_lines).strip() if code_lines else response_text.strip()
