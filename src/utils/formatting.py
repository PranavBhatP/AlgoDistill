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

    # Phase 6: Multi-Turn Execution Feedback & Self-Repair Prompt Templates
    @classmethod
    def format_repair_prompt(
        cls,
        problem_statement: str,
        previous_code: str,
        error_feedback: str,
        predicted_tag: Optional[str] = None,
        input_spec: str = "",
        output_spec: str = "",
        failed_input: str = "",
        expected_output: str = "",
        actual_output: str = "",
        turn_index: int = 1
    ) -> str:
        """Format conversational prompt for iterative code self-repair given sandbox feedback."""
        system_instruction = (
            "You are an expert competitive programmer and debugging specialist. "
            "A previous Python 3 solution to an algorithmic problem failed execution in the sandbox judge. "
            "Analyze the failure diagnostic trace, identify the defect (e.g., syntax error, off-by-one indexing, "
            "recursion limit, or wrong output logic), provide concise chain-of-thought debugging rationale, "
            "and write an optimal, fully corrected Python 3 solution."
        )

        user_content = []
        if predicted_tag and predicted_tag.strip():
            user_content.append(f"[ALGORITHM CATEGORY]: {predicted_tag.strip().lower()}\n")

        user_content.append(f"Problem Statement:\n{problem_statement}\n")
        if input_spec:
            user_content.append(f"Input Specification:\n{input_spec}\n")
        if output_spec:
            user_content.append(f"Output Specification:\n{output_spec}\n")

        user_content.append(f"--- Previous Implementation (Turn {turn_index - 1}) ---\n```python\n{previous_code.strip()}\n```\n")

        user_content.append(f"--- Execution Diagnostic Feedback ---\n{error_feedback.strip()}\n")
        if failed_input.strip():
            user_content.append(f"Failed Test Input:\n```\n{failed_input.strip()[:300]}\n```\n")
        if expected_output.strip():
            user_content.append(f"Expected Output:\n```\n{expected_output.strip()[:200]}\n```\n")
        if actual_output.strip():
            user_content.append(f"Actual Output:\n```\n{actual_output.strip()[:200]}\n```\n")


        user_content.append(
            "Please diagnose the issue and provide the repaired solution in the following format:\n"
            "Rationale:\n<concise 1-sentence debugging analysis>\n\nSolution:\n```python\n<complete corrected Python 3 code>\n```"
        )

        prompt = (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            f"<|im_start|>user\n{''.join(user_content)}<|im_end|>\n"
            f"<|im_start|>assistant\nRationale:\n"
        )
        return prompt

    @classmethod
    def format_repair_target(cls, repair_rationale: str, repaired_code: str) -> str:
        """Format target response for self-repair fine-tuning."""
        clean_code = repaired_code.strip()
        if not clean_code.startswith("```"):
            clean_code = f"```python\n{clean_code}\n```"
        return f"Rationale:\n{repair_rationale.strip()}\n\nSolution:\n{clean_code}<|im_end|>"

    @classmethod
    def extract_code_from_response(cls, response_text: str) -> str:
        """Extract clean executable Python code from model output."""
        text_to_search = response_text
        if "Solution:" in response_text:
            text_to_search = response_text.split("Solution:")[-1]

        # Extract markdown code blocks
        blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", text_to_search, re.DOTALL)
        if not blocks and text_to_search != response_text:
            blocks = re.findall(r"```(?:python)?\s*(.*?)\s*```", response_text, re.DOTALL)

        if blocks:
            best_block = blocks[-1]
            for b in reversed(blocks):
                if any(kw in b for kw in ["import ", "def ", "sys.stdin", "print("]):
                    best_block = b
                    break
            return cls._ensure_syntactic_validity(best_block.strip())

        # Handle unclosed code fence from token cutoff
        if "```python" in text_to_search:
            after = text_to_search.split("```python")[-1]
            code = after.split("```")[0].strip()
            return cls._ensure_syntactic_validity(code)
        elif "```" in text_to_search:
            after = text_to_search.split("```")[-1]
            code = after.split("```")[0].strip()
            return cls._ensure_syntactic_validity(code)

        lines = response_text.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith(("import ", "def ", "sys.", "from ", "class ", "if __name__")):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            return cls._ensure_syntactic_validity("\n".join(code_lines).strip())
        return response_text.strip()

    @classmethod
    def _ensure_syntactic_validity(cls, code: str) -> str:
        """Ensure extracted code is syntactically valid; trims incomplete trailing lines if truncated."""
        if not code:
            return ""
        try:
            compile(code, "<string>", "exec")
            return code
        except SyntaxError:
            pass
        lines = code.split("\n")
        while len(lines) > 2:
            lines.pop()
            candidate = "\n".join(lines).strip()
            try:
                compile(candidate, "<string>", "exec")
                return candidate
            except SyntaxError:
                continue
        return code


