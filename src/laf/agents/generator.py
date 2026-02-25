import json
from typing import List, Dict, Any
from ..llm import LLM
from ..config import GenConfig


class GeneratorAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.gen = GenConfig(max_new_tokens=400, do_sample=False, temperature=0.0)

    def generate(
        self,
        task: str,
        plan: Dict[str, Any],
        tool_results: List[Dict[str, Any]] | None = None,
        retrieved: List[Dict[str, Any]] | None = None,
    ):
        tool_results = tool_results or []
        retrieved = retrieved or []

        # 🚀 HARD BYPASS: if pure tool output, skip LLM entirely
        if tool_results and not retrieved:
            # If last tool result exists, return it directly
            last = tool_results[-1]["result"]
            return json.dumps(last, indent=2)

        # Otherwise grounded generation
        system = "You must answer using ONLY executed tool results and retrieved context."

        tool_json = json.dumps(tool_results, indent=2)
        ctx = "\n\n".join(
            [f"[doc score={d['score']:.3f}] {d['text']}" for d in retrieved]
        ) if retrieved else "(no docs)"

        user = f"""
Task:
{task}

Plan:
{json.dumps(plan, indent=2)}

Executed Tool Results:
{tool_json}

Retrieved Context:
{ctx}

Rules:
- If tool_results exist, DO NOT invent values.
- Use tool_results exactly as provided.
- If the task asks for JSON, return only valid JSON.
- Do NOT restate the plan.

Return final answer only.
"""

        return self.llm.chat(system=system, user=user, gen=self.gen)