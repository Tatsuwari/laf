from typing import List, Dict, Any
from ..llm import LLM
from ..config import GenConfig

class GeneratorAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.gen = GenConfig(max_new_tokens=450, do_sample=False, temperature=0.0)

    def generate(self, task: str, plan: Dict[str, Any], retrieved: List[Dict[str, Any]]):
        system = "You answer using provided context, and stay grounded."
        ctx = "\n\n".join([f"[doc score={d['score']:.3f}] {d['text']}" for d in retrieved]) if retrieved else "(no docs)"
        user = f"""
Task: {task}

Plan (JSON):
{plan}

Retrieved Context:
{ctx}

Write a clear, structured answer. If context is missing, state assumptions.
"""
        return self.llm.chat(system=system, user=user, gen=self.gen)
