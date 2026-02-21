from ..llm import LLM
from ..config import GenConfig

class CriticAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.gen = GenConfig(max_new_tokens=300, do_sample=False, temperature=0.0)

    def refine(self, task: str, draft: str):
        system = "You are a careful critic. Improve clarity and reduce hallucinations."
        user = f"""
Task: {task}

Draft Answer:
{draft}

Return an improved final answer. If you remove claims, explain briefly.
"""
        return self.llm.chat(system=system, user=user, gen=self.gen)
