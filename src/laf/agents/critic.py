from ..llm import LLM
from ..config import GenConfig

class CriticAgent:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.gen = GenConfig(max_new_tokens=250, do_sample=False, temperature=0.0)

    def refine(self, task: str, draft: str):
        system = "You refine answers. Do not repeat prompts."

        user = f"""
Task:
{task}

Answer:
{draft}

If the answer is already correct and grounded, return it unchanged.
Otherwise, improve clarity only.
Return final answer only.
"""

        return self.llm.chat(system=system, user=user, gen=self.gen)
    