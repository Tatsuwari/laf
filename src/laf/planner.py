from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .llm import LLM
from .config import SystemConfig, GenConfig
from .json_parse import safe_parse_struct

@dataclass
class PlanStep:
    id: int
    description: str

@dataclass
class Plan:
    goal: str
    subtasks: List[PlanStep]

class PlannerAgent:
    def __init__(self, llm: LLM, cfg: SystemConfig):
        self.llm = llm
        self.cfg = cfg
        self.gen = GenConfig(max_new_tokens=350, do_sample=False, temperature=0.0)

    def _prompt(self, goal: str) -> str:
        return f"""
You are a Planning Agent.

Return ONLY valid JSON.
Do NOT include markdown.
Use double quotes for keys and strings.
Output must start with '{{' and end with '}}'.

Schema:
{{
  "goal": "string",
  "subtasks": [
    {{"id": 1, "description": "string"}}
  ]
}}

Rules:
- Generate as many subtasks as needed, but no more than {self.cfg.max_steps}.
- Make steps small, concrete, and actionable.
- Each description must start with a verb.

User Goal: {goal}
"""

    def plan(self, goal: str, retries: int = 2) -> Plan:
        system = "You create structured plans."
        user = self._prompt(goal)

        for _ in range(retries + 1):
            raw = self.llm.chat(system=system, user=user, gen=self.gen)
            parsed = safe_parse_struct(raw)

            if isinstance(parsed, dict) and "goal" in parsed and "subtasks" in parsed:
                subtasks = []
                for item in parsed.get("subtasks", []):
                    if isinstance(item, dict) and "id" in item and "description" in item:
                        try:
                            sid = int(item["id"])
                        except Exception:
                            continue
                        desc = str(item["description"]).strip()
                        if desc:
                            subtasks.append(PlanStep(id=sid, description=desc))
                return Plan(goal=str(parsed["goal"]), subtasks=subtasks)

            user += "\nREMINDER: Output ONLY valid JSON that matches the schema."
        # fallback
        return Plan(goal=goal, subtasks=[])
