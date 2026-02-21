from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import re

from ..llm import LLM
from ..config import GenConfig


@dataclass
class RagDecision:
    use_rag: bool
    query: str
    top_k: int = 5
    reason: str = ""


class RagRouter:
    """
    Decide whether to retrieve + what query to use.
    Scalable: can later incorporate heuristics + telemetry.
    """

    def __init__(self, llm: Optional[LLM] = None, mode: str = "AUTO"):
        """
        mode: "AUTO" | "ALWAYS" | "NEVER"
        """
        self.llm = llm
        self.mode = mode.upper()
        self.gen = GenConfig(max_new_tokens=120, do_sample=False, temperature=0.0)

    def _heuristic(self, task: str) -> bool:
        # Quick local heuristic (no LLM) for obvious cases
        t = task.lower()
        triggers = [
            "latest", "today", "current", "news", "release", "compare", "paper",
            "research", "sources", "cite", "references", "documentation", "pricing",
            "requirements", "regulation", "policy", "standard"
        ]
        return any(k in t for k in triggers)

    def decide(self, task: str, plan: Optional[Dict[str, Any]] = None) -> RagDecision:
        if self.mode == "NEVER":
            return RagDecision(use_rag=False, query=task, top_k=0, reason="mode=NEVER")
        if self.mode == "ALWAYS":
            return RagDecision(use_rag=True, query=task, top_k=5, reason="mode=ALWAYS")

        # AUTO: try heuristic first
        if self._heuristic(task):
            return RagDecision(use_rag=True, query=task, top_k=5, reason="heuristic_trigger")

        # If no LLM, fallback to heuristic-only
        if self.llm is None:
            return RagDecision(use_rag=False, query=task, top_k=0, reason="no_llm_and_no_trigger")

        # LLM-based router: decide + craft query
        plan_txt = ""
        if plan and isinstance(plan, dict):
            # include up to a few steps to enrich retrieval query
            subs = plan.get("subtasks", [])[:6]
            plan_txt = "\n".join([f"- {s.get('description','')}" for s in subs])

        system = "You decide whether retrieval (RAG) is needed and craft a good search query."
        user = f"""
Task: {task}

Plan steps (optional):
{plan_txt if plan_txt else "(none)"}

Return ONLY a compact JSON object:
{{
  "use_rag": true/false,
  "query": "string",
  "top_k": 3..10,
  "reason": "short string"
}}

Rules:
- If the task can be answered from general knowledge (no need for external facts), set use_rag=false.
- If it needs facts, citations, specific details, or comparisons, set use_rag=true.
- query should be optimized for retrieving relevant docs.
- Output must start with '{{' and end with '}}'.
"""

        raw = self.llm.chat(system=system, user=user, gen=self.gen)

        # lightweight JSON parse (avoid importing your whole parser here)
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return RagDecision(use_rag=False, query=task, top_k=0, reason="router_parse_failed")

        import json
        try:
            obj = json.loads(m.group(0))
            use_rag = bool(obj.get("use_rag", False))
            query = str(obj.get("query", task)).strip() or task
            top_k = int(obj.get("top_k", 5))
            top_k = max(3, min(10, top_k))
            reason = str(obj.get("reason", "")).strip()
            return RagDecision(use_rag=use_rag, query=query, top_k=top_k, reason=reason)
        except Exception:
            return RagDecision(use_rag=False, query=task, top_k=0, reason="router_json_failed")