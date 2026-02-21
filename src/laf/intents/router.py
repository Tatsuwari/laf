import re
import numpy as np
from typing import Dict, Any, Tuple, List
from .store import IntentStore
from .logger import IntentLogger
from ..llm import LLM
from ..config import SystemConfig, GenConfig
from ..json_parse import safe_parse_struct

def normalize_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s_]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:48] if s else "intent"

class HybridIntentRouter:
    """
    1) Embedding similarity routes to existing intents
    2) If below threshold -> LLM proposes new intent (key, description, tool)
    """
    def __init__(self, llm: LLM, store: IntentStore, cfg: SystemConfig, logger: IntentLogger):
        self.llm = llm
        self.store = store
        self.cfg = cfg
        self.logger = logger
        self.gen = GenConfig(max_new_tokens=160, do_sample=False, temperature=0.0)

    def _embed_task(self, text: str) -> np.ndarray:
        v = self.store.embedder.encode([text], normalize_embeddings=True)[0]
        return np.array(v, dtype=np.float32)

    def _cos(self, a: np.ndarray, b: np.ndarray) -> float:
        # normalized embeddings: dot product = cosine
        return float(np.dot(a, b))

    def top_matches(self, task_desc: str) -> List[Tuple[str, float]]:
        tv = self._embed_task(task_desc)
        scored = []
        for key, intent in self.store.intents.items():
            if intent.centroid is None:
                self.store.recompute_centroid(key)
            iv = np.array(intent.centroid, dtype=np.float32)
            scored.append((key, self._cos(tv, iv)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _propose_intent(self, task_desc: str) -> Dict[str, Any]:
        existing = list(self.store.intents.keys())
        system = "You design intent keys and descriptions for routing."
        user = f"""
You are an Intent Designer.

Task:
"{task_desc}"

Existing intent keys:
{existing}

Return ONLY valid JSON:
{{
  "key": "short_snake_case_key",
  "description": "one sentence describing what belongs here",
  "tool": "tool_name_or_manual_review"
}}

Rules:
- If an existing key fits, return that existing key (do NOT invent a new one).
- Key must be concise snake_case, <= 4 words.
- Use double quotes only.
- Output must start with '{{' and end with '}}'.
"""
        raw = self.llm.chat(system=system, user=user, gen=self.gen)
        parsed = safe_parse_struct(raw)
        if not isinstance(parsed, dict) or "key" not in parsed:
            return {"key": "misc", "description": "Misc tasks that don't match existing intents.", "tool": "manual_review"}
        return {
            "key": normalize_key(str(parsed.get("key", "misc"))),
            "description": str(parsed.get("description", "Auto-generated intent.")).strip(),
            "tool": str(parsed.get("tool", "manual_review")).strip() or "manual_review"
        }

    def route(self, task_desc: str) -> Dict[str, Any]:
        matches = self.top_matches(task_desc)
        best_key, best_score = matches[0] if matches else ("unknown", 0.0)

        created = False
        chosen = best_key

        if best_score < self.cfg.similarity_threshold:
            proposal = self._propose_intent(task_desc)
            pkey = proposal["key"]

            if pkey not in self.store.intents:
                self.store.add_intent(pkey, proposal["description"], tool=proposal.get("tool", "manual_review"))
                created = True
                chosen = pkey
            else:
                chosen = pkey  # LLM decided existing key fits

        # update store with example + persist
        if chosen in self.store.intents:
            self.store.add_example(chosen, task_desc)
            self.store.save()

        top = [{"intent": k, "score": float(s)} for k, s in matches[: self.cfg.top_k_matches]]

        # logging rules
        if created and self.cfg.log_if_created_new:
            self.logger.log({
                "event": "NEW_INTENT",
                "task_desc": task_desc,
                "chosen": chosen,
                "best_match": {"intent": best_key, "score": float(best_score)},
                "top": top
            })
        elif best_score < self.cfg.log_if_below_threshold:
            self.logger.log({
                "event": "LOW_CONFIDENCE_ROUTE",
                "task_desc": task_desc,
                "chosen": chosen,
                "best_match": {"intent": best_key, "score": float(best_score)},
                "top": top
            })

        return {
            "intent": chosen,
            "created_new_intent": created,
            "best_match": {"intent": best_key, "score": float(best_score)},
            "top_matches": top,
            "tool": self.store.intents.get(chosen).tool if chosen in self.store.intents else "manual_review"
        }
