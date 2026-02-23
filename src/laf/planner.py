from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from .llm import LLM
from .config import SystemConfig, GenConfig
from .json_parse import safe_parse_struct

from .trace import Tracer

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

    def _prompt(self, goal: str, catalog: Optional[Dict[str, Any]], plan_format: str = 'linear') -> str:
        '''
        plan_format: 'linear' (default) | 'tree' | 'dag'
        '''
        # Provide a compact view of capabilities to keep prompt small.
        cat = catalog or {'intents': [], 'plugins': []}
        return f"""
You are a Planning Agent that MUST plan using the provided Intent + Plugin catalog.

Return ONLY valid JSON.
Do NOT include markdown.
Use double quotes for keys and strings.
Output must start with '{{' and end with '}}'.

You are allowed to output ONE of these plan formats:
1) linear:
{{
  "format": "linear",
  "goal": "string",
  "steps": [
    {{"id": "1", "type": "step", "description": "string"}}
  ]
}}

2) tree:
{{
  "format": "tree",
  "goal": "string",
  "root": {{
    "id": "root",
    "type": "category",
    "name": "string",
    "children": [
      {{
        "id": "n1",
        "type": "intent",
        "intent_key": "intent_key_from_catalog",
        "description": "string",
        "children": [
          {{
            "id": "n1.1",
            "type": "tool_call",
            "plugin": "plugin_key_from_catalog",
            "args": {{}},
            "description": "string"
          }}
        ]
      }}
    ]
  }}
}}

3) dag:
{{
  "format": "dag",
  "goal": "string",
  "nodes": [
    {{"id":"n1","type":"intent","intent_key":"intent_key_from_catalog","description":"string"}},
    {{"id":"n2","type":"tool_call","plugin":"plugin_key_from_catalog","args":{{}},"description":"string"}}
  ],
  "edges": [
    {{"from":"n1","to":"n2","data":"optional_string"}}
  ]
}}

Rules:
- Prefer format "{plan_format}" unless it is impossible.
- Use ONLY intent keys and plugin keys that exist in the catalog.
- If you need a capability that doesn't exist, output a step/node with type "manual_review" and explain what is missing.
- Keep steps small, concrete, and actionable.
- Maximum steps/nodes: {self.cfg.max_steps}.

Intent Catalog (subset):
{cat.get("intents", [])}

Plugin Catalog (subset):
{cat.get("plugins", [])}

User Goal: {goal}
"""


    def plan(
            self,
            goal: str,
            retries: int = 2,
            catalog: Optional[Dict[str, Any]] = None,
            plan_format: str = "linear",
            tracer: Optional['Tracer'] = None
            ) -> Union[Plan, Dict[str, Any]]:
        if tracer:
            tracer.emit('planner.start', goal=goal, plan_Format=plan_format, has_catalog=bool(catalog))
            
        system = "You create structured plans constrained to available capabilities."
        user = self._prompt(goal, catalog=catalog, plan_format=plan_format)

        for attempt in range(retries + 1):
            raw = self.llm.chat(system=system, user=user, gen=self.gen)
            if tracer:
                tracer.emit('planner.raw',attempt=attempt, raw_preview=str(raw)[:500])
            
            parsed = safe_parse_struct(raw)

            # If planner returns an IR dict, keep it.
            if isinstance(parsed, dict) and parsed.get("goal") and (parsed.get("steps") or parsed.get("root") or parsed.get("nodes")):
                # normalize minimal fields
                if "format" not in parsed:
                    parsed["format"] = plan_format

                if tracer:
                    tracer.emit(
                        'planner.output',
                        format=str(parsed.get("format")),
                        keys=list(parsed.keys()),
                        size_hint=len(json.dumps(parsed)) if isinstance(parsed, dict) else None,
                        )
                    
                return parsed
            
            # Backward-compatible fallback: accept legacy format. log it too if hit
            if isinstance(parsed, dict) and "goal" in parsed and "subtasks" in parsed:
                if tracer:
                    tracer.emit('planner.legacy_output', step_count=len(parsed.get("subtasks", [])))
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

            user += "\nREMINDER: Output ONLY valid JSON that matches the schema and uses only catalog keys."
        # fallback
        if tracer:
            tracer.emit('planner.fallback_empty',reason='parse_failed')
        return Plan(goal=goal, subtasks=[])
