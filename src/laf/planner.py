from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from .llm import LLM
from .config import SystemConfig, GenConfig
from .json_parse import safe_parse_struct
import json

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
        """
        plan_format: 'linear' (default) | 'tree' | 'dag'
        """
        cat = catalog or {"intents": [], "plugins": []}

        # Extract tool keys robustly (supports key/name just in case)
        tool_names = []
        for p in cat.get("plugins", []) or []:
            if isinstance(p, dict):
                k = p.get("key") or p.get("name")
                if k:
                    tool_names.append(str(k))
        tool_names = sorted(set(tool_names))

        intent_names = []
        for i in cat.get("intents", []) or []:
            if isinstance(i, dict):
                k = i.get("key") or i.get("intent_key") or i.get("name")
                if k:
                    intent_names.append(str(k))
        intent_names = sorted(set(intent_names))

        # Keep the allowlist small but explicit
        tools_allow = json.dumps(tool_names, ensure_ascii=False)
        intents_allow = json.dumps(intent_names, ensure_ascii=False)

        return f"""
    You are a Planning Agent that MUST plan using ONLY the provided catalog.

    Return ONLY valid JSON.
    Do NOT include markdown.
    Use double quotes for keys and strings.
    Output must start with '{{' and end with '}}'.

    IMPORTANT:
    - The following strings are EXAMPLES ONLY and MUST NEVER appear in your output:
    "plugin_key_from_catalog", "intent_key_from_catalog"
    - For any tool_call, the "plugin" field MUST be one of the allowed plugin keys below.
    - If no suitable plugin exists, output a node/step with type "manual_review" and explain what capability is missing.

    Allowed plugin keys (tool_call.plugin MUST be EXACTLY one of these):
    {tools_allow}

    Allowed intent keys (intent_key SHOULD be one of these, if you use intents):
    {intents_allow}

    You are allowed to output ONE of these plan formats:

    1) linear:
    {{
    "format": "linear",
    "goal": "string",
    "steps": [
        {{"id":"1","type":"tool_call","plugin":"{tool_names[0] if tool_names else 'manual_review'}","args":{{}},"description":"string"}}
    ]
    }}

    2) tree:
    {{
    "format": "tree",
    "goal": "string",
    "root": {{
        "id":"root",
        "type":"category",
        "name":"string",
        "children":[
        {{
            "id":"n1",
            "type":"tool_call",
            "plugin":"{tool_names[0] if tool_names else 'manual_review'}",
            "args":{{}},
            "description":"string"
        }}
        ]
    }}
    }}

    3) dag:
    {{
    "format":"dag",
    "goal":"string",
    "nodes":[
        {{"id":"n1","type":"tool_call","plugin":"{tool_names[0] if tool_names else 'manual_review'}","args":{{}},"description":"string"}}
    ],
    "edges":[]
    }}

    Rules:
    - Prefer format "{plan_format}" unless it is impossible.
    - Keep steps small, concrete, and actionable.
    - Maximum steps/nodes: {self.cfg.max_steps}.

    Catalog (subset):
    Intent Catalog:
    {cat.get("intents", [])}

    Plugin Catalog:
    {cat.get("plugins", [])}

    User Goal: {goal}
    """.strip()


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
