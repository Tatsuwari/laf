# laf/skills/registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import inspect


@dataclass
class SkillSpec:
    name: str
    description: str
    negative_description: Optional[str] = None
    fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    entrypoint: Optional[str] = None   # "module:function" for sandbox
    params_schema: Optional[dict] = None

    # policy flags
    internal: bool = False
    requires_internet: bool = False
    bypass_errors: bool = False
    sandboxed: bool = False

    # memory
    memory_extractor: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None
    memory_tags: Optional[List[str]] = None
    memory_ttl: Optional[int] = None  # seconds (not iterations)



class SkillRegistry:
    def __init__(self):
        self.tools: Dict[str, SkillSpec] = {}

    def register(self, skill: SkillSpec) -> None:
        self.tools[skill.name] = skill

    def has(self, name: str) -> bool:
        return name in self.tools

    def list(self) -> List[str]:
        return sorted(self.tools.keys())

    def get(self, name: str) -> Optional[SkillSpec]:
        return self.tools.get(name)

    def to_planner_schema(self) -> List[Dict[str, Any]]:
        out = []
        for skill in self.tools.values():
            params = []

            if skill.fn:
                sig = inspect.signature(skill.fn)
                for name, p in sig.parameters.items():
                    if p.default is inspect.Parameter.empty:
                        params.append(name)
                    else:
                        params.append(f"{name}?")

            out.append({
                "name": skill.name,
                "params": params,
                "description": skill.description,
            })

        return out