from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional

@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    params_schema: Optional[dict] = None  # simple JSON schema-like dict

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def has(self, name: str) -> bool:
        return name in self.tools

    def run(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.tools:
            return {"ok": False, "error": f"Tool not found: {name}"}
        try:
            return self.tools[name].fn(args)
        except Exception as e:
            return {"ok": False, "error": str(e)}
