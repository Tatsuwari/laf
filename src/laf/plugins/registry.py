from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple
from ..trace import Tracer


# ============================================================
# Execution Policy (controls runtime behavior)
# ============================================================

@dataclass
class ExecutionPolicy:
    internal_only: bool = False
    internet_available: bool = True


# ============================================================
# Tool Specification
# ============================================================

@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[[Dict[str, Any]], Dict[str, Any]]
    params_schema: Optional[dict] = None

    # Optional metadata
    category_path: Optional[list[str]] = None
    returns_schema: Optional[dict] = None
    tags: Optional[list[str]] = None
    constraints: Optional[dict] = None

    # Execution controls
    internal: bool = False
    requires_internet: bool = False
    bypass_errors: bool = False


# ============================================================
# Tool Registry
# ============================================================

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolSpec] = {}

    # --------------------------------------------------------
    # Registration
    # --------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def has(self, name: str) -> bool:
        return name in self.tools

    def clear(self) -> None:
        self.tools.clear()

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    # --------------------------------------------------------
    # Argument Validation (Minimal JSON-schema-like)
    # --------------------------------------------------------

    def validate_args(self, schema: dict, args: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(args, dict):
            return False, "args must be an object"

        stype = schema.get("type")
        if stype and stype != "object":
            return False, f"schema type must be object (got {stype})"

        required = schema.get("required", []) or []
        if isinstance(required, list):
            for k in required:
                if k not in args:
                    return False, f"missing required field: {k}"

        props = schema.get("properties", {}) or {}
        if isinstance(props, dict):
            for k, ps in props.items():
                if k not in args:
                    continue
                if not isinstance(ps, dict):
                    continue

                expected = ps.get("type")
                if not expected:
                    continue

                v = args[k]

                if expected == "string" and not isinstance(v, str):
                    return False, f"field {k} must be string"
                if expected == "number" and not isinstance(v, (int, float)):
                    return False, f"field {k} must be number"
                if expected == "integer" and not isinstance(v, int):
                    return False, f"field {k} must be integer"
                if expected == "boolean" and not isinstance(v, bool):
                    return False, f"field {k} must be boolean"
                if expected == "object" and not isinstance(v, dict):
                    return False, f"field {k} must be object"
                if expected == "array" and not isinstance(v, list):
                    return False, f"field {k} must be array"

        return True, ""

    # --------------------------------------------------------
    # Tool Execution
    # --------------------------------------------------------

    def run(
        self,
        name: str,
        args: Dict[str, Any],
        tracer: Optional["Tracer"] = None,
        policy: Optional[ExecutionPolicy] = None,
    ) -> Dict[str, Any]:

        policy = policy or ExecutionPolicy()

        if tracer:
            tracer.emit("tool.run.start", tool=name, args=args)

        # Tool existence check
        if name not in self.tools:
            out = {"ok": False, "error": f"Tool not found: {name}"}
            if tracer:
                tracer.emit("tool.run.end", tool=name, ok=False, error=out["error"])
            return out

        spec = self.tools[name]

        # ----------------------------------------------------
        # Policy Enforcement
        # ----------------------------------------------------

        # Block internet tools if internal_only=True
        if policy.internal_only and spec.requires_internet:
            out = {
                "ok": False,
                "tool": name,
                "error": "Internet-based tool disabled (internal_only=True)",
            }
            if tracer:
                tracer.emit("tool.run.end", tool=name, ok=False, error=out["error"])
            return out

        # Block if internet required but not available
        if spec.requires_internet and not policy.internet_available:
            out = {
                "ok": False,
                "tool": name,
                "error": "Internet not available",
            }
            if tracer:
                tracer.emit("tool.run.end", tool=name, ok=False, error=out["error"])
            return out

        # ----------------------------------------------------
        # Argument Validation
        # ----------------------------------------------------

        if spec.params_schema:
            ok, err = self.validate_args(spec.params_schema, args or {})
            if not ok:
                out = {
                    "ok": False,
                    "tool": name,
                    "error": f"Args validation failed: {err}",
                    "args": args,
                }
                if tracer:
                    tracer.emit("tool.run.end", tool=name, ok=False, error=out["error"])
                return out

        # ----------------------------------------------------
        # Execution
        # ----------------------------------------------------

        try:
            result = spec.fn(args or {})

            # Allow tool to control its own envelope
            if isinstance(result, dict) and "ok" in result:
                out = result
            else:
                out = {"ok": True, "tool": name, "result": result}

            if tracer:
                tracer.emit("tool.run.end", tool=name, ok=bool(out.get("ok", True)))

            return out

        except Exception as e:

            # If tool allows bypassing errors
            if spec.bypass_errors:
                out = {
                    "ok": False,
                    "tool": name,
                    "error": str(e),
                    "bypassed": True,
                }
                if tracer:
                    tracer.emit("tool.run.end", tool=name, ok=False, error=str(e))
                return out

            # Otherwise raise (fail fast)
            raise