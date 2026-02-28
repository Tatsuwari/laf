# laf/core/execution_engine.py
from __future__ import annotations
from typing import Any, Dict, Optional

from laf.core.execution_context import ExecutionContext
from laf.core.template import resolve_args
from laf.skills.registry import SkillRegistry, SkillSpec
from laf.sandbox.python_sandbox import PythonSandbox
from laf.sandbox.limits import SandboxLimits
from laf.core.trace_recorder import TraceRecorder


class Executor:
    """
    Deterministic step executor.

    - Enforces policy gates
    - Executes skills (sandboxed or in-process)
    - Emits structured trace events
    """

    def __init__(
        self,
        skills: SkillRegistry,
        internet_available: bool = True,
        internal_only: bool = False,
        trace: Optional[TraceRecorder] = None,
    ):
        self.skills = skills
        self.internet_available = internet_available
        self.internal_only = internal_only
        self.trace = trace

    def run(
        self,
        ctx: ExecutionContext,
        trace_run=None,
    ) -> ExecutionContext:

        steps = ctx.plan.get("steps", [])
        ctx.add_event("plan", goal=ctx.goal, steps=len(steps))

        if self.trace:
            self.trace.execution_start(
                trace_run,
                goal=ctx.goal,
                step_count=len(steps),
            )

        for step in steps:

            step_id = str(step.get("id", ""))
            stype = step.get("type", "manual_review")

            if self.trace:
                self.trace.execution_step(
                    trace_run,
                    step_id=step_id,
                    step_type=stype,
                )

            if stype != "tool_call":
                ctx.add_event(
                    "note",
                    step_id=step_id,
                    message=step.get("description", "manual_review"),
                )
                continue

            tool_name = (step.get("plugin") or "").strip()
            raw_args = step.get("args") if isinstance(step.get("args"), dict) else {}
            args = resolve_args(raw_args, ctx.memory)

            skill = self.skills.get(tool_name)

            if not skill:
                self._record_error(ctx, step_id, tool_name, f"Unknown skill: {tool_name}", trace_run)
                continue

            if self.internal_only and not skill.internal:
                self._record_error(ctx, step_id, tool_name, "Blocked by policy: internal_only", trace_run)
                continue

            if skill.requires_internet and not self.internet_available:
                self._record_error(ctx, step_id, tool_name, "Internet unavailable", trace_run)
                continue

            ctx.add_event("tool_start", step_id=step_id, tool=tool_name, args=args)

            if self.trace:
                self.trace.tool_start(
                    trace_run,
                    step_id=step_id,
                    tool=tool_name,
                    args=args,
                )

            try:
                result = self._execute_skill(skill, args)

                ctx.tool_results.append(
                    {
                        "step_id": step_id,
                        "tool": tool_name,
                        "result": result,
                    }
                )

                if result.get("ok"):
                    ctx.remember(f"step_{step_id}", result.get("result"))

                ctx.add_event(
                    "tool_end",
                    step_id=step_id,
                    tool=tool_name,
                    ok=bool(result.get("ok", True)),
                )

                if self.trace:
                    self.trace.tool_end(
                        trace_run,
                        step_id=step_id,
                        tool=tool_name,
                        ok=bool(result.get("ok", True)),
                        result=result,
                    )

            except Exception as e:

                if skill.bypass_errors:
                    out = {
                        "ok": False,
                        "tool": tool_name,
                        "error": str(e),
                        "bypassed": True,
                    }

                    ctx.tool_results.append(
                        {
                            "step_id": step_id,
                            "tool": tool_name,
                            "result": out,
                        }
                    )

                    ctx.add_event(
                        "tool_end",
                        step_id=step_id,
                        tool=tool_name,
                        ok=False,
                        bypassed=True,
                        error=str(e),
                    )

                    if self.trace:
                        self.trace.tool_end(
                            trace_run,
                            step_id=step_id,
                            tool=tool_name,
                            ok=False,
                            result=out,
                        )

                else:
                    ctx.add_event(
                        "tool_end",
                        step_id=step_id,
                        tool=tool_name,
                        ok=False,
                        error=str(e),
                    )

                    if self.trace:
                        self.trace.execution_error(
                            trace_run,
                            step_id=step_id,
                            tool=tool_name,
                            error=str(e),
                        )

                    raise

        if self.trace:
            self.trace.execution_finish(trace_run)

        return ctx

    def _execute_skill(self, skill: SkillSpec, args: Dict[str, Any]) -> Dict[str, Any]:

        if skill.sandboxed:

            if not skill.entrypoint:
                return {"ok": False, "error": f"Skill '{skill.name}' missing sandbox entrypoint"}

            allow_net = bool(self.internet_available and skill.requires_internet)

            sandbox = PythonSandbox(
                SandboxLimits(timeout_sec=15.0, allow_network=allow_net)
            )

            sb_result = sandbox.run(skill.entrypoint, args)

            if sb_result.ok:
                return {"ok": True, "tool": skill.name, "result": sb_result.result}

            return {
                "ok": False,
                "tool": skill.name,
                "error": sb_result.error,
                "traceback": sb_result.traceback,
            }

        if not skill.fn:
            return {"ok": False, "error": f"Skill '{skill.name}' has no callable"}

        out = skill.fn(**(args or {}))

        if isinstance(out, dict) and "ok" in out:
            return out

        return {"ok": True, "tool": skill.name, "result": out}

    def _record_error(
        self,
        ctx: ExecutionContext,
        step_id: str,
        tool: str,
        message: str,
        trace_run=None,
    ):
        ctx.add_event("error", step_id=step_id, tool=tool, error=message)

        ctx.tool_results.append(
            {
                "step_id": step_id,
                "tool": tool,
                "result": {"ok": False, "error": message},
            }
        )

        if self.trace:
            self.trace.execution_error(
                trace_run,
                step_id=step_id,
                tool=tool,
                error=message,
            )

    # TODO:
    # - Add per-skill execution time measurement
    # - Add deterministic timeout override support
    # - Add step dependency validation when DAG execution is enabled