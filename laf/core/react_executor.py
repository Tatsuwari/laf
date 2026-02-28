import json
from typing import Any, Dict, Optional

from laf.core.agent_hooks import AgentHooks
from laf.core.memory import MemoryManager
from laf.core.trace_recorder import TraceRecorder
from laf.prompts.react import (
    build_react_system_prompt,
    build_react_loop_prompt,
)


class ReActExecutor:
    """
    ReAct engine with optional Reflect, Critic, Revision,
    Memory extraction, Confidence scoring and automatic retry.
    """

    def __init__(
        self,
        llm,
        skills,
        trace: Optional[TraceRecorder] = None,
        max_iters: int = 8,
        hooks: Optional[AgentHooks] = None,
        enable_reflect: bool = False,
        enable_critique: bool = False,
        enable_revision: bool = False,
        enable_memory_extract: bool = False,
        enable_confidence: bool = True,
        dedupe_tool_calls: bool = True,
    ):
        self.llm = llm
        self.skills = skills
        self.trace = trace
        self.max_iters = max_iters

        self.hooks = hooks or AgentHooks()

        self.enable_reflect = enable_reflect
        self.enable_critique = enable_critique
        self.enable_revision = enable_revision
        self.enable_memory_extract = enable_memory_extract
        self.enable_confidence = enable_confidence
        self.dedupe_tool_calls = dedupe_tool_calls

        self._tool_call_cache: Dict[str, Any] = {}

        self.memory_manager = MemoryManager(trace=trace)

    def run(
        self,
        ctx,
        conf_threshold: float = 0.6,
        max_retries: int = 1,
    ):

        ctx.history = []
        ctx.tool_results = []
        ctx.final_answer = None
        ctx.iteration = 0
        ctx.retry_count = 0

        ctx.skills = self.skills
        ctx.memory_manager = self.memory_manager

        self.memory_manager.load_from_disk(ctx)

        system_prompt = build_react_system_prompt(
            self.skills,
            allow_reflect=self.enable_reflect,
        )

        for i in range(self.max_iters):
            ctx.iteration = i + 1

            user_prompt = build_react_loop_prompt(ctx)
            raw = self.llm.chat(system_prompt, user_prompt)

            if self.trace:
                self.trace.react_raw(
                    getattr(ctx, "trace_run", None),
                    iteration=i + 1,
                    raw=raw,
                )

            action = self._parse_action(ctx, raw)
            if action is None:
                ctx.add_event("error", error="Invalid LLM output")
                continue

            atype = action.get("type")

            if atype == "reasoning":
                self._handle_reasoning(ctx, action)
                continue

            if atype == "tool_call":
                self._execute_tool(ctx, action)

                # Hard stop if tool loop detected
                if getattr(ctx, "halt", False):
                    return ctx

                continue

            if atype == "reflect":
                self._handle_reflect(ctx, action)
                continue

            if atype == "final":
                self._handle_final(ctx, action, conf_threshold, max_retries)
                if ctx.final_answer is not None:
                    return ctx
                continue

            ctx.add_event("error", error="Unknown action type")
            continue

        ctx.add_event("error", error="Max iterations reached")
        return ctx

    def _handle_reasoning(self, ctx, action):
        content = action.get("content", "")

        ctx.add_event("reasoning", content=content)
        ctx.history.append({"action": action, "observation": None})

        self.hooks.on_reasoning(ctx, content)
        return ctx

    def _handle_reflect(self, ctx, action):
        if not self.enable_reflect:
            ctx.add_event("note", msg="reflect disabled")
            return

        note = self.hooks.on_reflect(ctx)
        if note:
            ctx.add_event("reflect", note=note)
            ctx.history.append({"action": action, "observation": {"note": note}})

    def _handle_final(
        self,
        ctx,
        action,
        conf_threshold,
        max_retries,
    ):
        draft = action.get("answer")

        if draft is None:
            ctx.add_event("error", error="Final answer missing")
            return ctx

        final_answer = str(draft)

        critique = None
        if self.enable_critique:
            critique = self.hooks.on_critique(ctx, draft)
            if critique:
                ctx.add_event("critique", critique=critique)

        if self.enable_revision and critique:
            revised = self.hooks.on_revise(ctx, draft, critique)
            if revised:
                ctx.add_event(
                    "revision",
                    from_answer=draft,
                    to_answer=revised,
                )
                final_answer = revised

        if self.enable_memory_extract:
            self.hooks.on_memory_extract(ctx)

        conf = None
        if self.enable_confidence:
            conf = self.hooks.on_confidence(ctx, final_answer)
            ctx.add_event(
                "confidence",
                score=conf.score,
                rationale=conf.rationale,
            )

        if conf and conf.score < conf_threshold:
            if ctx.retry_count < max_retries:
                ctx.retry_count += 1
                ctx.add_event(
                    "retry",
                    reason="low_confidence",
                    score=conf.score,
                )
                ctx.history.append(
                    {
                        "action": {
                            "type": "reasoning",
                            "content": "Re-evaluating previous answer due to low confidence.",
                        },
                        "observation": None,
                    }
                )
                return ctx

        if conf:
            if conf.score < conf_threshold:
                for e in getattr(ctx, "memory_entries", []):
                    self.memory_manager.decay_importance(e, 0.15)
            else:
                for e in getattr(ctx, "memory_entries", []):
                    self.memory_manager.bump_importance(e, 0.10)

        if conf and conf.score < 0.3:
            ctx.add_event("grounding_override", reason="very_low_confidence")
            for entry in reversed(ctx.tool_results):
                result = entry.get("result")
                if isinstance(result, dict) and "error" not in result:
                    ctx.final_answer = str(result)
                    break
        else:
            ctx.final_answer = str(final_answer)

        # capability gap detection
        if isinstance(ctx.final_answer, str):
            lower_answer = ctx.final_answer.lower()

            failure_markers = [
                "not possible",
                "cannot perform",
                "no tool available",
                "unsupported",
                "unable to",
            ]

            if any(marker in lower_answer for marker in failure_markers):
                self.hooks.on_skill_gap(
                    ctx,
                    {
                        "type": "capability_gap",
                        "reason": "Task could not be completed with available tools",
                        "task": ctx.task,
                        "final_answer": ctx.final_answer,
                    },
                )

        ctx.add_event("final", answer=ctx.final_answer)

        if self.trace:
            self.trace.react_final(
                getattr(ctx, "trace_run", None),
                answer=ctx.final_answer,
            )

        return ctx

    def _parse_action(self, ctx, raw: str) -> Optional[dict]:
        try:
            action = json.loads(raw)
        except Exception:
            ctx.add_event("error", error="Invalid JSON", raw=raw)
            return None

        if not isinstance(action, dict):
            ctx.add_event("error", error="Action must be JSON object", raw=raw)
            return None

        if "type" not in action:
            ctx.add_event("error", error="Action missing type", raw=raw)
            return None

        return action

    def _execute_tool(self, ctx, action: Dict[str, Any]):
        tool_name = action.get("tool")
        args = action.get("args", {}) or {}

        # Sanitize malformed keys like "days:"
        if isinstance(args, dict):
            args = {str(k).strip().rstrip(":"): v for k, v in args.items()}
        else:
            args = {}

        if not tool_name:
            ctx.add_event("error", error="tool_call missing tool name")
            return

        skill = self.skills.get(tool_name)
        if not skill:
            ctx.add_event("error", tool=tool_name, error="Unknown tool")
            self.hooks.on_skill_gap(
                ctx,
                {
                    "type": "missing_tool",
                    "tool": tool_name,
                    "error": "Unknown tool",
                    "suggested_signature": f"def {tool_name}(...) -> Dict[str, Any]:",
                    "reason": "LLM attempted to use a tool not registered",
                },
            )
            return

        # Detect repeated identical tool call in same iteration
        recent_calls = [
            tr
            for tr in ctx.tool_results
            if tr["tool"] == tool_name and tr["args"] == args
        ]

        if len(recent_calls) >= 2:
            self.hooks.on_skill_gap(
                ctx,
                {
                    "type": "tool_loop",
                    "tool": tool_name,
                    "reason": "Repeated identical tool calls without progress",
                    "task": ctx.task,
                },
            )

            ctx.final_answer = "Tool loop detected — missing higher-level capability."
            ctx.halt = True  # <- critical
            return

        cache_key = None
        if self.dedupe_tool_calls:
            cache_key = json.dumps(
                {"tool": tool_name, "args": args},
                sort_keys=True,
            )
            if cache_key in self._tool_call_cache:
                cached = self._tool_call_cache[cache_key]
                ctx.add_event(
                    "tool_start",
                    tool=tool_name,
                    args=args,
                    cached=True,
                )
                ctx.tool_results.append(
                    {
                        "iteration": ctx.iteration,
                        "tool": tool_name,
                        "args": args,
                        "result": cached,
                    }
                )
                ctx.history.append({"action": action, "observation": cached})
                ctx.add_event(
                    "tool_end",
                    tool=tool_name,
                    ok=True,
                    cached=True,
                )
                return

        ctx.add_event("tool_start", tool=tool_name, args=args)

        try:
            result = skill.fn(**args)
            ok = True
        except Exception as e:
            result = {"error": str(e)}
            ok = False
            ctx.add_event("error", tool=tool_name, error=str(e))
            self.hooks.on_skill_gap(
                ctx,
                {
                    "type": "tool_error",
                    "tool": tool_name,
                    "error": str(e),
                    "args": args,
                    "iteration": ctx.iteration,
                },
            )

        if cache_key and ok:
            self._tool_call_cache[cache_key] = result

        ctx.tool_results.append(
            {
                "iteration": ctx.iteration,
                "tool": tool_name,
                "args": args,
                "result": result,
            }
        )

        ctx.history.append({"action": action, "observation": result})

        ctx.add_event("tool_end", tool=tool_name, ok=ok)
