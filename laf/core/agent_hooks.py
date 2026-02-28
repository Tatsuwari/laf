# laf/core/agent_hooks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from laf.core.memory import MemoryEntry
from laf.core.trace_recorder import TraceRecorder


@dataclass
class ConfidenceResult:
    score: float
    rationale: str = ""
    retry: bool = False


class AgentHooks:
    """
    Default agent behavior hooks.

    These hooks are intentionally deterministic.
    No LLM calls happen here.
    """

    def __init__(self, trace: Optional[TraceRecorder] = None):
        self.trace = trace

    # --------------------------------------------------

    def on_reasoning(self, ctx, content: str) -> None:
        if self.trace:
            self.trace.reasoning(
                ctx.trace_run if hasattr(ctx, "trace_run") else None,
                iteration=getattr(ctx, "iteration", 0),
                content=content,
            )

    # --------------------------------------------------

    def on_reflect(self, ctx) -> Optional[str]:
        # default: no reflection
        return None

    # --------------------------------------------------

    def on_critique(self, ctx, draft_answer: str) -> Optional[str]:
        # default: no critique
        return None

    # --------------------------------------------------

    def on_revise(
        self,
        ctx,
        draft_answer: str,
        critique: str,
    ) -> Optional[str]:
        # default: no revision
        return None

    # --------------------------------------------------

    def on_skill_gap(
        self,
        ctx,
        last_error: Dict[str, Any],
    ) -> None:

        ctx.skill_gaps = getattr(ctx, "skill_gaps", [])
        ctx.skill_gaps.append(last_error)

        if self.trace:
            self.trace.skill_gap(
                ctx.trace_run if hasattr(ctx, "trace_run") else None,
                tool=last_error.get("tool"),
                error=last_error.get("error"),
                proposal=last_error,
            )

    # --------------------------------------------------

    def on_memory_extract(self, ctx) -> None:
        """
        Deterministic memory extraction.

        Rules:
        - Uses skill.memory_extractor if defined
        - Falls back to raw result
        - Assigns importance based on success
        - TTL comes from skill.memory_ttl
        """

        tool_results = getattr(ctx, "tool_results", None)
        if not tool_results:
            return

        mm = getattr(ctx, "memory_manager", None)
        if mm is None:
            return

        skills = getattr(ctx, "skills", None)

        for tr in tool_results:

            tool = tr.get("tool")
            args = tr.get("args", {}) or {}
            result = tr.get("result", {}) or {}

            ok = not (
                isinstance(result, dict) and result.get("ok") is False
            )

            tags = []
            ttl = 3600
            extractor = None

            if skills is not None:
                spec = skills.get(tool)
                if spec:
                    tags = list(spec.memory_tags or [])
                    ttl = (
                        spec.memory_ttl
                        if spec.memory_ttl is not None
                        else ttl
                    )
                    extractor = getattr(spec, "memory_extractor", None)

            payload = None

            if extractor:
                try:
                    payload = extractor(args, result)
                except Exception:
                    payload = None

            if not payload:
                payload = (
                    result
                    if isinstance(result, dict)
                    else {"result": result}
                )

            importance = 0.70 if ok else 0.15

            entry = MemoryEntry(
                tool=tool,
                data=payload,
                tags=tags,
                ttl_seconds=ttl,
                importance=importance,
                ok=ok,
            )

            mm.add(ctx, entry)

            if self.trace:
                self.trace.memory_add(
                    ctx.trace_run if hasattr(ctx, "trace_run") else None,
                    tool=tool,
                    importance=importance,
                    ok=ok,
                )

    # --------------------------------------------------

    def on_confidence(self, ctx, draft_answer: Any) -> ConfidenceResult:
        """
        Deterministic confidence scoring.

        Signals:
        - Tool usage
        - Error presence
        - Final answer length
        """

        tool_results = getattr(ctx, "tool_results", []) or []
        events = getattr(ctx, "events", []) or []

        had_tools = len(tool_results) > 0
        had_errors = any(e.kind == "error" for e in events)

        
        draft_str = "" if draft_answer is None else str(draft_answer)
        answer_len = len(draft_str)

        score = 0.2

        if had_tools and not had_errors:
            score = 0.85
        elif not had_errors:
            score = 0.55
        elif had_tools:
            score = 0.35
        else:
            score = 0.2

        # Short empty answers are penalized
        if answer_len < 3:
            score = min(score, 0.3)

        retry = score < 0.5

        result = ConfidenceResult(
            score=round(score, 3),
            rationale="deterministic",
            retry=retry,
        )

        if self.trace:
            self.trace.confidence(
                ctx.trace_run if hasattr(ctx, "trace_run") else None,
                score=result.score,
                rationale=result.rationale,
            )

        return result

    # TODO:
    # - Add per-skill confidence weighting
    # - Add answer verification against tool results
    # - Add multi-pass evaluator agent