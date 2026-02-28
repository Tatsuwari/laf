# laf/core/planner.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from laf.llm.provider import LLM as LLMProvider
from laf.modes.config import GenConfig

from laf.prompts.planner_system import planner_system_prompt
from laf.prompts.planner_user import build_planner_user_prompt

from laf.core.trace_recorder import TraceRecorder


@dataclass
class PlannerResult:
    raw: str
    plan: Dict[str, Any]
    ok: bool
    error: Optional[str] = None


class Planner:
    """
    Deterministic JSON planner.

    - Strict structure
    - Modular prompts
    - Emits trace events
    - Retry tightening
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_steps: int = 8,
        trace: Optional[TraceRecorder] = None,
    ):
        self.llm = llm
        self.max_steps = max_steps
        self.trace = trace

        self.gen = GenConfig(
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            stop=["\n\n\n", "</s>", "<|im_end|>"],
        )

    # --------------------------------------------------

    def plan(
        self,
        task: str,
        skills: List[Dict[str, Any]],
        preferred_format: str = "linear",
        trace_run=None,
        retries: int = 2,
    ) -> PlannerResult:

        system_prompt = planner_system_prompt()

        user_prompt = build_planner_user_prompt(
            task=task,
            skills=skills,
            max_steps=self.max_steps,
        )

        if self.trace:
            self.trace.planner_start(
                trace_run,
                preferred_format=preferred_format,
                skill_count=len(skills),
                task=task,
            )

        raw = ""
        last_err: Optional[str] = None

        for attempt in range(retries + 1):

            raw = self.llm.chat(
                system=system_prompt,
                user=user_prompt,
                gen=self.gen,
            ) or ""

            if self.trace:
                self.trace.planner_raw(
                    trace_run,
                    attempt=attempt,
                    raw_preview=raw[:400],
                )

            parsed, err = self._extract_and_parse_json(raw)

            if parsed is not None:

                normalized = self._normalize(parsed)

                if self.trace:
                    self.trace.planner_ok(
                        trace_run,
                        plan=normalized,
                        attempt=attempt,
                    )

                return PlannerResult(
                    raw=raw,
                    plan=normalized,
                    ok=True,
                )

            last_err = err

            user_prompt += (
                "\n\nREMINDER:\n"
                "Return ONLY a valid JSON object.\n"
                "No text.\n"
                "No markdown.\n"
                "No backticks.\n"
                "No explanations."
            )

        fallback = self._fallback(task, last_err)

        if self.trace:
            self.trace.planner_fail(
                trace_run,
                error=last_err,
                fallback_plan=fallback,
            )

        return PlannerResult(
            raw=raw,
            plan=fallback,
            ok=False,
            error=last_err,
        )

    # --------------------------------------------------

    def _extract_and_parse_json(
        self,
        raw: str,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:

        s = (raw or "").strip()
        if not s:
            return None, "empty_output"

        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj, None
        except Exception:
            pass

        match = re.search(r"\{.*\}", s, re.DOTALL)
        if not match:
            return None, "no_json_object_found"

        block = match.group(0).strip()

        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj, None
        except Exception as e:
            return None, f"json_parse_error:{e}"

        return None, "unknown_parse_failure"

    # --------------------------------------------------

    def _normalize(
        self,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:

        # DAG support is future work
        fmt = plan.get("format") or "linear"
        if fmt not in ("linear", "tree", "dag"):
            fmt = "linear"

        goal = str(plan.get("goal") or "").strip() or "Untitled goal"

        steps = plan.get("steps")
        if not isinstance(steps, list):
            steps = []

        fixed_steps: List[Dict[str, Any]] = []

        for i, step in enumerate(steps[: self.max_steps], start=1):

            if not isinstance(step, dict):
                continue

            stype = step.get("type") or "manual_review"

            if stype not in ("tool_call", "manual_review"):
                stype = "manual_review"

            if stype == "tool_call":
                fixed_steps.append(
                    {
                        "id": str(step.get("id") or i),
                        "type": "tool_call",
                        "plugin": str(step.get("plugin") or ""),
                        "args": step.get("args")
                        if isinstance(step.get("args"), dict)
                        else {},
                        "description": str(step.get("description") or ""),
                    }
                )
            else:
                fixed_steps.append(
                    {
                        "id": str(step.get("id") or i),
                        "type": "manual_review",
                        "description": str(
                            step.get("description") or "Needs manual review"
                        ),
                    }
                )

        if not fixed_steps:
            fixed_steps = [
                {
                    "id": "1",
                    "type": "manual_review",
                    "description": "No steps produced.",
                }
            ]

        return {
            "format": "linear",
            "goal": goal,
            "steps": fixed_steps,
        }

    # --------------------------------------------------

    def _fallback(
        self,
        task: str,
        error: Optional[str],
    ) -> Dict[str, Any]:

        return {
            "format": "linear",
            "goal": task.strip(),
            "steps": [
                {
                    "id": "1",
                    "type": "manual_review",
                    "description": f"Planner failed: {error or 'unknown'}",
                }
            ],
        }

    # TODO:
    # - Validate template chaining syntax {{step_x.field}}
    # - Enforce negative_description constraint at validation layer
    # - Add DAG normalization once executor supports DAG