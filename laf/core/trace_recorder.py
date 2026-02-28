# laf/core/trace_recorder.py
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _now() -> float:
    return time.time()


def _safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


@dataclass
class TraceEvent:
    ts: float
    kind: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRun:
    run_id: str
    task_id: str
    task: str
    meta: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=_now)
    ended_at: Optional[float] = None
    ok: Optional[bool] = None

    events: List[TraceEvent] = field(default_factory=list)

    final_answer: Optional[Any] = None
    error: Optional[str] = None

    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    planner: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    confidence: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, Any]] = None


class TraceRecorder:
    """
    Structured run tracer.
    Writes one JSON line per run.

    The API is intentionally tolerant:
    - accepts both old and new keyword names
    - ignores unknown kwargs to avoid breaking on refactors
    """

    def __init__(
        self,
        enabled: bool = True,
        out_dir: str = ".laf_traces",
        filename: str = "runs.jsonl",
        max_event_payload: int = 20_000,
    ):
        self.enabled = enabled
        self.out_dir = out_dir
        self.filename = filename
        self.max_event_payload = max_event_payload

        self._runs: List[TraceRun] = []

        if self.enabled:
            os.makedirs(self.out_dir, exist_ok=True)

    def start_run(
        self,
        task: str,
        task_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[TraceRun]:
        if not self.enabled:
            return None

        run = TraceRun(
            run_id=str(uuid.uuid4()),
            task_id=task_id or str(uuid.uuid4()),
            task=task,
            meta=_safe_json(meta or {}),
        )

        self._runs.append(run)

        self._emit(
            run,
            "run_start",
            {"task": task, "task_id": run.task_id, "meta": run.meta},
        )

        return run

    def finish_run(
        self,
        run: Optional[TraceRun],
        ok: bool = True,
        final_answer: Any = None,
        error: Optional[str] = None,
        ctx: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or run is None:
            return

        run.ended_at = _now()
        run.ok = bool(ok)
        run.final_answer = _safe_json(final_answer)
        run.error = error

        if ctx is not None:
            try:
                run.tool_calls = _safe_json(getattr(ctx, "tool_results", []))
            except Exception:
                pass

            try:
                mm = getattr(ctx, "memory_manager", None)
                if mm and hasattr(mm, "snapshot_for_trace"):
                    run.memory = _safe_json(mm.snapshot_for_trace(ctx))
            except Exception:
                pass

        if extra:
            self._emit(run, "run_extra", {"extra": _safe_json(extra)})

        self._emit(run, "run_end", {"ok": run.ok, "error": run.error})
        self._flush_run(run)

    def planner_start(
        self,
        run: Optional[TraceRun],
        preferred_format: str,
        skills_count: Optional[int] = None,
        skill_count: Optional[int] = None,
        **extra: Any,
    ) -> None:
        count = skills_count if skills_count is not None else (skill_count or 0)
        self._emit(
            run,
            "planner_start",
            {"preferred_format": preferred_format, "skills_count": int(count), **_safe_json(extra)},
        )

    def planner_raw(
        self,
        run: Optional[TraceRun],
        attempt: int,
        raw_preview: str,
        **_: Any,
    ) -> None:
        self._emit(run, "planner_raw", {"attempt": int(attempt), "raw_preview": raw_preview[:400]})

    def planner_ok(
        self,
        run: Optional[TraceRun],
        plan: Dict[str, Any],
        attempt: Optional[int] = None,
        **_: Any,
    ) -> None:
        if run is not None:
            run.planner = {"ok": True, "plan": _safe_json(plan), "attempt": attempt}
        self._emit(run, "planner_ok", {"plan": _safe_json(plan), "attempt": attempt})

    def planner_fail(
        self,
        run: Optional[TraceRun],
        error: Optional[str],
        fallback_plan: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        if run is not None:
            run.planner = {"ok": False, "error": error, "fallback_plan": _safe_json(fallback_plan or {})}
        self._emit(run, "planner_fail", {"error": error, "fallback_plan": _safe_json(fallback_plan or {})})

    def validation_result(
        self,
        run: Optional[TraceRun],
        ok: bool,
        errors: List[Dict[str, Any]],
        **_: Any,
    ) -> None:
        if run is not None:
            run.validation = {"ok": bool(ok), "errors": _safe_json(errors)}
        self._emit(run, "validation", {"ok": bool(ok), "errors": _safe_json(errors)})

    def skill_gap(
        self,
        run: Optional[TraceRun],
        tool: str,
        error: str,
        proposal: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        self._emit(run, "skill_gap", {"tool": tool, "error": error, "proposal": _safe_json(proposal or {})})

    def react_raw(
        self,
        run: Optional[TraceRun],
        iteration: int,
        raw: str,
        **_: Any,
    ) -> None:
        self._emit(run, "react_raw", {"iteration": int(iteration), "raw_preview": raw[:600]})

    def llm_raw(self, run: Optional[TraceRun], iteration: int, raw: str, **kwargs: Any) -> None:
        self.react_raw(run, iteration, raw, **kwargs)

    def react_final(self, run: Optional[TraceRun], answer: Any, **_: Any) -> None:
        self._emit(run, "react_final", {"answer": _safe_json(answer)})

    def tool_start(
        self,
        run: Optional[TraceRun],
        iteration: int,
        tool: str,
        args: Dict[str, Any],
        cached: bool = False,
        **_: Any,
    ) -> None:
        self._emit(
            run,
            "tool_start",
            {"iteration": int(iteration), "tool": tool, "args": _safe_json(args), "cached": bool(cached)},
        )

    def tool_end(
        self,
        run: Optional[TraceRun],
        iteration: int,
        tool: str,
        ok: bool,
        result: Any = None,
        cached: bool = False,
        **_: Any,
    ) -> None:
        self._emit(
            run,
            "tool_end",
            {
                "iteration": int(iteration),
                "tool": tool,
                "ok": bool(ok),
                "result": _safe_json(result),
                "cached": bool(cached),
            },
        )

    def confidence(self, run: Optional[TraceRun], score: float, rationale: str, **_: Any) -> None:
        if run is not None:
            run.confidence = {"score": float(score), "rationale": rationale}
        self._emit(run, "confidence", {"score": float(score), "rationale": rationale})

    def retry(
        self,
        run: Optional[TraceRun],
        reason: str,
        attempt: Optional[int] = None,
        **extra: Any,
    ) -> None:
        self._emit(run, "retry", {"reason": reason, "attempt": attempt, **_safe_json(extra)})

    def note(self, run: Optional[TraceRun], msg: str, data: Optional[Dict[str, Any]] = None, **_: Any) -> None:
        payload: Dict[str, Any] = {"msg": msg}
        if data:
            payload["data"] = _safe_json(data)
        self._emit(run, "note", payload)

    def memory_add(
        self,
        run: Optional[TraceRun],
        tool: str,
        importance: Optional[float] = None,
        ok: Optional[bool] = None,
        **_: Any,
    ) -> None:
        self._emit(
            run,
            "memory_add",
            {"tool": tool, "importance": float(importance or 0.0), "ok": bool(ok) if ok is not None else None},
        )

    def memory_prune(
        self,
        run: Optional[TraceRun],
        removed: Optional[int] = None,
        removed_count: Optional[int] = None,
        **_: Any,
    ) -> None:
        count = removed_count if removed_count is not None else removed
        if count is None:
            count = 0
        self._emit(run, "memory_prune", {"removed_count": int(count)})

    def memory_select(
        self,
        run: Optional[TraceRun],
        selected_count: Optional[int] = None,
        count: Optional[int] = None,
        **_: Any,
    ) -> None:
        n = selected_count if selected_count is not None else (count or 0)
        self._emit(run, "memory_select", {"selected_count": int(n)})

    def memory_load(self, run: Optional[TraceRun], count: Optional[int] = None, path: Optional[str] = None, **_: Any) -> None:
        self._emit(run, "memory_load", {"count": int(count or 0), "path": path})

    def memory_save(self, run: Optional[TraceRun], count: Optional[int] = None, path: Optional[str] = None, **_: Any) -> None:
        self._emit(run, "memory_save", {"count": int(count or 0), "path": path})

    def memory_decay(self, run: Optional[TraceRun], key: str, amount: float, **_: Any) -> None:
        self._emit(run, "memory_decay", {"key": key, "amount": float(amount)})

    def memory_bump(self, run: Optional[TraceRun], key: str, amount: float, **_: Any) -> None:
        self._emit(run, "memory_bump", {"key": key, "amount": float(amount)})

    def _emit(self, run: Optional[TraceRun], kind: str, data: Dict[str, Any]) -> None:
        if not self.enabled or run is None:
            return

        safe = _safe_json(data)

        try:
            raw = json.dumps(safe)
            if len(raw) > self.max_event_payload:
                safe = {"truncated": True, "kind": kind, "size": len(raw)}
        except Exception:
            safe = {"truncated": True, "kind": kind, "note": "unserializable"}

        run.events.append(TraceEvent(ts=_now(), kind=kind, data=safe))

    def _flush_run(self, run: TraceRun) -> None:
        path = os.path.join(self.out_dir, self.filename)
        os.makedirs(self.out_dir, exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self._run_to_dict(run), ensure_ascii=False) + "\n")

    def _run_to_dict(self, run: TraceRun) -> Dict[str, Any]:
        return {
            "run_id": run.run_id,
            "task_id": run.task_id,
            "task": run.task,
            "meta": _safe_json(run.meta),
            "started_at": run.started_at,
            "ended_at": run.ended_at,
            "ok": run.ok,
            "final_answer": _safe_json(run.final_answer),
            "error": run.error,
            "planner": _safe_json(run.planner),
            "validation": _safe_json(run.validation),
            "confidence": _safe_json(run.confidence),
            "memory": _safe_json(run.memory),
            "tool_calls": _safe_json(run.tool_calls),
            "events": [{"ts": e.ts, "kind": e.kind, "data": _safe_json(e.data)} for e in run.events],
        }