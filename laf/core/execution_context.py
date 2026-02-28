# laf/core/execution_context.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class ExecutionEvent:
    kind: str
    step_id: Optional[str] = None
    tool: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class ExecutionContext:
    """
    Holds execution state for a single task.

    This object is intentionally lightweight and deterministic.
    It does not perform logging itself — TraceRecorder handles persistence.
    """

    task: str
    goal: str
    plan: Optional[Dict[str, Any]] = None

    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    events: List[ExecutionEvent] = field(default_factory=list)

    skill_gaps: List[Dict[str, Any]] = field(default_factory=list)

    memory: Dict[str, Any] = field(default_factory=dict)

    history: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    iteration: int = 0

    started_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    finished_at_ms: Optional[int] = None

    def add_event(
        self,
        kind: str,
        step_id: Optional[str] = None,
        tool: Optional[str] = None,
        **data: Any,
    ) -> None:
        self.events.append(
            ExecutionEvent(
                kind=kind,
                step_id=step_id,
                tool=tool,
                data=dict(data),
            )
        )

    def remember(self, key: str, value: Any) -> None:
        self.memory[key] = value

    def snapshot(self) -> Dict[str, Any]:
        """
        Returns a safe serializable snapshot of execution state.
        Useful for tracing or debugging.
        """
        return {
            "task": self.task,
            "goal": self.goal,
            "plan": self.plan,
            "tool_results": self.tool_results,
            "skill_gaps": self.skill_gaps,
            "memory_keys": list(self.memory.keys()),
            "final_answer": self.final_answer,
            "iteration": self.iteration,
            "started_at_ms": self.started_at_ms,
            "finished_at_ms": self.finished_at_ms,
        }

    def finish(self) -> None:
        self.finished_at_ms = int(time.time() * 1000)

    # TODO:
    # - Add step-level timing map {step_id: duration_ms}
    # - Add memory size tracking for token budgeting
    # - Add structured error categorization