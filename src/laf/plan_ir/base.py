from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol


@dataclass
class ExecUnit:
    '''
    Common executable unit used by pipeline regardless of plan shape.
    '''
    id: str
    description: str
    node_type: str = 'step'  # step | intent | tool_call | manual_review
    intent: Optional[str] = None
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None


class PlanIR(Protocol):
    def format(self) -> str: ...
    def goal(self) -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...
    def iter_units(self) -> Iterator[ExecUnit]: ...


def _norm_id(x: Any, fallback: str) -> str:
    if x is None:
        return fallback
    s = str(x).strip()
    return s if s else fallback