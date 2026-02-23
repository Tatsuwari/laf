from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Iterator, List, Optional

from .base import ExecUnit, PlanIR, _norm_id


class LinearPlanIR:
    def __init__(self, plan: Dict[str, Any]):
        self._plan = plan

    def format(self) -> str:
        return 'linear'

    def goal(self) -> str:
        return str(self._plan.get('goal', '') or '')

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self._plan)
        out['format'] = 'linear'
        # normalize key name
        if 'steps' not in out and 'subtasks' in out:
            out['steps'] = out['subtasks']
        return out

    def iter_units(self) -> Iterator[ExecUnit]:
        steps = self._plan.get('steps') or self._plan.get('subtasks') or []
        if not isinstance(steps, list):
            return
        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                continue

            stype = s.get('type') or 'step'
            sid = _norm_id(s.get('id'), str(i + 1))

            if stype == 'tool_call':
                desc = str(s.get('description') or s.get('text') or '').strip()
                if not desc:
                    desc = f'Run tool {s.get('tool') or s.get('plugin')}'
                yield ExecUnit(
                    id=sid,
                    description=desc,
                    node_type='tool_call',
                    tool=str(s.get('tool') or s.get('plugin') or s.get('plugin_key') or ''),
                    args=s.get('args') if isinstance(s.get('args'), dict) else {},
                )
            elif stype == 'intent':
                desc = str(s.get('goal') or s.get('description') or s.get('text') or '').strip()
                yield ExecUnit(
                    id=sid,
                    description=desc,
                    node_type='intent',
                    intent=str(s.get('intent') or s.get('intent_key') or s.get('key') or ''),
                )
            elif stype == 'manual_review':
                desc = str(s.get('question') or s.get('description') or 'Manual review required').strip()
                yield ExecUnit(id=sid, description=desc, node_type='manual_review')
            else:
                desc = str(s.get('description') or s.get('text') or '').strip()
                if not desc:
                    continue
                yield ExecUnit(id=sid, description=desc, node_type='step')