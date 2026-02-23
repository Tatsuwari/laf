from __future__ import annotations
from typing import Any, Dict, Optional

from .linear import LinearPlanIR
from .tree import TreePlanIR
from .dag import DagPlanIR


class PlanIRFactory:
    @staticmethod
    def from_plan_dict(plan: Dict[str, Any], preferred_format: Optional[str] = None):
        fmt = (plan.get('format') or plan.get('plan_format') or preferred_format or 'linear').lower().strip()
        if fmt == 'tree':
            return TreePlanIR(plan)
        if fmt == 'dag':
            return DagPlanIR(plan)
        return LinearPlanIR(plan)

    @staticmethod
    def from_planner_output(planner_output: Any, preferred_format: Optional[str] = None):
        '''
        Planner may return:
          - a dataclass Plan (legacy) -> convert to linear dict
          - a dict representing IR (linear/tree/dag)
        '''
        if isinstance(planner_output, dict):
            return PlanIRFactory.from_plan_dict(planner_output, preferred_format=preferred_format)

        # Legacy Plan dataclass: has .goal and .subtasks with id/description
        goal = getattr(planner_output, 'goal', '') or ''
        subtasks = getattr(planner_output, 'subtasks', []) or []
        steps = []
        for s in subtasks:
            sid = getattr(s, 'id', None)
            desc = getattr(s, 'description', '')
            steps.append({'id': sid, 'description': desc, 'type': 'step'})
        plan = {'format': 'linear', 'goal': str(goal), 'steps': steps}
        return PlanIRFactory.from_plan_dict(plan, preferred_format=preferred_format)