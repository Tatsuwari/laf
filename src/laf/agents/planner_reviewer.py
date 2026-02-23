from __future__ import annotations
from typing import Any, Dict, Optional

from ..plugins.registry import ToolRegistry
from ..planner_review import PlanValidator, ReviewResult


class PlannerReviewer:
    '''
    Deterministic reviewer. (Later you can add an LLM-based reviewer too,
    but this one is the hard gate.)
    '''

    def __init__(self, tools: ToolRegistry):
        self.validator = PlanValidator(tools)

    def review(self, plan_ir: Dict[str, Any], catalog: Optional[Dict[str, Any]] = None) -> ReviewResult:
        return self.validator.validate(plan_ir=plan_ir, catalog=catalog)