from typing import Dict, Any, Optional, List

from laf.skills.registry import SkillRegistry


class PlanValidator:
    """
    Validates planner output against the registered SkillRegistry.

    Responsibilities:
    - Enforce valid step structure
    - Ensure tool existence
    - Convert invalid tool calls to manual_review
    - Detect skill gaps
    - Produce deterministic validated plan
    """

    VALID_TYPES = {"tool_call", "reasoning", "manual_review"}

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def validate(
        self,
        draft: Dict[str, Any],
        trace=None,
        trace_run=None,
    ) -> Dict[str, Any]:
        """
        Validate a planner draft and return normalized structure:

        {
            "goal": str,
            "steps": [...],
            "skill_gaps": [...]
        }
        """

        if not isinstance(draft, dict):
            return self._emit_fallback(
                goal="Invalid draft",
                error="draft_not_dict",
                trace=trace,
                trace_run=trace_run,
            )

        goal = str(draft.get("goal") or "Unknown goal").strip()

        steps = draft.get("steps")
        if not isinstance(steps, list) or not steps:
            return self._emit_fallback(
                goal=goal,
                error="empty_or_invalid_steps",
                trace=trace,
                trace_run=trace_run,
            )

        validated_steps: List[Dict[str, Any]] = []
        skill_gaps: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        for i, step in enumerate(steps, start=1):

            step_id = str(i)

            if not isinstance(step, dict):
                errors.append(
                    {
                        "step_id": step_id,
                        "error": "step_not_dict",
                    }
                )
                validated_steps.append(
                    {
                        "id": step_id,
                        "type": "manual_review",
                        "description": "Invalid step structure",
                    }
                )
                continue

            step_type = step.get("type", "manual_review")
            if step_type not in self.VALID_TYPES:
                step_type = "manual_review"

            description = str(step.get("description") or "").strip()

            if step_type == "tool_call":

                plugin = step.get("plugin")
                args = step.get("args", {})

                if not plugin or not self.registry.has(plugin):

                    gap = {
                        "missing_plugin": plugin,
                        "goal": goal,
                        "step_description": description,
                    }

                    skill_gaps.append(gap)

                    errors.append(
                        {
                            "step_id": step_id,
                            "error": "unknown_skill",
                            "plugin": plugin,
                        }
                    )

                    if trace:
                        trace.skill_gap(
                            trace_run,
                            tool=str(plugin),
                            error="Unknown skill",
                            proposal=gap,
                        )

                    validated_steps.append(
                        {
                            "id": step_id,
                            "type": "manual_review",
                            "description": f"Unknown skill: {plugin}",
                        }
                    )

                    continue

                validated_steps.append(
                    {
                        "id": step_id,
                        "type": "tool_call",
                        "plugin": plugin,
                        "args": args if isinstance(args, dict) else {},
                        "description": description,
                    }
                )

                continue

            validated_steps.append(
                {
                    "id": step_id,
                    "type": step_type,
                    "description": description,
                }
            )

        result = {
            "goal": goal,
            "steps": validated_steps,
            "skill_gaps": skill_gaps,
        }

        if trace:
            trace.validation_result(
                trace_run,
                ok=len(errors) == 0,
                errors=errors,
            )

        return result

    def _emit_fallback(
        self,
        goal: str,
        error: str,
        trace=None,
        trace_run=None,
    ) -> Dict[str, Any]:

        if trace:
            trace.validation_result(
                trace_run,
                ok=False,
                errors=[{"error": error}],
            )

        return {
            "goal": goal,
            "steps": [
                {
                    "id": "1",
                    "type": "manual_review",
                    "description": "Invalid or empty plan",
                }
            ],
            "skill_gaps": [],
        }

    # TODO:
    # - Validate tool argument schema against SkillSpec signature
    # - Enforce negative tool constraints
    # - Validate template references ({{step_x.field}})
    # - Add DAG dependency validation once DAG execution is implemented