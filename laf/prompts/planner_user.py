# laf/prompts/planner_user.py

from typing import List, Dict
from .planner_schema import planner_schema


def build_planner_user_prompt(task: str, skills: List[Dict], max_steps: int) -> str:

    allowed_names = ", ".join([s["name"] for s in skills])

    tool_blocks = []

    for s in skills:
        params = ", ".join(s.get("params", [])) or "none"

        block = f"- {s['name']}({params})\n"
        block += f"  description: {s.get('description','')}\n"

        if s.get("negative_description"):
            block += f"  limitations: {s['negative_description']}\n"

        tool_blocks.append(block)

    tools_section = "\n".join(tool_blocks)

    return f"""
Return ONLY a valid JSON object.
No markdown.
No explanations.

Allowed plugin names:
{allowed_names}

Available tools:
{tools_section}

{planner_schema(max_steps)}

CRITICAL CHAINING RULE:
To use output from a previous step, you MUST use template syntax:
{{{{step_<id>.<field>}}}}

Task:
{task}
""".strip()
