import inspect
import json


def build_router_prompt(skills, user_input: str) -> str:
    """
    Decide if input requires structured tool execution.
    """

    tool_blocks = []

    for name in skills.list():
        skill = skills.get(name)

        sig = ""
        if skill and skill.fn:
            sig = str(inspect.signature(skill.fn))

        block = {
            "name": skill.name,
            "signature": sig,
            "capability": skill.description,
            "limitations": skill.negative_description,
        }

        tool_blocks.append(block)

    return f"""
You are a routing classifier.

Your job:
Decide whether the user request requires structured tool execution
or can be answered conversationally.

Available Tools:
{json.dumps(tool_blocks, indent=2)}

User Input:
{user_input}

Return ONLY valid JSON in this format:

{{
  "mode": "structured" | "interactive",
  "confidence": 0.0-1.0,
  "reason": "short explanation"
}}

Rules:
- Choose "structured" ONLY if tool execution is necessary.
- Greetings and explanations are interactive.
- Mathematical computation is structured.
- Time lookup is structured.
- If uncertain, choose interactive.
- Do not call tools.
- Output JSON only.
""".strip()
