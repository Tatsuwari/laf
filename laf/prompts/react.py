import inspect
import json
from typing import Dict, Any


# =========================================================
# SYSTEM PROMPT
# =========================================================

def build_react_system_prompt(skills, allow_reflect: bool = True) -> str:
    """
    Build a structured ReAct system prompt with:
    - Positive tool capabilities
    - Negative tool constraints
    - No task-specific instructions
    """

    tool_blocks = []

    for name in skills.list():
        skill = skills.get(name)

        sig = ""
        if skill and skill.fn:
            sig = str(inspect.signature(skill.fn))

        block = f"- {skill.name}{sig}\n"
        block += f"  capability: {skill.description}\n"

        if getattr(skill, "negative_description", None):
            block += f"  limitations: {skill.negative_description}\n"

        tool_blocks.append(block)

    tool_section = "\n".join(tool_blocks)

    reflect_line = '- "reflect" : optional reflection step\n' if allow_reflect else ""

    allowed_names = ", ".join(skills.list())

    return f"""
You are a reasoning agent operating in a structured action loop.

You must select tools carefully based on their capabilities and limitations.

Available tools:
{tool_section}

Allowed action types (only these):
- "reasoning" : short, goal-directed step
- "tool_call" : call a tool directly
{reflect_line}- "final" : finish with an answer

Strict Rules:
- Respond with valid JSON ONLY.
- Never output markdown.
- Never output explanations outside JSON.
- The "tool" field MUST be exactly one of: {allowed_names}
- Never use "tool_call" as a tool name.
- Never invent new tool names.
- Respect tool limitations.
- If a tool cannot perform a requested operation, choose another tool or reason further.
- Keep reasoning concise.
- Do not repeat identical tool calls if the previous observation did not change.

Examples:

{{ "type": "reasoning", "content": "I need structured numeric computation." }}

{{ "type": "tool_call", "tool": "calculator", "args": {{ "expression": "2 + 2" }} }}

{{ "type": "final", "answer": "4" }}
""".strip()


# =========================================================
# LOOP PROMPT
# =========================================================

def build_react_loop_prompt(ctx, history_window: int = 6):
    """
    Build user loop prompt with:
    - Clean task
    - Memory injection
    - History replay
    """

    prompt = f"User Task:\n{ctx.task}\n\n"

    # -------------------------
    # Memory Injection
    # -------------------------

    if hasattr(ctx, "memory_manager") and ctx.memory_manager:
        selected = ctx.memory_manager.select(ctx, ctx.task, k=10)
        if selected:
            prompt += "Relevant Memory:\n"
            prompt += json.dumps(
                ctx.memory_manager.to_injection_block(selected),
                indent=2
            )
            prompt += "\n\n"

    # -------------------------
    # History
    # -------------------------

    if ctx.history:
        prompt += "History:\n"
        for step in ctx.history[-history_window:]:
            prompt += "Action:\n"
            prompt += json.dumps(step["action"], indent=2) + "\n"

            if step["observation"] is not None:
                prompt += "Observation:\n"
                prompt += json.dumps(step["observation"], indent=2) + "\n"

            prompt += "\n"

    prompt += "Choose the next action.\nReturn JSON only."

    return prompt