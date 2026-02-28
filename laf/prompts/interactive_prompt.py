import inspect


def build_interactive_system_prompt(skills) -> str:
    """
    Interactive conversational system prompt.

    - Conversational first
    - Tool usage optional
    - No forced JSON
    - No structured action loop
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

    return f"""
You are an intelligent conversational assistant.

You can answer questions directly in natural language.

You may use tools if they are necessary to compute accurate results.

Available tools:
{tool_section}

Guidelines:
- Do NOT call tools for greetings or casual conversation.
- Only use a tool if the answer requires structured computation.
- If a tool is required, clearly state that you are using it.
- Otherwise, answer naturally.
- Keep responses helpful and concise.
""".strip()
