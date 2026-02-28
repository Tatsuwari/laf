# laf/prompts/planner_schema.py

def planner_schema(max_steps: int) -> str:
    return f"""
Schema:

{{
  "format": "linear",
  "goal": "string",
  "steps": [
    {{
      "id": "1",
      "type": "tool_call",
      "plugin": "one_of_allowed_plugins",
      "args": {{}},
      "description": "what this step does"
    }}
  ]
}}

Rules:
- Output format must be "linear".
- type must be "tool_call" or "manual_review".
- plugin must exactly match one of the allowed plugin names.
- All required parameters must be provided.
- Maximum steps: {max_steps}
- If no suitable plugin exists, return a single manual_review step.
"""
