# laf/prompts/planner_system.py

def planner_system_prompt() -> str:
    return (
        "You are a planning assistant.\n"
        "You must output ONLY valid JSON matching the provided schema.\n"
        "No markdown. No backticks. No explanations."
    )
