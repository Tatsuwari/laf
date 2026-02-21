import json, ast
from typing import Any, Optional, List

def extract_braced_blocks(text: str) -> List[str]:
    blocks = []
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    blocks.append(text[start:i+1])
                    start = None
    return blocks

def safe_parse_struct(text: str) -> Optional[Any]:
    # Prefer parsing last {...} block
    blocks = extract_braced_blocks(text)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except Exception:
            pass

    # fallback: whole text
    try:
        return json.loads(text)
    except Exception:
        pass

    # fallback: python dict-style
    try:
        return ast.literal_eval(text)
    except Exception:
        return None
