# laf/skills/builtins/time_tool.py

from datetime import datetime
from typing import Dict, Any


def time_now(format: str = "iso") -> Dict[str, Any]:
    """
    Returns the current local time.

    format:
        - "iso" (default)
        - "human"
        - "timestamp"
    """

    now = datetime.now()

    if format == "iso":
        return {
            "format": "iso",
            "value": now.isoformat()
        }

    if format == "human":
        return {
            "format": "human",
            "value": now.strftime("%Y-%m-%d %H:%M:%S")
        }

    if format == "timestamp":
        return {
            "format": "timestamp",
            "value": int(now.timestamp())
        }

    raise ValueError("Unsupported format")


# -------------------------------------------------
# MEMORY EXTRACTOR
# -------------------------------------------------

def time_memory(args: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "time_queries": [
            {
                "format": result.get("format"),
                "value": result.get("value"),
            }
        ]
    }
