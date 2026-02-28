from typing import Dict, Any


def text_stats(text: str = "") -> Dict[str, Any]:
    text = text or ""

    return {
        "chars": len(text),
        "words": len(text.split()),
        "lines": text.count("\n") + 1 if text else 1,
    }


# MEMORY EXTRACTOR

def text_stats_memory(args: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    text = args.get("text", "")

    return {
        "text_stats": {
            text: {
                "chars": result.get("chars"),
                "words": result.get("words"),
                "lines": result.get("lines"),
            }
        }
    }