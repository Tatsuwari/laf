import re
from laf.plugins.registry import ToolSpec

def setup(registry):
    def text_stats(args):
        text = args.get("text") or ""
        # normalize whitespace
        stripped = text.strip()
        lines = stripped.splitlines() if stripped else []
        words = re.findall(r"\b\w+\b", stripped)
        chars = len(text)

        return {
            "ok": True,
            "chars": chars,
            "lines": len(lines),
            "words": len(words),
            "preview": stripped[:200],
        }

    registry.register(ToolSpec(
        name="text_stats",
        description="Return simple stats about provided text (chars/lines/words/preview).",
        fn=text_stats,
        params_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Input text"}
            },
            "required": ["text"]
        }
    ))