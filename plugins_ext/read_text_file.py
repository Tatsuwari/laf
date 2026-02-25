from pathlib import Path
from laf.plugins.registry import ToolSpec

def setup(registry):
    def read_text_file(args):
        rel_path = (args.get("path") or "").strip()
        max_chars = int(args.get("max_chars") or 20_000)

        if not rel_path:
            return {"ok": False, "error": "path is required"}

        # Lock to ./data by default to avoid arbitrary filesystem reads
        base = Path(args.get("base_dir") or "data").resolve()
        target = (base / rel_path).resolve()

        if base not in target.parents and target != base:
            return {"ok": False, "error": "path escapes base_dir", "base_dir": str(base)}

        if not target.exists():
            return {"ok": False, "error": "file not found", "path": str(target)}

        if target.is_dir():
            return {"ok": False, "error": "path is a directory", "path": str(target)}

        # Read as text (utf-8)
        try:
            txt = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"ok": False, "error": f"failed to read file: {e}", "path": str(target)}

        trimmed = txt[:max_chars]
        return {
            "ok": True,
            "path": str(target),
            "chars": len(txt),
            "returned_chars": len(trimmed),
            "text": trimmed,
            "truncated": len(txt) > len(trimmed),
        }

    registry.register(ToolSpec(
        name="read_text_file",
        description="Read a UTF-8 text file from a restricted base directory (default: ./data).",
        fn=read_text_file,
        params_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path inside base_dir, e.g. docs.jsonl"},
                "base_dir": {"type": "string", "description": "Base directory (default: data)", "default": "data"},
                "max_chars": {"type": "integer", "description": "Max characters to return", "default": 20000},
            },
            "required": ["path"]
        }
    ))