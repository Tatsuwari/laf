from .registry import ToolSpec, ToolRegistry

def setup_builtins(registry: ToolRegistry) -> None:
    def manual_review(args):
        return {"ok": True, "note": "manual_review", "args": args}

    def echo(args):
        return {"ok": True, "echo": args}

    registry.register(ToolSpec(
        name="manual_review",
        description="Fallback tool when no automation exists.",
        fn=manual_review
    ))
    registry.register(ToolSpec(
        name="echo",
        description="Debug tool that returns args.",
        fn=echo
    ))
