from laf.plugins.registry import ToolSpec

def setup(registry):
    def web_search(args):
        q = (args.get("text") or "").strip()
        top_k = int(args.get("top_k") or 5)

        # placeholder: integrate real search later
        results = [
            {"title": f"Stub result {i+1} for: {q}", "snippet": "Replace with real search integration."}
            for i in range(max(0, min(top_k, 10)))
        ]

        return {
            "ok": True,
            "query": q,
            "results": results,
            "note": "This is a stub tool. Replace results with real web search later."
        }

    registry.register(ToolSpec(
        name="web_search",
        description="Stub web search tool (replace with real implementation).",
        fn=web_search,
        params_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "How many results to return (max 10)", "default": 5},
            },
            "required": ["text"]
        }
    ))