from laf.plugins.registry import ToolSpec

def setup(registry):
    def web_search(args):
        # placeholder: integrate real search later
        q = args.get("text", "")
        return {
            "ok": True,
            "query": q,
            "results": [
                {"title": "Example result 1", "snippet": "Replace with real search."},
                {"title": "Example result 2", "snippet": "Use your web/RAG tool here."}
            ]
        }

    registry.register(ToolSpec(
        name="web_search",
        description="Stub web search tool (replace with real implementation).",
        fn=web_search,
        params_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        }
    ))
