import os
import json
import time
import zipfile
import threading
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from laf.config import SystemConfig
from laf.llm import LLM
from laf.planner import PlannerAgent
from laf.intents.store import IntentStore
from laf.intents.logger import IntentLogger
from laf.intents.router import HybridIntentRouter
from laf.plugins.registry import ToolRegistry, ToolSpec
from laf.plugins.loader import load_plugins
from laf.plugins.builtins import setup_builtins
from laf.rag.store import InMemoryVectorStore, load_jsonl, Document
from laf.agents.generator import GeneratorAgent
from laf.agents.critic import CriticAgent
from laf.pipeline import TaskPipeline


# --------------------------------------------------
# Global state (similar to app.state in FastAPI)
# --------------------------------------------------

_STATE_LOCK = threading.Lock()
_POOL_INDEX_LOCK = threading.Lock()

_MODEL_POOL: List[Dict[str, Any]] = []
_POOL_INDEX: int = 0
_CFG: Optional[SystemConfig] = None


# --------------------------------------------------
# Startup / pool management
# --------------------------------------------------

def _build_instance(cfg: SystemConfig) -> Dict[str, Any]:
    llm = LLM(cfg)

    store = IntentStore(cfg.intent_store_path, cfg.embed_model)
    store.load()

    logger = IntentLogger(cfg.intent_log_path)
    router = HybridIntentRouter(llm=llm, store=store, cfg=cfg, logger=logger)

    tools = ToolRegistry()
    setup_builtins(tools)
    load_plugins(cfg.plugin_dir, tools)

    rag_store = InMemoryVectorStore(store.embedder)
    load_jsonl(rag_store, Path("data/docs.jsonl"))

    pipeline = TaskPipeline(
        cfg=cfg,
        llm=llm,
        planner=PlannerAgent(llm, cfg),
        router=router,
        tools=tools,
        rag_store=rag_store,
        generator=GeneratorAgent(llm),
        critic=CriticAgent(llm),
    )

    return {
        "pipeline": pipeline,
        "metrics": {"runs": 0, "total_time": 0.0},
    }


def startup_once() -> None:
    global _MODEL_POOL, _POOL_INDEX, _CFG

    with _STATE_LOCK:
        if _MODEL_POOL:
            return

        cfg = SystemConfig()
        _CFG = cfg

        model_pool: List[Dict[str, Any]] = []
        for i in range(cfg.pool_size):
            print(f"[gradio] Loading model instance {i}")
            model_pool.append(_build_instance(cfg))

        _MODEL_POOL = model_pool
        _POOL_INDEX = 0
        print(f"[gradio] Startup complete. Instances: {len(_MODEL_POOL)}")


def reset_pool() -> None:
    global _MODEL_POOL, _POOL_INDEX, _CFG
    with _STATE_LOCK:
        _MODEL_POOL = []
        _POOL_INDEX = 0
        _CFG = None
    startup_once()


def get_next_instance() -> Dict[str, Any]:
    global _POOL_INDEX
    if not _MODEL_POOL:
        startup_once()

    with _POOL_INDEX_LOCK:
        idx = _POOL_INDEX
        _POOL_INDEX = (_POOL_INDEX + 1) % len(_MODEL_POOL)
        return _MODEL_POOL[idx]


def get_any_pipeline() -> TaskPipeline:
    if not _MODEL_POOL:
        startup_once()
    return _MODEL_POOL[0]["pipeline"]


# --------------------------------------------------
# Helpers (health / metrics / artifacts / listings)
# --------------------------------------------------

def health() -> Dict[str, Any]:
    if not _MODEL_POOL:
        startup_once()
    return {"ok": True, "instances": len(_MODEL_POOL)}


def metrics() -> List[Dict[str, Any]]:
    if not _MODEL_POOL:
        startup_once()
    out = []
    for i, p in enumerate(_MODEL_POOL):
        out.append(
            {
                "instance": i,
                "runs": p["metrics"]["runs"],
                "total_time_sec": p["metrics"]["total_time"],
            }
        )
    return out


def list_tools() -> Dict[str, Any]:
    pipe = get_any_pipeline()
    tools = pipe.tools.tools  # name -> ToolSpec
    items = []
    for _, spec in sorted(tools.items(), key=lambda x: x[0]):
        items.append(
            {
                "name": spec.name,
                "description": spec.description,
                "params_schema": spec.params_schema,
            }
        )
    return {"count": len(items), "tools": items}


def list_intents(query: str = "") -> Dict[str, Any]:
    pipe = get_any_pipeline()
    store = pipe.router.store
    q = (query or "").strip().lower()

    intents = []
    for k, rec in store.intents.items():
        if q and (q not in k.lower()) and (q not in (rec.description or "").lower()):
            continue
        intents.append(
            {
                "key": rec.key,
                "description": rec.description,
                "tool": rec.tool,
                "category_path": rec.category_path,
                "tags": rec.tags,
                "examples_count": len(rec.examples or []),
            }
        )

    intents.sort(key=lambda x: x["key"])
    return {"count": len(intents), "intents": intents}


def _parse_cat_path(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return ["misc"]
    s = s.replace("\\", "/")
    parts = [p.strip() for p in s.split("/") if p.strip()]
    return parts[:4]  # allow "deeper dungeon" levels


def build_dungeon_tree() -> Dict[str, Any]:
    pipe = get_any_pipeline()
    store = pipe.router.store

    root: Dict[str, Any] = {"_count": 0, "_keys": [], "_children": {}}

    for k, rec in store.intents.items():
        path = rec.category_path or ["misc"]
        node = root
        node["_count"] += 1
        for p in path:
            ch = node["_children"].setdefault(p, {"_count": 0, "_keys": [], "_children": {}})
            ch["_count"] += 1
            node = ch
        node["_keys"].append(k)

    def render(node: Dict[str, Any], prefix: str = "") -> str:
        out = []
        for name, child in sorted(node["_children"].items(), key=lambda x: x[0]):
            out.append(f"{prefix}- **{name}** ({child['_count']})")
            if child["_keys"]:
                keys = ", ".join(sorted(child["_keys"])[:25])
                more = "" if len(child["_keys"]) <= 25 else f" …(+{len(child['_keys'])-25})"
                out.append(f"{prefix}  - {keys}{more}")
            rendered = render(child, prefix + "  ")
            if rendered:
                out.append(rendered)
        return "\n".join(out)

    return {"count": root["_count"], "markdown": render(root) or "_No intents yet._"}


def trace_root() -> Path:
    pipe = get_any_pipeline()
    base = Path(getattr(pipe.cfg, "trace_dir", "data/traces"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def list_traces(limit: int = 200) -> Dict[str, Any]:
    base = trace_root()
    task_dirs = [p for p in base.iterdir() if p.is_dir()]
    task_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    task_dirs = task_dirs[: max(1, int(limit))]

    items = []
    for d in task_dirs:
        trace_file = d / "trace.jsonl"
        items.append(
            {
                "task_id": d.name,
                "has_trace": trace_file.exists(),
                "modified_ts": int(d.stat().st_mtime),
            }
        )
    return {"count": len(items), "traces": items}


def read_trace(task_id: str) -> Dict[str, Any]:
    base = trace_root()
    d = base / task_id
    if not d.exists():
        return {"ok": False, "error": "task_id not found", "task_id": task_id}

    trace_file = d / "trace.jsonl"
    if not trace_file.exists():
        return {"ok": False, "error": "trace.jsonl missing", "task_id": task_id}

    events = []
    for line in trace_file.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except Exception:
            events.append({"raw": line})

    return {"ok": True, "task_id": task_id, "events": events}


def zip_trace(task_id: str) -> Tuple[Optional[str], Optional[str]]:
    base = trace_root()
    trace_dir = base / task_id
    if not trace_dir.exists():
        return None, "Trace not found."

    zip_path = trace_dir.parent / f"{task_id}.zip"
    try:
        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in trace_dir.glob("*"):
                if file.is_file():
                    zf.write(file, arcname=file.name)
        return str(zip_path), None
    except Exception as e:
        return None, f"Failed to create zip: {e}"


def upload_plugin(file_obj) -> Dict[str, Any]:
    if not file_obj:
        return {"ok": False, "error": "No file uploaded."}

    if not _CFG:
        startup_once()

    src_path = Path(file_obj.name)
    if src_path.suffix.lower() != ".py":
        return {"ok": False, "error": "Only .py files are supported."}

    plugin_dir = Path(_CFG.plugin_dir)
    plugin_dir.mkdir(parents=True, exist_ok=True)

    dst_path = plugin_dir / src_path.name
    shutil.copyfile(src_path, dst_path)

    # BEST: rebuild ALL instances so tools are consistent everywhere
    reset_pool()

    for inst in _MODEL_POOL:
        pipe: TaskPipeline = inst["pipeline"]
        pipe.tools.clear()
        setup_builtins(pipe.tools)
        load_plugins(_CFG.plugin_dir, pipe.tools)

    return {"ok": True, "saved_to": str(dst_path), "tools": list_tools()}


def ingest_docs_jsonl(file_path: str) -> Dict[str, Any]:
    if not file_path:
        return {"ok": False, "error": "No file uploaded."}

    src_path = Path(file_path)
    lines = src_path.read_text(encoding="utf-8").splitlines()
    added = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)

        doc = Document(
            id=str(obj["id"]),
            text=str(obj["text"]),
            meta=obj.get("meta", {}) or {},
        )

        for inst in _MODEL_POOL:
            pipe: TaskPipeline = inst["pipeline"]
            pipe.rag_store.add(doc)

        added += 1

    pipe0 = get_any_pipeline()

    return {
        "ok": True,
        "added": added,
        "docs_total_instance0": len(pipe0.rag_store.docs)
    }


def rag_stats() -> Dict[str, Any]:
    pipe = get_any_pipeline()
    return {"docs_total_instance0": len(pipe.rag_store.docs)}


# --------------------------------------------------
# Intent Ops (merge / promote) — store-only changes (no tool changes)
# --------------------------------------------------

def _ensure_store_ops(store: IntentStore) -> None:
    """
    If you haven't added these methods into IntentStore yet, this will fail.
    Keeping helper here so UI errors are clearer.
    """
    required = ["promote_intent", "merge_intents", "save"]
    missing = [m for m in required if not hasattr(store, m)]
    if missing:
        raise NotImplementedError(f"IntentStore is missing required methods: {missing}")


def promote_intent_ui(intent_key: str, category_path: str, tool: str, tags_csv: str, description: str) -> Dict[str, Any]:
    pipe = get_any_pipeline()
    store = pipe.router.store
    _ensure_store_ops(store)

    key = (intent_key or "").strip().lower()
    if not key:
        return {"ok": False, "error": "intent_key required"}
    if key not in store.intents:
        return {"ok": False, "error": "intent not found", "intent_key": key}

    cat = _parse_cat_path(category_path)
    tool = (tool or "").strip() or store.intents[key].tool
    tags = [t.strip() for t in (tags_csv or "").split(",") if t.strip()]
    desc = (description or "").strip() or store.intents[key].description

    ok = store.promote_intent(
        key,
        category_path=cat,
        tool=tool,
        tags=tags if tags else store.intents[key].tags,
        description=desc,
    )
    if not ok:
        return {"ok": False, "error": "promote failed"}

    store.save()
    return {"ok": True, "intent": key, "updated": list_intents(key), "dungeon": build_dungeon_tree()}


def merge_intents_ui(target_key: str, sources_csv: str, delete_sources: bool) -> Dict[str, Any]:
    pipe = get_any_pipeline()
    store = pipe.router.store
    _ensure_store_ops(store)

    tgt = (target_key or "").strip().lower()
    sources = [s.strip().lower() for s in (sources_csv or "").split(",") if s.strip()]
    if not tgt:
        return {"ok": False, "error": "target_key required"}
    if not sources:
        return {"ok": False, "error": "at least one source intent required"}

    res = store.merge_intents(tgt, sources, delete_sources=bool(delete_sources))
    if res.get("ok"):
        store.save()
        res["dungeon"] = build_dungeon_tree()
        res["target_after"] = list_intents(tgt)
    return res


# --------------------------------------------------
# Core run logic
# --------------------------------------------------

def run_task(
    task: str,
    execute_tools: bool,
    use_rag: bool,
    reflect: bool,
    runs: int,
) -> Dict[str, Any]:
    if not task or not isinstance(task, str):
        return {"error": "task must be a non-empty string"}
    if runs < 1:
        return {"error": "runs must be >= 1"}

    if not _MODEL_POOL:
        startup_once()

    results = []
    for _ in range(runs):
        instance = get_next_instance()
        pipeline: TaskPipeline = instance["pipeline"]

        start = time.time()
        result = pipeline.run(
            task,
            execute_tools=execute_tools,
            use_rag=use_rag,
            reflect=reflect,
        )
        duration = time.time() - start

        instance["metrics"]["runs"] += 1
        instance["metrics"]["total_time"] += duration

        if isinstance(result, dict):
            result["performance"] = {"duration_sec": duration}
        else:
            result = {"output": result, "performance": {"duration_sec": duration}}

        results.append(result)

    return {"task": task, "runs_requested": runs, "results": results}


def prettify(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# --------------------------------------------------
# Gradio UI
# --------------------------------------------------

def build_demo() -> gr.Blocks:
    startup_once()

    with gr.Blocks(title="Laf UI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Laf — Gradio UI\n"
            "Interactive runner for your pipeline (model pool, intents, plugins, RAG, traces, intent ops)."
        )

        with gr.Tabs():

            # ----------------------------
            # Run Tab
            # ----------------------------
            with gr.Tab("Run"):
                task_in = gr.Textbox(
                    label="Task",
                    placeholder="Describe what you want Laf to do...",
                    lines=6,
                )
                with gr.Row():
                    execute_tools_in = gr.Checkbox(False, label="execute_tools")
                    use_rag_in = gr.Checkbox(False, label="use_rag")
                    reflect_in = gr.Checkbox(False, label="reflect")
                    runs_in = gr.Slider(1, 10, value=1, step=1, label="runs")

                run_btn = gr.Button("Run", variant="primary")
                out_json = gr.JSON(label="Result JSON")
                out_text = gr.Textbox(label="Raw JSON (copy-friendly)", lines=18)

                def on_run(task, execute_tools, use_rag, reflect, runs):
                    out = run_task(task, execute_tools, use_rag, reflect, int(runs))
                    return out, prettify(out)

                run_btn.click(
                    fn=on_run,
                    inputs=[task_in, execute_tools_in, use_rag_in, reflect_in, runs_in],
                    outputs=[out_json, out_text],
                )

            # ----------------------------
            # System Tab
            # ----------------------------
            with gr.Tab("System"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", variant="secondary")
                    reset_btn = gr.Button("Hard Reset Pool", variant="stop")

                health_json = gr.JSON(label="Health")
                metrics_json = gr.JSON(label="Metrics")
                rag_json = gr.JSON(label="RAG Stats (instance 0)")

                def on_refresh():
                    return health(), metrics(), rag_stats()

                def on_reset():
                    reset_pool()
                    return on_refresh()

                refresh_btn.click(fn=on_refresh, inputs=[], outputs=[health_json, metrics_json, rag_json])
                reset_btn.click(fn=on_reset, inputs=[], outputs=[health_json, metrics_json, rag_json])

                demo.load(fn=on_refresh, inputs=[], outputs=[health_json, metrics_json, rag_json])

            # ----------------------------
            # Plugins Tab
            # ----------------------------
            with gr.Tab("Plugins"):
                gr.Markdown("Manage tools (builtins + plugins). Upload `.py` plugins into your plugin directory and reload.")

                tools_json = gr.JSON(label="Tools")
                tools_refresh_btn = gr.Button("Refresh Tools", variant="secondary")

                plugin_upload = gr.File(label="Upload plugin (.py)", file_types=[".py"])
                plugin_upload_btn = gr.Button("Save + Reload Plugins", variant="primary")
                plugin_upload_out = gr.JSON(label="Upload Result")

                tools_refresh_btn.click(fn=list_tools, inputs=[], outputs=[tools_json])
                plugin_upload_btn.click(fn=upload_plugin, inputs=[plugin_upload], outputs=[plugin_upload_out])

                demo.load(fn=list_tools, inputs=[], outputs=[tools_json])

            # ----------------------------
            # Intents Tab
            # ----------------------------
            with gr.Tab("Intents"):
                gr.Markdown("Inspect the intent store. Search by key or description.")

                q_in = gr.Textbox(label="Search", placeholder="e.g. 'search' or 'computer_vision'")
                intents_btn = gr.Button("Refresh", variant="secondary")
                intents_json = gr.JSON(label="Intents")

                intents_btn.click(fn=lambda q: list_intents(q or ""), inputs=[q_in], outputs=[intents_json])
                demo.load(fn=lambda: list_intents(""), inputs=[], outputs=[intents_json])

            # ----------------------------
            # Intent Dungeon Tab (Promote / Merge)
            # ----------------------------
            with gr.Tab("Intent Dungeon"):
                gr.Markdown("Organize and deduplicate your self-growing intent catalog: promote intents into deep modules and merge near-duplicates.")

                dungeon_md = gr.Markdown()
                dungeon_refresh_btn = gr.Button("Refresh Dungeon", variant="secondary")

                dungeon_refresh_btn.click(fn=lambda: build_dungeon_tree()["markdown"], inputs=[], outputs=[dungeon_md])
                demo.load(fn=lambda: build_dungeon_tree()["markdown"], inputs=[], outputs=[dungeon_md])

                gr.Markdown("## Promote / Move Intent (Catalog → Deep Module)")
                with gr.Row():
                    promote_key = gr.Textbox(label="intent_key", placeholder="e.g. search_web")
                    promote_cat = gr.Textbox(label="category_path", placeholder="e.g. nlp/search/web")
                with gr.Row():
                    promote_tool = gr.Textbox(label="tool (optional)", placeholder="e.g. web_search")
                    promote_tags = gr.Textbox(label="tags (comma)", placeholder="e.g. search,web,rag")
                promote_desc = gr.Textbox(label="description (optional)", lines=2)

                promote_btn = gr.Button("Promote", variant="primary")
                promote_out = gr.JSON(label="Promote Result")

                promote_btn.click(
                    fn=promote_intent_ui,
                    inputs=[promote_key, promote_cat, promote_tool, promote_tags, promote_desc],
                    outputs=[promote_out],
                )

                gr.Markdown("## Merge Intents (Deduplicate)")
                merge_target = gr.Textbox(label="target_intent_key (keep)", placeholder="e.g. search_web")
                merge_sources = gr.Textbox(label="source_intent_keys (comma)", placeholder="e.g. web_lookup, websearch, search_online")
                merge_delete = gr.Checkbox(True, label="delete merged sources")

                merge_btn = gr.Button("Merge", variant="primary")
                merge_out = gr.JSON(label="Merge Result")

                merge_btn.click(
                    fn=merge_intents_ui,
                    inputs=[merge_target, merge_sources, merge_delete],
                    outputs=[merge_out],
                )

            # ----------------------------
            # Traces Tab
            # ----------------------------
            with gr.Tab("Traces"):
                gr.Markdown("View trace ids (all) or open one by id. Download ZIP bundles.")

                with gr.Row():
                    trace_limit = gr.Slider(20, 500, value=200, step=10, label="List limit")
                    trace_list_btn = gr.Button("List Traces", variant="secondary")

                traces_json = gr.JSON(label="Trace List")

                with gr.Row():
                    task_id_in = gr.Textbox(label="task_id", placeholder="Paste a task_id here (e.g. task_abc123...)")
                    trace_open_btn = gr.Button("Open Trace", variant="primary")

                trace_view = gr.JSON(label="Trace (events)")

                with gr.Row():
                    dl_btn = gr.Button("Create ZIP & Download", variant="primary")
                    dl_file = gr.File(label="ZIP file")
                    dl_err = gr.Markdown()

                trace_list_btn.click(fn=lambda lim: list_traces(int(lim)), inputs=[trace_limit], outputs=[traces_json])
                trace_open_btn.click(fn=lambda tid: read_trace((tid or "").strip()), inputs=[task_id_in], outputs=[trace_view])

                def on_download(task_id: str):
                    if not (task_id or "").strip():
                        return None, "Provide a task_id first."
                    zip_path, err = zip_trace(task_id.strip())
                    if err:
                        return None, f"**Error:** {err}"
                    return zip_path, f"Created: `{Path(zip_path).name}`"

                dl_btn.click(fn=on_download, inputs=[task_id_in], outputs=[dl_file, dl_err])

                demo.load(fn=lambda: list_traces(200), inputs=[], outputs=[traces_json])

            # ----------------------------
            # Docs Tab
            # ----------------------------
            with gr.Tab("Docs"):
                gr.Markdown("Upload JSONL docs to the in-memory RAG store (all instances).")

                docs_stats = gr.JSON(label="RAG Stats (instance 0)")
                docs_refresh_btn = gr.Button("Refresh", variant="secondary")

                docs_upload = gr.File(label="Upload docs (.jsonl)", file_types=[".jsonl"], type='filepath')
                docs_upload_btn = gr.Button("Ingest docs.jsonl", variant="primary")
                docs_upload_out = gr.JSON(label="Ingest Result")

                docs_refresh_btn.click(fn=rag_stats, inputs=[], outputs=[docs_stats])
                docs_upload_btn.click(fn=ingest_docs_jsonl, inputs=[docs_upload], outputs=[docs_upload_out])

                demo.load(fn=rag_stats, inputs=[], outputs=[docs_stats])

    return demo


def main():
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    demo = build_demo()
    # demo.queue(concurrency_count=max(1, len(_MODEL_POOL)))
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()