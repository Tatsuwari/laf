from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

from laf.config import SystemConfig
from laf.llm import LLM
from laf.planner import PlannerAgent
from laf.intents.store import IntentStore
from laf.intents.logger import IntentLogger
from laf.intents.router import HybridIntentRouter
from laf.plugins.registry import ToolRegistry
from laf.plugins.loader import load_plugins
from laf.plugins.builtins import setup_builtins
from laf.rag.store import InMemoryVectorStore, load_jsonl
from laf.agents.generator import GeneratorAgent
from laf.agents.critic import CriticAgent
from laf.pipeline import TaskPipeline

app = FastAPI(title="Laf API", version="0.1.0")

class TaskReq(BaseModel):
    task: str
    execute_tools: bool = False
    use_rag: bool = False
    reflect: bool = False

@app.on_event("startup")
def startup():
    cfg = SystemConfig()
    cfg.intent_store_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.plugin_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(cfg)

    # intent store
    store = IntentStore(cfg.intent_store_path, cfg.embed_model)
    store.load()
    if not store.intents:
        # seed some starter intents
        store.add_intent("research", "Gather information, constraints, requirements.", tool="manual_review")
        store.add_intent("planning", "Plan structure, layout, order of work.", tool="manual_review")
        store.add_intent("execution", "Perform concrete build/implementation steps.", tool="manual_review")
        store.add_intent("maintenance", "Monitor, maintain, iterate, fix issues.", tool="manual_review")
        store.save()

    logger = IntentLogger(cfg.intent_log_path)
    router = HybridIntentRouter(llm=llm, store=store, cfg=cfg, logger=logger)

    # tools/plugins
    tools = ToolRegistry()
    setup_builtins(tools)
    load_plugins(cfg.plugin_dir, tools)

    # RAG store
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

    app.state.cfg = cfg
    app.state.llm = llm
    app.state.intent_store = store
    app.state.router = router
    app.state.tools = tools
    app.state.rag_store = rag_store
    app.state.pipeline = pipeline

@app.get("/v1/health")
def health():
    return {"ok": True}

@app.get("/v1/intents")
def intents():
    store: IntentStore = app.state.intent_store
    return {
        "count": len(store.intents),
        "intents": [
            {"key": v.key, "description": v.description, "tool": v.tool, "examples": len(v.examples)}
            for v in store.intents.values()
        ]
    }

@app.get("/v1/plugins")
def plugins():
    tools: ToolRegistry = app.state.tools
    return {"tools": [{"name": t.name, "description": t.description} for t in tools.tools.values()]}

@app.post("/v1/plan")
def plan(req: TaskReq):
    pipeline: TaskPipeline = app.state.pipeline
    out = pipeline.run(req.task, execute_tools=False, use_rag=False, reflect=False)
    return {"task": req.task, "plan": out["plan"]}

@app.post("/v1/route")
def route(req: TaskReq):
    pipeline: TaskPipeline = app.state.pipeline
    out = pipeline.run(req.task, execute_tools=False, use_rag=False, reflect=False)
    return {"task": req.task, "plan": out["plan"], "routed_steps": out["routed_steps"]}

@app.post("/v1/run")
def run(req: TaskReq):
    pipeline: TaskPipeline = app.state.pipeline
    return pipeline.run(req.task, execute_tools=req.execute_tools, use_rag=req.use_rag, reflect=req.reflect)
