from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import zipfile
import time

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

app = FastAPI(title="Laf API", version="0.3.0")

# --------------------------------------------------
# Request Model
# --------------------------------------------------

class TaskReq(BaseModel):
    task: str
    execute_tools: bool = False
    use_rag: bool = False
    reflect: bool = False
    runs: int = 1  # number of times to execute

# --------------------------------------------------
# Startup
# --------------------------------------------------

@app.on_event("startup")
def startup():
    cfg = SystemConfig()

    model_pool = []

    for i in range(cfg.pool_size):
        print(f"Loading model instance {i}")

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

        model_pool.append({
            "pipeline": pipeline,
            "metrics": {
                "runs": 0,
                "total_time": 0.0
            }
        })

    app.state.model_pool = model_pool
    app.state.pool_index = 0  # round robin pointer

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_next_instance():
    pool = app.state.model_pool
    idx = app.state.pool_index
    app.state.pool_index = (idx + 1) % len(pool)
    return pool[idx]

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/v1/system/health")
def health():
    return {
        "ok": True,
        "instances": len(app.state.model_pool)
    }

@app.get("/v1/system/metrics")
def metrics():
    pool = app.state.model_pool
    return [
        {
            "instance": i,
            "runs": p["metrics"]["runs"],
            "total_time_sec": p["metrics"]["total_time"]
        }
        for i, p in enumerate(pool)
    ]

@app.post("/v1/run")
def run(req: TaskReq):

    if req.runs < 1:
        raise HTTPException(status_code=400, detail="runs must be >= 1")

    results = []

    for _ in range(req.runs):
        instance = get_next_instance()
        pipeline = instance["pipeline"]

        start = time.time()

        result = pipeline.run(
            req.task,
            execute_tools=req.execute_tools,
            use_rag=req.use_rag,
            reflect=req.reflect
        )

        duration = time.time() - start

        instance["metrics"]["runs"] += 1
        instance["metrics"]["total_time"] += duration

        result["performance"] = {
            "duration_sec": duration
        }

        results.append(result)

    return {
        "task": req.task,
        "runs_requested": req.runs,
        "results": results
    }

@app.get("/v1/artifacts/{task_id}")
def download_trace(task_id: str):
    trace_dir = Path("data/traces") / task_id

    if not trace_dir.exists():
        raise HTTPException(status_code=404, detail="Trace not found")

    zip_path = trace_dir.parent / f"{task_id}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in trace_dir.glob("*"):
            zf.write(file, arcname=file.name)

    return FileResponse(zip_path, media_type="application/zip", filename=f"{task_id}.zip")