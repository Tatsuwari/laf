import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import runpod

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

_MODEL_POOL: List[Dict[str, Any]] = []
_POOL_INDEX: int = 0


def _build_pool() -> None:
    global _MODEL_POOL, _POOL_INDEX
    if _MODEL_POOL:
        return

    cfg = SystemConfig()

    pool: List[Dict[str, Any]] = []
    for i in range(cfg.pool_size):
        print(f"[handler] Loading model instance {i}")
        llm = LLM(cfg)

        store = IntentStore(cfg.intent_store_path, cfg.embed_model)
        store.load()

        logger = IntentLogger(cfg.intent_log_path)
        router = HybridIntentRouter(llm=llm, store=store, cfg=cfg, logger=logger)

        tools = ToolRegistry()
        setup_builtins(tools)
        load_plugins(cfg.plugin_dir, tools)

        rag_store = InMemoryVectorStore(store.embedder)
        # Safe if missing; load_jsonl returns early if file doesn't exist.
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

        pool.append({"pipeline": pipeline, "metrics": {"runs": 0, "total_time": 0.0}})

    _MODEL_POOL = pool
    _POOL_INDEX = 0


def _next_instance() -> Dict[str, Any]:
    global _POOL_INDEX
    pool = _MODEL_POOL
    idx = _POOL_INDEX
    _POOL_INDEX = (idx + 1) % len(pool)
    return pool[idx]


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected input payload (serverless style):
    {
      "input": {
        "task": "...",
        "execute_tools": false,
        "use_rag": true,
        "reflect": false,
        "runs": 1
      }
    }
    """
    _build_pool()

    job_input = event.get("input", {}) or {}

    task = job_input.get("task")
    if not task or not isinstance(task, str):
        return {"error": "Missing required field: input.task (string)"}

    execute_tools = bool(job_input.get("execute_tools", False))
    use_rag = bool(job_input.get("use_rag", False))
    reflect = bool(job_input.get("reflect", False))
    runs = int(job_input.get("runs", 1))
    runs = max(1, runs)

    results = []
    for _ in range(runs):
        instance = _next_instance()
        pipeline = instance["pipeline"]

        start = time.time()
        out = pipeline.run(
            task,
            execute_tools=execute_tools,
            use_rag=use_rag,
            reflect=reflect,
        )
        dur = time.time() - start

        instance["metrics"]["runs"] += 1
        instance["metrics"]["total_time"] += dur

        out["performance"] = {"duration_sec": dur}
        results.append(out)

    return {
        "task": task,
        "runs_requested": runs,
        "results": results,
    }


def _local_test_if_present() -> bool:
    """
    If you mount a test_input.json into the container, this runs once and exits.
    """
    p = Path("test_input.json")
    if not p.exists():
        return False

    event = json.loads(p.read_text(encoding="utf-8"))
    print("[handler] Running local test_input.json ...")
    print(json.dumps(handler(event), indent=2)[:5000])
    return True


if __name__ == "__main__":
    if _local_test_if_present():
        raise SystemExit(0)

    # Start serverless worker
    runpod.serverless.start(
        {
            "handler": handler,
            # Keep it simple/safe by default
            "concurrency_modifier": lambda current: 1,
        }
    )