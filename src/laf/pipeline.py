from dataclasses import asdict
from typing import Dict, Any
from pathlib import Path

from .config import SystemConfig
from .llm import LLM
from .planner import PlannerAgent
from .intents.router import HybridIntentRouter
from .plugins.registry import ToolRegistry
from .rag.store import InMemoryVectorStore
from .rag.retriever import Retriever
from .policies import ExecutionPolicy
from .agents.generator import GeneratorAgent
from .agents.critic import CriticAgent
from .capabilities.catalog import build_capability_catalog, summarize_catalog
from .agents.planner_reviewer import PlannerReviewer
from .plan_ir.factory import PlanIRFactory
from .trace import Tracer


class TaskPipeline:

    def __init__(
        self,
        cfg: SystemConfig,
        llm: LLM,
        planner: PlannerAgent,
        router: HybridIntentRouter,
        tools: ToolRegistry,
        rag_store: InMemoryVectorStore,
        generator: GeneratorAgent,
        critic: CriticAgent
    ):
        self.cfg = cfg
        self.llm = llm
        self.planner = planner
        self.router = router
        self.tools = tools
        self.rag_store = rag_store
        self.retriever = Retriever(rag_store)
        self.generator = generator
        self.critic = critic
        self.planner_reviewer = PlannerReviewer(tools)

    # ============================================================
    # MAIN PIPELINE
    # ============================================================

    def run(
        self,
        task: str,
        execute_tools: bool = False,
        use_rag: bool = False,
        reflect: bool = False
    ) -> Dict[str, Any]:

        # --------------------------------------------------------
        # Execution Policy
        # --------------------------------------------------------

        policy = ExecutionPolicy(
            internal_only=self.cfg.internal_only,
            internet_available=True
        )

        tracer = Tracer(
            base_dir=getattr(self.cfg, "trace_dir", "data/traces"),
            enabled=getattr(self.cfg, "trace_enabled", True)
        )

        tracer.emit(
            "pipeline.start",
            task=task,
            execute_tools=execute_tools,
            use_rag=use_rag,
            reflect=reflect
        )

        # --------------------------------------------------------
        # Build Capability Catalog
        # --------------------------------------------------------

        full_catalog = build_capability_catalog(
            self.router.store,
            self.tools,
            include_examples=True
        )

        prompt_catalog = summarize_catalog(full_catalog, max_items=50)

        tracer.emit(
            "catalog.built",
            intents=len(full_catalog.get("intents", [])),
            plugins=len(full_catalog.get("plugins", []))
        )

        tracer.emit(
            "tools.snapshot",
            tool_keys=sorted(list(self.tools.tools.keys()))
        )

        # --------------------------------------------------------
        # Planning
        # --------------------------------------------------------

        plan_format = getattr(self.cfg, "plan_format", "linear")

        plan_obj = self.planner.plan(
            task,
            catalog=prompt_catalog,
            plan_format=plan_format
        )

        plan_ir = PlanIRFactory.from_planner_output(
            plan_obj,
            preferred_format=plan_format
        )

        plan_dict = plan_ir.to_dict()

        tracer.emit(
            "plan.ir",
            format=plan_ir.format(),
            goal=plan_ir.goal()
        )

        # --------------------------------------------------------
        # Review
        # --------------------------------------------------------

        self.planner_reviewer = PlannerReviewer(self.tools)

        review = self.planner_reviewer.review(
            plan_dict,
            catalog=full_catalog
        )

        tracer.emit(
            "planner.review",
            ok=review.ok,
            issues=[vars(i) for i in review.issues]
        )

        if not review.ok:
            trace_path = str(tracer.flush())
            return {
                "task": task,
                "trace": {"task_id": tracer.task_id, "path": trace_path},
                "plan": plan_dict,
                "plan_review": {
                    "ok": review.ok,
                    "issues": [vars(i) for i in review.issues]
                },
                "routed_steps": [],
                "tool_results": [],
                "rag_decision": {"use_rag": False, "reason": "plan_invalid"},
                "retrieved_docs": [],
                "answer": None
            }

        # --------------------------------------------------------
        # Execute Plan Units
        # --------------------------------------------------------

        routed = []
        tool_results = []

        for unit in plan_ir.iter_units():

            if unit.node_type == "tool_call":

                routed.append({
                    "id": unit.id,
                    "description": unit.description,
                    "type": "tool_call",
                    "tool": unit.tool
                })

                if execute_tools and unit.tool:
                    out = self.tools.run(
                        unit.tool,
                        unit.args or {},
                        tracer=tracer,
                        policy=policy
                    )
                    tool_results.append({
                        "step_id": unit.id,
                        "tool": unit.tool,
                        "result": out
                    })

        # --------------------------------------------------------
        # Deterministic RAG
        # --------------------------------------------------------

        retrieved_docs = []
        rag_decision = {
            "use_rag": False,
            "query": task,
            "top_k": 0,
            "reason": "disabled"
        }

        if use_rag:

            docs = self.retriever.retrieve(
                task,
                top_k=self.cfg.rag_top_k
            )

            if docs:
                max_score = max(d.score for d in docs)

                if max_score >= self.cfg.rag_score_threshold:

                    retrieved_docs = [d.__dict__ for d in docs]

                    rag_decision = {
                        "use_rag": True,
                        "query": task,
                        "top_k": self.cfg.rag_top_k,
                        "reason": f"score={max_score:.3f}"
                    }

                else:
                    rag_decision = {
                        "use_rag": False,
                        "query": task,
                        "top_k": self.cfg.rag_top_k,
                        "reason": f"low_score={max_score:.3f}"
                    }

            else:
                rag_decision = {
                    "use_rag": False,
                    "query": task,
                    "top_k": 0,
                    "reason": "no_docs"
                }

            tracer.emit("rag.decision", **rag_decision)

            # ----------------------------------------------------
            # AUTO FALLBACK → WEB SEARCH
            # ----------------------------------------------------

            if (
                not rag_decision["use_rag"]
                and policy.internet_available
                and self.tools.has("web_search")
                and not policy.internal_only
            ):

                tracer.emit("rag.fallback.web_search", reason=rag_decision["reason"])

                web_out = self.tools.run(
                    "web_search",
                    {"text": task},
                    tracer=tracer,
                    policy=policy
                )

                if web_out.get("ok"):

                    # Add web results into retrieved_docs
                    for r in web_out.get("results", []):
                        retrieved_docs.append({
                            "score": 1.0,
                            "text": f"{r.get('title','')} - {r.get('snippet','')}"
                        })

                    rag_decision = {
                        "use_rag": True,
                        "query": task,
                        "top_k": len(retrieved_docs),
                        "reason": "web_fallback"
                    }

        # --------------------------------------------------------
        # Generation
        # --------------------------------------------------------

        answer = None

        if use_rag or reflect or tool_results:

            tracer.emit("generator.start")

            draft = self.generator.generate(
                task=task,
                plan=plan_dict,
                tool_results=tool_results,
                retrieved=retrieved_docs
            )

            tracer.emit("generator.draft", preview=str(draft)[:400])

            if reflect:
                tracer.emit("critic.start")
                answer = self.critic.refine(task, draft)
                tracer.emit("critic.output", preview=str(answer)[:400])
            else:
                answer = draft

        tracer.emit("pipeline.end", ok=True)
        trace_path = str(tracer.flush())

        return {
            "task": task,
            "trace": {
                "task_id": tracer.task_id,
                "path": trace_path
            },
            "plan": plan_dict,
            "plan_review": {"ok": True, "issues": []},
            "routed_steps": routed,
            "tool_results": tool_results,
            "rag_decision": rag_decision,
            "retrieved_docs": retrieved_docs,
            "answer": answer
        }