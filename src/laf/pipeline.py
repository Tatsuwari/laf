from dataclasses import asdict
from typing import Dict, Any
from .config import SystemConfig
from .llm import LLM
from .planner import PlannerAgent
from .intents.router import HybridIntentRouter
from .plugins.registry import ToolRegistry
from .rag.store import InMemoryVectorStore
from .rag.retriever import Retriever
from .rag.router import RagRouter
from .agents.generator import GeneratorAgent
from .agents.critic import CriticAgent

from .capabilities.catalog import build_capability_catalog, summarize_catalog
from .agents.planner_reviewer import PlannerReviewer
from .plan_ir.factory import PlanIRFactory

from .trace import Tracer
from pathlib import Path


class TaskPipeline:
    def __init__(self,
                 cfg: SystemConfig,
                 llm: LLM,
                 planner: PlannerAgent,
                 router: HybridIntentRouter,
                 tools: ToolRegistry,
                 rag_store: InMemoryVectorStore,
                 generator: GeneratorAgent,
                 critic: CriticAgent):
        self.cfg = cfg
        self.llm = llm
        self.planner = planner
        self.router = router
        self.tools = tools

        self.rag_store = rag_store
        self.retriever = Retriever(rag_store)
        self.rag_router = RagRouter(llm=llm, mode="AUTO")

        self.generator = generator
        self.critic = critic

        self.planner_reviewer = PlannerReviewer(tools)

    def run(self, task: str, execute_tools: bool = False, use_rag: bool = False, reflect: bool = False) -> Dict[str, Any]:
        trace_enabled = getattr(self.cfg, 'trace_enabled', True)
        trace_dir = getattr(self.cfg, 'trace_dir', 'data/traces')
        tracer = Tracer(base_dir=trace_dir, enabled=trace_enabled)

        tracer.emit('pipeline.start', task=task, execute_tools=execute_tools, use_rag=use_rag, reflect=reflect)


        
        # catalog
        full_catalog = build_capability_catalog(self.router.store, self.tools, include_examples=True)
        prompt_catalog = summarize_catalog(full_catalog, max_items=50)
        tracer.emit('catalog.built', intents=len(full_catalog.get('intents', [])), plugins=len(full_catalog.get('plugins', [])))
        
        # Debugging aid: snapshot tool keys in tracer for easy reference
        tracer.emit("tools.snapshot", tool_keys=sorted(list(self.tools.tools.keys())))
        
        # plan_format should come from cfg; keep backward compatible default
        plan_format = getattr(self.cfg, 'plan_format', 'linear')

        # planner
        plan_obj = self.planner.plan(task, catalog=prompt_catalog, plan_format=plan_format)
        plan_ir = PlanIRFactory.from_planner_output(plan_obj, preferred_format=plan_format)
        plan_dict = plan_ir.to_dict()

        tracer.emit('plan.ir', format=plan_ir.format(), goal=plan_ir.goal())

        # review (always bind reviewer to current registry)
        self.planner_reviewer = PlannerReviewer(self.tools)
        review = self.planner_reviewer.review(plan_dict, catalog=full_catalog)

        tracer.emit('planner.review', ok=review.ok, issues=[ vars(i) for i in review.issues ])



        # If plan invalid, return early with review issues (manual intervention path)
        if not review.ok:
            trace_path = str(tracer.flush())
            return {
                "task": task,
                "tracer": {"task_id": tracer.task_id, "path": trace_path,"events": tracer.events()},
                "plan": plan_dict,
                "plan_review": {
                    "ok": review.ok,
                    "issues": [vars(i) for i in review.issues],
                },
                "routed_steps": [],
                "tool_results": [],
                "rag_decision": {"use_rag": False, "query": task, "top_k": 0, "reason": "plan_invalid"},
                "retrieved_docs": [],
                "answer": None
            }
        
        routed = []
        tool_results = []

        # 4) Execute plan units (linear/tree/dag all normalized into ExecUnit stream)
        for unit in plan_ir.iter_units():
                if unit.node_type == "step":
                    r = self.router.route(unit.description, tracer=tracer)
                    routed.append({"id": unit.id, "description": unit.description, "type": "step", **r})

                    if execute_tools:
                        args = {"text": unit.description, "intent": r["intent"], "tool": r["tool"]}
                        out = self.tools.run(r["tool"], args, tracer=tracer)
                        tool_results.append({"step_id": unit.id, "tool": r["tool"], "result": out})

                elif unit.node_type == "tool_call":
                    routed.append({"id": unit.id, "description": unit.description, "type": "tool_call", "tool": unit.tool})
                    if execute_tools and unit.tool:
                        out = self.tools.run(unit.tool, unit.args or {}, tracer=tracer)
                        tool_results.append({"step_id": unit.id, "tool": unit.tool, "result": out})

                elif unit.node_type == "manual_review":
                    tracer.emit("fallback.manual_review", step_id=unit.id, description=unit.description)
                    routed.append({"id": unit.id, "description": unit.description, "type": "manual_review", "tool": "manual_review"})

                else:
                    routed.append({"id": unit.id, "description": unit.description, "type": unit.node_type, "intent": unit.intent})

        # --- RAG FLOW (unchanged, but now uses plan_dict) ---
        retrieved_docs = []
        rag_decision = {"use_rag": False, "query": task, "top_k": 0, "reason": "disabled"}

        if use_rag:
            decision = self.rag_router.decide(task, plan=plan_dict)
            rag_decision = {
                "use_rag": decision.use_rag,
                "query": decision.query,
                "top_k": decision.top_k,
                "reason": decision.reason
            }
            tracer.emit('rag.decision', **rag_decision)

            if decision.use_rag:
                docs = self.retriever.retrieve(decision.query, top_k=decision.top_k)
                retrieved_docs = [d.__dict__ for d in docs]
                tracer.emit('rag.retrieve',top_k=decision.top_k, returned=len(retrieved_docs))
        # --- END RAG FLOW ---
        # generate/refine
        answer = None
        if use_rag or reflect:
            tracer.emit('generator.start')
            draft = self.generator.generate(task=task,plan=plan_dict,tool_results=tool_results,retrieved=retrieved_docs)
            tracer.emit('generator.draft',draft_preview=str(draft)[:500])

            if reflect:
                tracer.emit('critic.start')
                answer = self.critic.refine(task, draft)
                tracer.emit('critic.output',answer_preview=str(answer)[:500])
            else:
                answer = draft

        tracer.emit('pipeline.end', ok=True)
        trace_path = str(tracer.flush())

        return {
            "task": task,
            "trace": {"task_id": tracer.task_id, "path": trace_path,"events": tracer.events()},
            "plan": plan_dict,
            "plan_review": {
                "ok": True,
                "issues": []
            },
            "routed_steps": routed,
            "tool_results": tool_results,
            "rag_decision": rag_decision,
            "retrieved_docs": retrieved_docs,
            "answer": answer
        }