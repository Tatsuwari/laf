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

    def run(self, task: str, execute_tools: bool = False, use_rag: bool = False, reflect: bool = False) -> Dict[str, Any]:
        plan_obj = self.planner.plan(task)
        plan = {"goal": plan_obj.goal, "subtasks": [asdict(s) for s in plan_obj.subtasks]}

        routed = []
        tool_results = []

        for step in plan["subtasks"]:
            desc = step["description"]
            r = self.router.route(desc)
            routed.append({**step, **r})

            if execute_tools:
                args = {"text": desc, "intent": r["intent"], "tool": r["tool"]}
                tool_results.append({
                    "step_id": step["id"],
                    "tool": r["tool"],
                    "result": self.tools.run(r["tool"], args)
                })

        # --- NEW RAG FLOW ---
        retrieved_docs = []
        rag_decision = {"use_rag": False, "query": task, "top_k": 0, "reason": "disabled"}

        if use_rag:
            decision = self.rag_router.decide(task, plan=plan)
            rag_decision = {
                "use_rag": decision.use_rag,
                "query": decision.query,
                "top_k": decision.top_k,
                "reason": decision.reason
            }

            if decision.use_rag:
                docs = self.retriever.retrieve(decision.query, top_k=decision.top_k)
                retrieved_docs = [d.__dict__ for d in docs]
        # --- END NEW RAG FLOW ---

        answer = None
        if use_rag or reflect:
            draft = self.generator.generate(task, plan, retrieved_docs)
            answer = self.critic.refine(task, draft) if reflect else draft

        return {
            "task": task,
            "plan": plan,
            "routed_steps": routed,
            "tool_results": tool_results,
            "rag_decision": rag_decision,     # NEW
            "retrieved_docs": retrieved_docs,
            "answer": answer
        }