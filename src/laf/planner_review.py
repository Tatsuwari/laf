from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .plugins.registry import ToolRegistry


@dataclass
class ReviewIssue:
    severity: str  # 'ERROR' | 'WARN'
    code: str
    message: str
    node_id: Optional[str] = None


@dataclass
class ReviewResult:
    ok: bool
    issues: List[ReviewIssue] = field(default_factory=list)
    patched_plan: Optional[Dict[str, Any]] = None


def _is_dict(x: Any) -> bool:
    return isinstance(x, dict)


def _is_list(x: Any) -> bool:
    return isinstance(x, list)


def _collect_plugin_keys(catalog: Dict[str, Any]) -> set[str]:
    plugins = catalog.get('plugins', []) if _is_dict(catalog) else []
    keys = set()
    for p in plugins:
        if isinstance(p, dict) and p.get('key'):
            keys.add(str(p['key']))
    return keys


def _collect_intent_keys(catalog: Dict[str, Any]) -> set[str]:
    intents = catalog.get('intents', []) if _is_dict(catalog) else []
    keys = set()
    for i in intents:
        if isinstance(i, dict) and i.get('key'):
            keys.add(str(i['key']))
    return keys


class PlanValidator:
    '''
    Hard validator for any PlanIR dict emitted by PlannerAgent.
    This is *not* an LLM agent; deterministic checks only.
    '''

    def __init__(self, tools: ToolRegistry):
        self.tools = tools

    def validate(
        self,
        plan_ir: Dict[str, Any],
        catalog: Optional[Dict[str, Any]] = None,
    ) -> ReviewResult:
        issues: List[ReviewIssue] = []

        if not _is_dict(plan_ir):
            return ReviewResult(ok=False, issues=[ReviewIssue('ERROR', 'plan.not_dict', 'Plan must be a JSON object')])

        fmt = plan_ir.get('format') or plan_ir.get('plan_format') or 'linear'
        if fmt not in ('linear', 'tree', 'dag'):
            issues.append(ReviewIssue('ERROR', 'plan.bad_format', f'Unsupported plan format: {fmt!r}'))

        # collect known keys from catalog if present
        known_plugins = _collect_plugin_keys(catalog or {})
        known_intents = _collect_intent_keys(catalog or {})

        # validate by format
        if fmt == 'linear':
            self._validate_linear(plan_ir, issues, known_plugins, known_intents)
        elif fmt == 'tree':
            self._validate_tree(plan_ir, issues, known_plugins, known_intents)
        else:
            self._validate_dag(plan_ir, issues, known_plugins, known_intents)

        ok = all(i.severity != 'ERROR' for i in issues)
        return ReviewResult(ok=ok, issues=issues, patched_plan=None)

    def _validate_tool_call(
        self,
        node: Dict[str, Any],
        issues: List[ReviewIssue],
        known_plugins: set[str],
    ) -> None:
        nid = str(node.get('id') or '')
        plugin = node.get('plugin') or node.get('plugin_key') or node.get('tool')
        if not plugin:
            issues.append(ReviewIssue('ERROR', 'tool.missing', 'tool_call missing plugin/tool name', node_id=nid))
            return

        plugin = str(plugin)
        if known_plugins and plugin not in known_plugins:
            issues.append(ReviewIssue('ERROR', 'tool.unknown_catalog', f'Tool not in catalog: {plugin}', node_id=nid))

        if not self.tools.has(plugin):
            issues.append(ReviewIssue('ERROR', 'tool.not_registered', f'Tool not registered: {plugin}', node_id=nid))

        args = node.get('args', {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            issues.append(ReviewIssue('ERROR', 'tool.args_not_object', 'tool_call args must be an object', node_id=nid))

        # optional schema validation: ToolRegistry will validate at runtime too,
        # but checking early helps planning quality.
        if self.tools.has(plugin):
            spec = self.tools.tools[plugin]
            if spec.params_schema:
                ok, err = self.tools.validate_args(spec.params_schema, args)
                if not ok:
                    issues.append(ReviewIssue('ERROR', 'tool.args_schema', f'{plugin} args schema invalid: {err}', node_id=nid))

    def _validate_intent_node(
        self,
        node: Dict[str, Any],
        issues: List[ReviewIssue],
        known_intents: set[str],
    ) -> None:
        nid = str(node.get('id') or '')
        key = node.get('intent') or node.get('intent_key') or node.get('key')
        if not key:
            issues.append(ReviewIssue('WARN', 'intent.missing', 'intent node missing intent key', node_id=nid))
            return
        key = str(key)
        if known_intents and key not in known_intents:
            issues.append(ReviewIssue('WARN', 'intent.unknown_catalog', f'Intent not in catalog: {key}', node_id=nid))

    def _validate_linear(
        self,
        plan: Dict[str, Any],
        issues: List[ReviewIssue],
        known_plugins: set[str],
        known_intents: set[str],
    ) -> None:
        if 'goal' not in plan:
            issues.append(ReviewIssue('ERROR', 'plan.goal_missing', "Plan missing 'goal'"))

        steps = plan.get('steps') or plan.get('subtasks')
        if not isinstance(steps, list):
            issues.append(ReviewIssue('ERROR', 'plan.steps_missing', "Linear plan must have 'steps' (list)"))
            return

        for idx, s in enumerate(steps):
            if not isinstance(s, dict):
                issues.append(ReviewIssue('ERROR', 'plan.step_not_object', f'Step[{idx}] must be an object'))
                continue

            # either it is a simple step, or already a tool_call spec
            stype = s.get('type') or 'step'
            if stype == 'tool_call':
                self._validate_tool_call(s, issues, known_plugins)
            elif stype == 'intent':
                self._validate_intent_node(s, issues, known_intents)
            else:
                # basic step must have description
                desc = s.get('description') or s.get('text')
                if not desc or not str(desc).strip():
                    issues.append(ReviewIssue('ERROR', 'step.desc_missing', f'Step[{idx}] missing description', node_id=str(s.get('id') or idx)))

    def _validate_tree(
        self,
        plan: Dict[str, Any],
        issues: List[ReviewIssue],
        known_plugins: set[str],
        known_intents: set[str],
    ) -> None:
        root = plan.get('root') or plan.get('plan_tree')
        if not isinstance(root, dict):
            issues.append(ReviewIssue('ERROR', 'tree.root_missing', "Tree plan must have 'root' object"))
            return
        self._walk_tree(root, issues, known_plugins, known_intents)

    def _walk_tree(
        self,
        node: Dict[str, Any],
        issues: List[ReviewIssue],
        known_plugins: set[str],
        known_intents: set[str],
    ) -> None:
        if not isinstance(node, dict):
            issues.append(ReviewIssue('ERROR', 'tree.node_not_object', 'Tree node must be object'))
            return

        ntype = node.get('type') or 'category'
        if ntype == 'tool_call':
            self._validate_tool_call(node, issues, known_plugins)
        elif ntype == 'intent':
            self._validate_intent_node(node, issues, known_intents)

        children = node.get('children', [])
        if children is None:
            return
        if not isinstance(children, list):
            issues.append(ReviewIssue('ERROR', 'tree.children_not_list', 'Tree node children must be list', node_id=str(node.get('id') or '')))
            return
        for c in children:
            if isinstance(c, dict):
                self._walk_tree(c, issues, known_plugins, known_intents)
            else:
                issues.append(ReviewIssue('ERROR', 'tree.child_not_object', 'Tree child must be object', node_id=str(node.get('id') or '')))

    def _validate_dag(
        self,
        plan: Dict[str, Any],
        issues: List[ReviewIssue],
        known_plugins: set[str],
        known_intents: set[str],
    ) -> None:
        nodes = plan.get('nodes')
        edges = plan.get('edges')

        if not isinstance(nodes, list) or not nodes:
            issues.append(ReviewIssue('ERROR', 'dag.nodes_missing', "DAG plan must have non-empty 'nodes' list"))
            return
        if not isinstance(edges, list):
            issues.append(ReviewIssue('ERROR', 'dag.edges_missing', "DAG plan must have 'edges' list"))
            edges = []

        node_ids = set()
        for n in nodes:
            if not isinstance(n, dict):
                issues.append(ReviewIssue('ERROR', 'dag.node_not_object', "DAG node must be object"))
                continue
            nid = n.get('id')
            if not nid:
                issues.append(ReviewIssue('ERROR', 'dag.node_id_missing', "DAG node missing id"))
                continue
            node_ids.add(str(nid))

            ntype = n.get('type')
            if ntype == 'tool_call':
                self._validate_tool_call(n, issues, known_plugins)
            elif ntype == 'intent':
                self._validate_intent_node(n, issues, known_intents)

        for e in edges:
            if not isinstance(e, dict):
                issues.append(ReviewIssue('ERROR', 'dag.edge_not_object', 'DAG edge must be object'))
                continue
            a = str(e.get('from') or '')
            b = str(e.get('to') or '')
            if not a or not b:
                issues.append(ReviewIssue('ERROR', 'dag.edge_missing', 'DAG edge missing from/to'))
                continue
            if a not in node_ids or b not in node_ids:
                issues.append(ReviewIssue('ERROR', 'dag.edge_unknown_node', f'DAG edge references unknown node(s): {a}->{b}'))