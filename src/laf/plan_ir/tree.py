from __future__ import annotations
from typing import Any, Dict, Iterator, List

from .base import ExecUnit, _norm_id


class TreePlanIR:
    '''
    Tree plan is expected like:
    {
      'format':'tree',
      'goal':'...',
      'root': { 'id':'root', 'type':'category|intent|tool_call|manual_review', 'children':[...] }
    }
    '''

    def __init__(self, plan: Dict[str, Any]):
        self._plan = plan
        self._root = plan.get('root') or plan.get('plan_tree') or {}

    def format(self) -> str:
        return 'tree'

    def goal(self) -> str:
        return str(self._plan.get('goal', '') or '')

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self._plan)
        out['format'] = 'tree'
        if 'root' not in out and 'plan_tree' in out:
            out['root'] = out['plan_tree']
        return out

    def iter_units(self) -> Iterator[ExecUnit]:
        if not isinstance(self._root, dict):
            return
        yield from self._walk(self._root, parent_prefix='')

    def _walk(self, node: Dict[str, Any], parent_prefix: str) -> Iterator[ExecUnit]:
        ntype = node.get('type') or 'category'
        nid = _norm_id(node.get('id'), 'node')
        if parent_prefix:
            nid = f'{parent_prefix}.{nid}'

        if ntype == 'tool_call':
            plugin = str(node.get('plugin') or node.get('plugin_key') or node.get('tool') or '')
            args = node.get('args') if isinstance(node.get('args'), dict) else {}
            desc = str(node.get('description') or node.get('text') or f'Run tool {plugin}').strip()
            yield ExecUnit(id=nid, description=desc, node_type='tool_call', tool=plugin, args=args)
        elif ntype == 'intent':
            intent = str(node.get('intent') or node.get('intent_key') or node.get('key') or '')
            desc = str(node.get('goal') or node.get('description') or node.get('text') or '').strip()
            yield ExecUnit(id=nid, description=desc, node_type='intent', intent=intent)
        elif ntype == 'manual_review':
            desc = str(node.get('question') or node.get('description') or 'Manual review required').strip()
            yield ExecUnit(id=nid, description=desc, node_type='manual_review')
        else:
            # category nodes are not executable; no unit emitted
            pass

        children = node.get('children', [])
        if not isinstance(children, list):
            return
        for i, c in enumerate(children):
            if isinstance(c, dict):
                yield from self._walk(c, parent_prefix=nid)