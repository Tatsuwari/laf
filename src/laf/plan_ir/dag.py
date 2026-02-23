from __future__ import annotations
from typing import Any, Dict, Iterator, List

from .base import ExecUnit, _norm_id


class DagPlanIR:
    '''
    DAG plan is expected like:
    {
      'format':'dag',
      'goal':'...',
      'nodes':[ {id,type,...}, ... ],
      'edges':[ {from,to,data?}, ... ]
    }
    Execution order here is a simple stable order:
      - topological sort if possible
      - fallback to given nodes order
    '''

    def __init__(self, plan: Dict[str, Any]):
        self._plan = plan
        self._nodes = plan.get('nodes') or []
        self._edges = plan.get('edges') or []

    def format(self) -> str:
        return 'dag'

    def goal(self) -> str:
        return str(self._plan.get('goal', '') or '')

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self._plan)
        out['format'] = 'dag'
        return out

    def iter_units(self) -> Iterator[ExecUnit]:
        if not isinstance(self._nodes, list):
            return
        nodes_by_id = {str(n.get('id')): n for n in self._nodes if isinstance(n, dict) and n.get('id') is not None}
        order = self._toposort(nodes_by_id, self._edges)
        for nid in order:
            n = nodes_by_id.get(nid)
            if not isinstance(n, dict):
                continue
            ntype = n.get('type') or 'step'

            if ntype == 'tool_call':
                tool = str(n.get('plugin') or n.get('plugin_key') or n.get('tool') or '')
                args = n.get('args') if isinstance(n.get('args'), dict) else {}
                desc = str(n.get('description') or n.get('text') or f'Run tool {tool}').strip()
                yield ExecUnit(id=nid, description=desc, node_type='tool_call', tool=tool, args=args)
            elif ntype == 'intent':
                intent = str(n.get('intent') or n.get('intent_key') or n.get('key') or '')
                desc = str(n.get('goal') or n.get('description') or n.get('text') or '').strip()
                yield ExecUnit(id=nid, description=desc, node_type='intent', intent=intent)
            elif ntype == 'manual_review':
                desc = str(n.get('question') or n.get('description') or 'Manual review required').strip()
                yield ExecUnit(id=nid, description=desc, node_type='manual_review')
            else:
                desc = str(n.get('description') or n.get('text') or '').strip()
                if desc:
                    yield ExecUnit(id=nid, description=desc, node_type='step')

    def _toposort(self, nodes_by_id: Dict[str, Dict[str, Any]], edges: Any) -> List[str]:
        # Kahn's algorithm; if edges invalid, fall back to node order
        ids = list(nodes_by_id.keys())

        if not isinstance(edges, list):
            return ids

        indeg = {i: 0 for i in ids}
        out_edges = {i: [] for i in ids}

        for e in edges:
            if not isinstance(e, dict):
                continue
            a = e.get('from')
            b = e.get('to')
            if a is None or b is None:
                continue
            a = str(a); b = str(b)
            if a not in nodes_by_id or b not in nodes_by_id:
                continue
            out_edges[a].append(b)
            indeg[b] += 1

        q = [i for i in ids if indeg[i] == 0]
        res: List[str] = []

        while q:
            cur = q.pop(0)
            res.append(cur)
            for nxt in out_edges.get(cur, []):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)

        # cycle fallback
        if len(res) != len(ids):
            return ids
        return res