from __future__ import annotations
from dataclasses import asdict
from typing import Any,Dict, List, Optional

from ..intents.store import IntentStore
from ..plugins.registry import ToolRegistry


def _safe_intent_dict(intent_obj: Any) -> Dict[str, Any]:
    '''
    Intent objects may evolve. We extract fields defensively.
    '''
    d: Dict[str, Any] = {}

    # required
    d['key'] = getattr(intent_obj, 'key', None) or getattr(intent_obj, 'name', None) or ''
    d['description'] = getattr(intent_obj, 'description', '') or ''

    # existing laf fields
    d['tool'] = getattr(intent_obj, 'tool', None) or 'manual_review'

    # NEW: hierarchical routing fields (optional)
    d['category_path'] = getattr(intent_obj, 'category_path', None) or ['misc']
    d['input_schema'] = getattr(intent_obj, 'input_schema', None)
    d['output_schema'] = getattr(intent_obj, 'output_schema', None)

    # examples: store typically has examples list
    ex = getattr(intent_obj, 'examples', None)
    if ex is None:
        ex = []
    d['examples'] = list(ex) if isinstance(ex, (list, tuple)) else []

    # optional: tags / constraints
    d['tags'] = getattr(intent_obj, 'tags', None) or []
    d['constraints'] = getattr(intent_obj, 'constraints', None)

    return d


def build_capability_catalog(
    intent_store: IntentStore,
    tools: ToolRegistry,
    include_examples: bool = True,
) -> Dict[str, Any]:
    '''
    Unified catalog passed to planner/reviewer/router:
      - intents: semantic routing nodes
      - plugins/tools: actual callables with param schemas
    '''
    intents: List[Dict[str, Any]] = []
    for k, intent in intent_store.intents.items():
        d = _safe_intent_dict(intent)
        if not include_examples:
            d['examples'] = []
        intents.append(d)

    plugins: List[Dict[str, Any]] = []
    for name, spec in tools.tools.items():
        plugins.append(
            {
                'key': spec.name,
                'category_path': getattr(spec, 'category_path', None) or ['tools'],
                'description': spec.description or '',
                'params_schema': spec.params_schema,
                'returns_schema': getattr(spec, 'returns_schema', None),
                'tags': getattr(spec, 'tags', None) or [],
                'constraints': getattr(spec, 'constraints', None),
            }
        )

    return {
        'intents': intents,
        'plugins': plugins,
    }


def summarize_catalog(catalog: Dict[str, Any], max_items: int = 50) -> Dict[str, Any]:
    '''
    Smaller payload for planner prompts.
    Keeps key fields only; trims long lists.
    '''
    intents = catalog.get('intents', [])[:max_items]
    plugins = catalog.get('plugins', [])[:max_items]

    def strip_intent(x: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'key': x.get('key', ''),
            'description': x.get('description', ''),
            'tool': x.get('tool', 'manual_review'),
            'category_path': x.get('category_path') or ['misc'],
            'input_schema': x.get('input_schema'),
            'output_schema': x.get('output_schema'),
            'examples': (x.get('examples') or [])[:3],
        }

    def strip_plugin(x: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'key': x.get('key', ''),
            'description': x.get('description', ''),
            'category_path': x.get('category_path') or ['tools'],
            'params_schema': x.get('params_schema'),
            'returns_schema': x.get('returns_schema'),
        }

    return {
        'intents': [strip_intent(i) for i in intents],
        'plugins': [strip_plugin(p) for p in plugins],
    }