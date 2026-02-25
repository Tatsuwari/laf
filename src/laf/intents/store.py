import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class IntentRecord:
    key: str
    description: str
    tool: str = 'manual_review'

    category_path: List[str] = field(default_factory=lambda: ['misc'])
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)

    examples: List[str] = field(default_factory=list)
    centroid: Optional[List[float]] = None  # store as list for JSON

class IntentStore:
    def __init__(self, path: Path, embed_model: str):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(embed_model)
        self.intents: Dict[str, IntentRecord] = {}

    def _embed(self, text: str) -> np.ndarray:
        v = self.embedder.encode([text], normalize_embeddings=True)[0]
        return np.array(v, dtype=np.float32)

    def _intent_text(self, intent: IntentRecord) -> str:
        '''
        Include new structured metadata in embedding text.
        This improves routing quality automatically.
        '''

        ex = intent.examples[:8]
        cat = '/'.join(intent.category_path or ['misc'])
        tags = ','.join(intent.tags or [])
        return (
            f'Intent: {intent.key}\n'
            f'Category: {cat}\n'
            f'Desc: {intent.description}\n'
            f'Tool: {intent.tool}\n'
            f'Tags: {tags}\n'
            f'Examples:\n- ' + '\n'.join(ex)
        )

    def recompute_centroid(self, key: str) -> None:
        intent = self.intents[key]
        vec = self._embed(self._intent_text(intent))
        intent.centroid = vec.tolist()

    def add_intent(
            self,
            key: str,
            description: str,
            tool: str = 'manual_review',
            category_path: Optional[List[str]] = None,
            input_schema: Optional[Dict[str, Any]] = None,
            output_schema: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
            ) -> IntentRecord:
        
        key = key.strip().lower()


        if key in self.intents:
            return self.intents[key]
        
        
        rec = IntentRecord(
            key=key,
            description=description.strip(),
            tool=tool,
            category_path=category_path or ["misc"],
            input_schema=input_schema,
            output_schema=output_schema,
            tags=tags or [],
            )
        self.intents[key] = rec
        self.recompute_centroid(key)
        return rec
    
    def update_intent_metadata(
        self,
        key: str,
        category_path: Optional[List[str]] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        '''
        Allows evolution of intent design without recreating it.
        '''
        if key not in self.intents:
            return

        intent = self.intents[key]

        if category_path is not None:
            intent.category_path = category_path
        if input_schema is not None:
            intent.input_schema = input_schema
        if output_schema is not None:
            intent.output_schema = output_schema
        if tags is not None:
            intent.tags = tags

        self.recompute_centroid(key)

    def add_example(self, key: str, example: str) -> None:
        if key not in self.intents:
            return
        self.intents[key].examples.append(example)
        self.recompute_centroid(key)

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding='utf-8'))
        self.intents = {k: IntentRecord(**v) for k, v in data.items()}

    def save(self) -> None:
        data = {k: asdict(v) for k, v in self.intents.items()}
        self.path.write_text(json.dumps(data, indent=2), encoding='utf-8')

    # --- add to IntentStore ---

def delete_intent(self, key: str) -> bool:
    key = key.strip().lower()
    if key not in self.intents:
        return False
    del self.intents[key]
    return True

def rename_intent(self, old_key: str, new_key: str) -> bool:
    old_key = old_key.strip().lower()
    new_key = new_key.strip().lower()
    if old_key not in self.intents:
        return False
    if new_key in self.intents:
        return False

    rec = self.intents[old_key]
    rec.key = new_key
    self.intents[new_key] = rec
    del self.intents[old_key]
    self.recompute_centroid(new_key)
    return True

def promote_intent(
    self,
    key: str,
    category_path: Optional[List[str]] = None,
    tool: Optional[str] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> bool:
    key = key.strip().lower()
    if key not in self.intents:
        return False

    intent = self.intents[key]
    if category_path is not None:
        intent.category_path = category_path
    if tool is not None:
        intent.tool = tool
    if tags is not None:
        intent.tags = tags
    if description is not None:
        intent.description = description.strip()

    self.recompute_centroid(key)
    return True

def merge_intents(
    self,
    target_key: str,
    source_keys: List[str],
    *,
    delete_sources: bool = True,
    prefer: str = "target",  # "target" | "source_first"
) -> Dict[str, Any]:
    """
    Merge multiple intents into target_key.
    - examples/tags are unioned
    - schemas keep target unless missing
    - category/tool keep target unless prefer == "source_first"
    """
    target_key = target_key.strip().lower()
    if target_key not in self.intents:
        return {"ok": False, "error": "target intent not found"}

    sources = [k.strip().lower() for k in (source_keys or []) if k.strip()]
    sources = [k for k in sources if k != target_key]
    missing = [k for k in sources if k not in self.intents]
    if missing:
        return {"ok": False, "error": f"missing source intents: {missing}"}

    tgt = self.intents[target_key]

    def _first_nonempty(vals):
        for v in vals:
            if v not in (None, "", [], {}):
                return v
        return None

    for sk in sources:
        src = self.intents[sk]

        # examples (keep unique, preserve order-ish)
        seen = set(tgt.examples)
        for ex in src.examples:
            if ex not in seen:
                tgt.examples.append(ex)
                seen.add(ex)

        # tags (set union, stable)
        tset = set(tgt.tags)
        for t in (src.tags or []):
            if t not in tset:
                tgt.tags.append(t)
                tset.add(t)

        # schemas: fill gaps
        if tgt.input_schema is None and src.input_schema is not None:
            tgt.input_schema = src.input_schema
        if tgt.output_schema is None and src.output_schema is not None:
            tgt.output_schema = src.output_schema

        if prefer == "source_first":
            tgt.tool = _first_nonempty([src.tool, tgt.tool]) or tgt.tool
            tgt.category_path = _first_nonempty([src.category_path, tgt.category_path]) or tgt.category_path
            tgt.description = _first_nonempty([src.description, tgt.description]) or tgt.description

    # recompute centroid after merge
    self.recompute_centroid(target_key)

    if delete_sources:
        for sk in sources:
            self.delete_intent(sk)

    return {
        "ok": True,
        "target": target_key,
        "merged_sources": sources,
        "deleted_sources": bool(delete_sources),
        "target_examples": len(self.intents[target_key].examples),
        "target_tags": len(self.intents[target_key].tags),
    }
