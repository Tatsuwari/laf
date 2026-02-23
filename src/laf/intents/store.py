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
