import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class IntentRecord:
    key: str
    description: str
    tool: str = "manual_review"
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
        ex = intent.examples[:8]
        return f"Intent: {intent.key}\nDesc: {intent.description}\nExamples:\n- " + "\n- ".join(ex)

    def recompute_centroid(self, key: str) -> None:
        intent = self.intents[key]
        vec = self._embed(self._intent_text(intent))
        intent.centroid = vec.tolist()

    def add_intent(self, key: str, description: str, tool: str = "manual_review") -> IntentRecord:
        key = key.strip().lower()
        if key in self.intents:
            return self.intents[key]
        rec = IntentRecord(key=key, description=description.strip(), tool=tool)
        self.intents[key] = rec
        self.recompute_centroid(key)
        return rec

    def add_example(self, key: str, example: str) -> None:
        self.intents[key].examples.append(example)
        self.recompute_centroid(key)

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.intents = {k: IntentRecord(**v) for k, v in data.items()}

    def save(self) -> None:
        data = {k: asdict(v) for k, v in self.intents.items()}
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
