import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class Document:
    id: str
    text: str
    meta: Dict[str, Any]

class InMemoryVectorStore:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.docs: List[Document] = []
        self.vecs: List[np.ndarray] = []

    def add(self, doc: Document):
        v = self.embedder.encode([doc.text], normalize_embeddings=True)[0]
        self.docs.append(doc)
        self.vecs.append(np.array(v, dtype=np.float32))

    def query(self, q: str, top_k: int = 5):
        qv = self.embedder.encode([q], normalize_embeddings=True)[0]
        qv = np.array(qv, dtype=np.float32)
        scored = []
        for d, v in zip(self.docs, self.vecs):
            scored.append((d, float(np.dot(qv, v))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

def load_jsonl(store: InMemoryVectorStore, path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        store.add(Document(id=str(obj["id"]), text=str(obj["text"]), meta=obj.get("meta", {})))
