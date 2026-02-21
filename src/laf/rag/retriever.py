from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import re

from .store import InMemoryVectorStore, Document


@dataclass
class RetrievedDoc:
    id: str
    text: str
    meta: Dict[str, Any]
    score: float


class Retriever:
    """
    Thin retrieval layer over the vector store.
    - Formats results consistently for generator
    - Optionally dedupes, truncates, and compresses
    """

    def __init__(self, store: InMemoryVectorStore):
        self.store = store

    def _dedupe(self, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        seen = set()
        out = []
        for d in docs:
            key = (d.id, re.sub(r"\s+", " ", d.text.strip())[:200])
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out

    def _truncate(self, text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars].rsplit(" ", 1)[0] + "…"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        max_doc_chars: int = 1200,
        dedupe: bool = True,
    ) -> List[RetrievedDoc]:
        raw: List[Tuple[Document, float]] = self.store.query(query, top_k=top_k)

        out = []
        for doc, score in raw:
            if min_score is not None and score < min_score:
                continue
            out.append(
                RetrievedDoc(
                    id=doc.id,
                    text=self._truncate(doc.text, max_doc_chars),
                    meta=doc.meta,
                    score=float(score),
                )
            )

        if dedupe:
            out = self._dedupe(out)

        return out