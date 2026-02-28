# laf/core/memory.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import time
from pathlib import Path

from laf.core.trace_recorder import TraceRecorder


@dataclass
class MemoryEntry:
    tool: str
    data: Dict[str, Any]

    tags: List[str] = field(default_factory=list)

    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[int] = 3600

    importance: float = 0.50
    uses: int = 0

    ok: bool = True

    def expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_seconds is None:
            return False
        now = now or time.time()
        return (now - self.created_at) > self.ttl_seconds

    def touch(self) -> None:
        self.uses += 1
        self.last_accessed_at = time.time()


class MemoryManager:
    """
    Deterministic memory system.

    - Per-skill quotas
    - Global cap
    - Importance + recency scoring
    - Token budgeting
    - Optional disk persistence
    """

    def __init__(
        self,
        max_entries: int = 200,
        max_tokens: int = 800,
        per_skill_quota: Optional[Dict[str, int]] = None,
        default_skill_quota: int = 30,
        persist_path: str = "laf/data/memory/memory.jsonl",
        enable_persistence: bool = True,
        trace: Optional[TraceRecorder] = None,
    ):
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.per_skill_quota = per_skill_quota or {}
        self.default_skill_quota = default_skill_quota
        self.persist_path = Path(persist_path)
        self.enable_persistence = enable_persistence
        self.trace = trace

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)

    def ensure_ctx(self, ctx) -> None:
        if not hasattr(ctx, "memory_entries") or ctx.memory_entries is None:
            ctx.memory_entries = []

    def add(self, ctx, entry: MemoryEntry) -> None:
        self.ensure_ctx(ctx)
        self.prune_expired(ctx)

        fp = self._fingerprint(entry.tool, entry.data)
        for e in ctx.memory_entries:
            if self._fingerprint(e.tool, e.data) == fp:
                return

        ctx.memory_entries.append(entry)

        self._enforce_skill_quota(ctx, entry.tool)
        self._enforce_global_cap(ctx)

        if self.enable_persistence:
            self._append_to_disk(entry)

        if self.trace:
            self.trace.memory_add(
                getattr(ctx, "trace_run", None),
                tool=entry.tool,
                importance=entry.importance,
                ok=entry.ok,
            )

    def prune_expired(self, ctx) -> None:
        self.ensure_ctx(ctx)
        now = time.time()
        before = len(ctx.memory_entries)

        ctx.memory_entries = [
            e for e in ctx.memory_entries if not e.expired(now)
        ]

        removed = before - len(ctx.memory_entries)
        if removed > 0 and self.trace:
            self.trace.memory_prune(
                getattr(ctx, "trace_run", None),
                removed=removed,
            )

    def select(self, ctx, task: str, k: int = 12) -> List[MemoryEntry]:
        self.ensure_ctx(ctx)
        self.prune_expired(ctx)

        task_lower = (task or "").lower().strip()
        if not task_lower:
            return []

        keywords = self._keywords(task_lower)

        scored: List[Tuple[float, MemoryEntry]] = []
        now = time.time()

        for e in ctx.memory_entries:
            if e.expired(now):
                continue

            score = self._score_entry(e, keywords, task_lower, now)

            if score <= 0.0:
                continue

            scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = self._apply_token_budget(
            [e for _, e in scored],
            k=k,
        )

        for e in selected:
            e.touch()

        if self.trace:
            self.trace.memory_select(
                getattr(ctx, "trace_run", None),
                selected_count=len(selected),
            )

        return selected

    def to_injection_block(self, entries: List[MemoryEntry]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"memory": []}
        for e in entries:
            out["memory"].append(
                {
                    "tool": e.tool,
                    "tags": e.tags,
                    "importance": round(float(e.importance), 3),
                    "uses": e.uses,
                    "ok": e.ok,
                    "data": e.data,
                }
            )
        return out

    def bump_importance(self, entry: MemoryEntry, delta: float) -> None:
        entry.importance = float(
            max(0.0, min(1.0, entry.importance + delta))
        )

    def decay_importance(self, entry: MemoryEntry, delta: float) -> None:
        entry.importance = float(
            max(0.0, min(1.0, entry.importance - delta))
        )

    def _skill_quota(self, tool: str) -> int:
        return int(
            self.per_skill_quota.get(tool, self.default_skill_quota)
        )

    def _enforce_skill_quota(self, ctx, tool: str) -> None:
        quota = self._skill_quota(tool)

        entries = [
            e for e in ctx.memory_entries if e.tool == tool
        ]

        if len(entries) <= quota:
            return

        candidates = sorted(
            entries,
            key=lambda e: (
                e.importance,
                e.last_accessed_at,
                e.created_at,
            ),
        )

        to_remove = len(entries) - quota
        remove_set = set(candidates[:to_remove])

        ctx.memory_entries = [
            e for e in ctx.memory_entries if e not in remove_set
        ]

    def _enforce_global_cap(self, ctx) -> None:
        if len(ctx.memory_entries) <= self.max_entries:
            return

        candidates = sorted(
            ctx.memory_entries,
            key=lambda e: (
                e.importance,
                e.last_accessed_at,
                e.created_at,
            ),
        )

        to_remove = len(ctx.memory_entries) - self.max_entries
        remove_set = set(candidates[:to_remove])

        ctx.memory_entries = [
            e for e in ctx.memory_entries if e not in remove_set
        ]

    def _score_entry(
        self,
        e: MemoryEntry,
        keywords: List[str],
        task_lower: str,
        now: float,
    ) -> float:

        score = 0.0

        if not e.ok:
            if (
                "error" not in task_lower
                and e.tool.lower() not in task_lower
            ):
                return 0.0
            score -= 0.25

        tag_hits = sum(
            1 for t in (e.tags or [])
            if t.lower() in task_lower
        )
        score += 0.35 * tag_hits

        raw = json.dumps(
            e.data,
            ensure_ascii=False,
        ).lower()

        hits = sum(
            1 for kw in keywords if kw in raw
        )
        score += 0.08 * hits

        if e.tool.lower() in task_lower:
            score += 0.2

        score += 0.9 * float(e.importance)

        age = now - e.last_accessed_at
        score += max(
            0.0,
            0.2 - (age / 3600.0) * 0.2,
        )

        score += min(0.2, 0.02 * e.uses)

        return score

    def _keywords(self, text: str) -> List[str]:
        words = []

        for w in re_split_words(text):
            w = w.strip().lower()

            if len(w) <= 2:
                continue

            if w in {
                "the", "and", "for",
                "with", "that", "this",
                "what", "should", "next"
            }:
                continue

            words.append(w)

        seen = set()
        out = []

        for w in words:
            if w not in seen:
                out.append(w)
                seen.add(w)

        return out[:24]

    def _apply_token_budget(
        self,
        entries: List[MemoryEntry],
        k: int = 12,
    ) -> List[MemoryEntry]:

        selected: List[MemoryEntry] = []
        total_tokens = 0

        for e in entries:
            approx_tokens = max(
                1,
                len(json.dumps(
                    e.data,
                    ensure_ascii=False,
                )) // 4,
            )

            if total_tokens + approx_tokens > self.max_tokens:
                continue

            selected.append(e)
            total_tokens += approx_tokens

            if len(selected) >= k:
                break

        return selected

    def load_from_disk(self, ctx) -> None:
        self.ensure_ctx(ctx)

        if not self.persist_path.exists():
            return

        existing = {
            self._fingerprint(e.tool, e.data)
            for e in ctx.memory_entries
        }

        with open(self.persist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                tool = obj.get("tool", "")
                data = obj.get("data", {})

                if not tool or not isinstance(data, dict):
                    continue

                fp = self._fingerprint(tool, data)
                if fp in existing:
                    continue

                entry = MemoryEntry(
                    tool=tool,
                    data=data,
                    tags=obj.get("tags", []) or [],
                    created_at=float(obj.get("created_at", time.time())),
                    last_accessed_at=float(obj.get("last_accessed_at", time.time())),
                    ttl_seconds=obj.get("ttl_seconds", 3600),
                    importance=float(obj.get("importance", 0.5)),
                    uses=int(obj.get("uses", 0)),
                    ok=bool(obj.get("ok", True)),
                )

                ctx.memory_entries.append(entry)
                existing.add(fp)

        self.prune_expired(ctx)
        self._enforce_global_cap(ctx)

    def _append_to_disk(self, entry: MemoryEntry) -> None:
        obj = {
            "tool": entry.tool,
            "data": entry.data,
            "tags": entry.tags,
            "created_at": entry.created_at,
            "last_accessed_at": entry.last_accessed_at,
            "ttl_seconds": entry.ttl_seconds,
            "importance": entry.importance,
            "uses": entry.uses,
            "ok": entry.ok,
        }

        with open(self.persist_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(obj, ensure_ascii=False) + "\n"
            )

    def _fingerprint(self, tool: str, data: Dict[str, Any]) -> str:
        return json.dumps(
            {"tool": tool, "data": data},
            sort_keys=True,
            ensure_ascii=False,
        )


def re_split_words(text: str) -> List[str]:
    buf = []
    cur = []

    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            cur.append(ch)
        else:
            if cur:
                buf.append("".join(cur))
                cur = []

    if cur:
        buf.append("".join(cur))

    return buf