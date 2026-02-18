from __future__ import annotations

import logging
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger("Brain")


# ═══════════════════════════════════════════════════════════════════
# MEMORY TYPES
# ═══════════════════════════════════════════════════════════════════

class MemoryType(Enum):
    """
    Semantic category of a memory record.

    FACT     — user stated something about themselves or the world
    PREF     — user expressed a preference or desire
    ENTITY   — a named thing (person, place, product, etc.)
    CONTEXT  — transient conversational context (short TTL)
    GOAL     — something the user wants to achieve
    FEEDBACK — user expressed satisfaction or correction
    SYSTEM   — internal Brain metadata (never shown to user)
    """
    FACT     = auto()
    PREF     = auto()
    ENTITY   = auto()
    CONTEXT  = auto()
    GOAL     = auto()
    FEEDBACK = auto()
    SYSTEM   = auto()


# Default TTL in seconds per memory type. None = never expires.
_DEFAULT_TTL: dict[MemoryType, Optional[float]] = {
    MemoryType.FACT:     None,          # permanent
    MemoryType.PREF:     None,          # permanent
    MemoryType.ENTITY:   86_400 * 30,   # 30 days
    MemoryType.CONTEXT:  600,           # 10 minutes
    MemoryType.GOAL:     86_400 * 7,    # 7 days
    MemoryType.FEEDBACK: 86_400 * 3,    # 3 days
    MemoryType.SYSTEM:   None,
}


# ═══════════════════════════════════════════════════════════════════
# MEMORY RECORD
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MemoryRecord:
    """
    Atomic unit of stored knowledge.

    id:            Unique record identifier.
    type:          Semantic category (MemoryType).
    key:           Normalized topic label (e.g. "user.name", "pref.language").
    value:         The extracted fact/preference as a string.
    confidence:    Extraction confidence [0.0 - 1.0].
    source:        Where this came from: "user", "brain", "llm_response".
    timestamp:     Monotonic time when created.
    ttl:           Seconds until expiry. None = immortal.
    access_count:  How many times this record was retrieved (for scoring).
    last_accessed: Monotonic time of last retrieval.
    tags:          Free-form tags for grouping (e.g. ["personal", "identity"]).
    """
    id:            str
    type:          MemoryType
    key:           str
    value:         str
    confidence:    float            = 1.0
    source:        str              = "user"
    timestamp:     float            = field(default_factory=time.monotonic)
    ttl:           Optional[float]  = None
    access_count:  int              = 0
    last_accessed: float            = field(default_factory=time.monotonic)
    tags:          list[str]        = field(default_factory=list)

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.monotonic() - self.timestamp) > self.ttl

    def relevance_score(self) -> float:
        """
        Composite ranking score used during retrieval.
        Score = confidence x log1p(access_count) x recency_decay
        Recency decay: e^(-age_hours / 24) — halves every ~17 hours.
        """
        age_hours  = (time.monotonic() - self.timestamp) / 3600
        recency    = math.exp(-age_hours / 24)
        freq_bonus = math.log1p(self.access_count)
        return self.confidence * (1 + freq_bonus) * recency

    def touch(self) -> None:
        """Record an access event."""
        self.access_count  += 1
        self.last_accessed  = time.monotonic()


# ═══════════════════════════════════════════════════════════════════
# MEMORY STORE
# ═══════════════════════════════════════════════════════════════════

class MemoryStore:
    """
    Structured, scored, expiring key-value memory store.

    Design:
      - Primary index:  id -> MemoryRecord
      - Secondary index: key -> list[id]   (fast key lookup)
      - Type index:     MemoryType -> set[id]

    Write access:
      - ONLY Brain / MemoryExtractor may call write methods.
      - LLM engines receive a read-only MemoryContext view.

    Capacity:
      - max_records cap with LRU-style eviction of lowest-scoring records.
    """

    def __init__(self, max_records: int = 1_000):
        self.max_records  = max_records
        self._store:      dict[str, MemoryRecord]     = {}
        self._key_index:  dict[str, list[str]]        = defaultdict(list)
        self._type_index: dict[MemoryType, set[str]]  = defaultdict(set)

    # ----------------------------------------------------------------
    # WRITE  (Brain-internal use only)
    # ----------------------------------------------------------------

    def upsert(self, record: MemoryRecord) -> MemoryRecord:
        """
        Insert or update a memory record.

        If a record with the same key already exists and the new confidence
        is higher, it replaces the old one. Otherwise the existing record
        is kept and its access_count is bumped (soft merge).
        """
        existing_ids = self._key_index.get(record.key, [])
        for eid in existing_ids:
            existing = self._store.get(eid)
            if existing and not existing.is_expired():
                if record.confidence >= existing.confidence:
                    self._remove(eid)
                    break
                else:
                    existing.touch()
                    logger.debug("Memory soft-merge: %s", record.key)
                    return existing

        self._enforce_capacity()
        self._store[record.id] = record
        self._key_index[record.key].append(record.id)
        self._type_index[record.type].add(record.id)
        logger.debug(
            "Memory upsert [%s] %s = %s",
            record.type.name, record.key, record.value,
        )
        return record

    def delete(self, record_id: str) -> bool:
        if record_id in self._store:
            self._remove(record_id)
            return True
        return False

    def purge_expired(self) -> int:
        """Remove all expired records. Returns count removed."""
        expired = [rid for rid, r in self._store.items() if r.is_expired()]
        for rid in expired:
            self._remove(rid)
        if expired:
            logger.debug("Purged %d expired memory records.", len(expired))
        return len(expired)

    # ----------------------------------------------------------------
    # READ  (exposed via MemoryContext for LLM / handlers)
    # ----------------------------------------------------------------

    def get_by_key(self, key: str) -> Optional[MemoryRecord]:
        ids = self._key_index.get(key, [])
        candidates = [
            self._store[i] for i in ids
            if i in self._store and not self._store[i].is_expired()
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda r: r.relevance_score())
        best.touch()
        return best

    def search(
        self,
        query: str,
        types: Optional[list[MemoryType]] = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        """Fuzzy substring search over keys and values, ranked by relevance_score."""
        q       = query.lower()
        results = []
        for record in self._store.values():
            if record.is_expired():
                continue
            if types and record.type not in types:
                continue
            if q in record.key.lower() or q in record.value.lower():
                results.append(record)

        results.sort(key=lambda r: r.relevance_score(), reverse=True)
        top = results[:limit]
        for r in top:
            r.touch()
        return top

    def get_by_type(
        self,
        mem_type: MemoryType,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        ids = self._type_index.get(mem_type, set())
        records = [
            self._store[i] for i in ids
            if i in self._store and not self._store[i].is_expired()
        ]
        records.sort(key=lambda r: r.relevance_score(), reverse=True)
        return records[:limit]

    def snapshot(self) -> list[dict]:
        """Serializable dump of all live records (for persistence layer)."""
        return [
            {
                "id":           r.id,
                "type":         r.type.name,
                "key":          r.key,
                "value":        r.value,
                "confidence":   r.confidence,
                "source":       r.source,
                "timestamp":    r.timestamp,
                "ttl":          r.ttl,
                "access_count": r.access_count,
                "tags":         r.tags,
            }
            for r in self._store.values()
            if not r.is_expired()
        ]

    def __len__(self) -> int:
        return len(self._store)

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _remove(self, record_id: str) -> None:
        record = self._store.pop(record_id, None)
        if record:
            self._key_index[record.key] = [
                i for i in self._key_index[record.key] if i != record_id
            ]
            self._type_index[record.type].discard(record_id)

    def _enforce_capacity(self) -> None:
        if len(self._store) < self.max_records:
            return
        evictable = [r for r in self._store.values() if r.ttl is not None]
        if not evictable:
            logger.warning("Memory store full with only permanent records. Cannot evict.")
            return
        victim = min(evictable, key=lambda r: r.relevance_score())
        logger.debug("Memory eviction: %s (score=%.4f)", victim.key, victim.relevance_score())
        self._remove(victim.id)


# ═══════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════

def _new_id() -> str:
    return uuid.uuid4().hex[:12]