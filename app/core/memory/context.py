from __future__ import annotations

from typing import Optional

from .store import MemoryStore, MemoryType


# ═══════════════════════════════════════════════════════════════════
# MEMORY CONTEXT  —  read-only view for LLM / handlers
# ═══════════════════════════════════════════════════════════════════

class MemoryContext:
    """
    Immutable read-only lens over MemoryStore.

    This is the ONLY object the LLM engine or response handlers may
    receive. It exposes zero write methods — the LLM cannot pollute
    memory regardless of what it generates.
    """

    def __init__(self, store: MemoryStore):
        self._store = store

    def recall(self, key: str) -> Optional[str]:
        """Retrieve the value of a specific memory key."""
        r = self._store.get_by_key(key)
        return r.value if r else None

    def search(
        self,
        query: str,
        types: Optional[list[MemoryType]] = None,
        limit: int = 5,
    ) -> list[dict]:
        """Search memory by substring query, optionally filtered by type."""
        records = self._store.search(query, types=types, limit=limit)
        return [
            {"key": r.key, "value": r.value, "type": r.type.name}
            for r in records
        ]

    def facts(self, limit: int = 5) -> list[dict]:
        """Return top-ranked FACT records."""
        return [
            {"key": r.key, "value": r.value}
            for r in self._store.get_by_type(MemoryType.FACT, limit=limit)
        ]

    def preferences(self, limit: int = 5) -> list[dict]:
        """Return top-ranked PREF records."""
        return [
            {"key": r.key, "value": r.value}
            for r in self._store.get_by_type(MemoryType.PREF, limit=limit)
        ]

    def goals(self, limit: int = 5) -> list[dict]:
        """Return top-ranked GOAL records."""
        return [
            {"key": r.key, "value": r.value}
            for r in self._store.get_by_type(MemoryType.GOAL, limit=limit)
        ]

    def build_context_block(self, max_items: int = 6) -> str:
        """
        Build a formatted memory summary string for injection into LLM prompts.
        The LLM reads this — it never writes back.
        """
        lines  = []
        facts  = self.facts(limit=max_items // 2)
        prefs  = self.preferences(limit=max_items // 2)
        goals  = self.goals(limit=2)

        for f in facts:
            lines.append(f"[FACT] {f['key']}: {f['value']}")
        for p in prefs:
            lines.append(f"[PREF] {p['key']}: {p['value']}")
        for g in goals:
            lines.append(f"[GOAL] {g['key']}: {g['value']}")

        return "\n".join(lines) if lines else ""