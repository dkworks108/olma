from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .store import MemoryStore, MemoryRecord, MemoryType, _DEFAULT_TTL, _new_id

logger = logging.getLogger("Brain")


@dataclass
class ExtractionRule:
    pattern: re.Pattern
    key: str
    mem_type: MemoryType
    confidence: float = 0.9
    tags: list[str] = field(default_factory=list)
    ttl: Optional[float] = None


_EXTRACTION_RULES: list[ExtractionRule] = [
    ExtractionRule(
        pattern=re.compile(
            r"\b(?:my name is|i am|i'm|call me)\s+(?P<value>[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",
            re.I,
        ),
        key="user.name",
        mem_type=MemoryType.FACT,
        confidence=0.95,
        tags=["personal", "identity"],
    ),
    ExtractionRule(
        pattern=re.compile(
            r"\b(?:i(?:'m| am) (?:from|in|based in)|i live in)\s+(?P<value>[A-Z][a-zA-Z\s,]+?)(?:\.|$)",
            re.I,
        ),
        key="user.location",
        mem_type=MemoryType.FACT,
        confidence=0.88,
        tags=["personal", "location"],
    ),
    ExtractionRule(
        pattern=re.compile(
            r"\b(?:i(?:'m| am) a|i work as a?n?)\s+(?P<value>[a-z][a-z\s]+?)(?:\.|,|$)",
            re.I,
        ),
        key="user.occupation",
        mem_type=MemoryType.FACT,
        confidence=0.85,
        tags=["personal"],
    ),
    ExtractionRule(
        pattern=re.compile(
            r"\bi(?:'m| am)\s+(?P<value>\d{1,3})\s+years?\s+old\b",
            re.I,
        ),
        key="user.age",
        mem_type=MemoryType.FACT,
        confidence=0.92,
        tags=["personal"],
    ),
    ExtractionRule(
        pattern=re.compile(
            r"\b(?:i prefer|please (?:use|speak|write(?: in)?))\s+(?P<value>[a-zA-Z]+)(?:\s+language)?",
            re.I,
        ),
        key="pref.language",
        mem_type=MemoryType.PREF,
        confidence=0.90,
        tags=["preference", "language"],
    ),
    ExtractionRule(
        pattern=re.compile(
            r"\b(?:keep (?:it|your (?:answers?|responses?)) (?P<value>short|brief|concise|detailed|simple|formal|casual))\b",
            re.I,
        ),
        key="pref.response_style",
        mem_type=MemoryType.PREF,
        confidence=0.88,
        tags=["preference", "style"],
    ),
    ExtractionRule(
        pattern=re.compile(
            r"\b(?:i want to|i need to|i(?:'m| am) trying to|my goal is to)\s+(?P<value>.+?)(?:\.|$)",
            re.I,
        ),
        key="user.goal",
        mem_type=MemoryType.GOAL,
        confidence=0.80,
        tags=["intent"],
        ttl=86_400 * 7,
    ),
    ExtractionRule(
        pattern=re.compile(
            r"\b(?:that(?:'s| was) (?:great|perfect|correct|right|helpful)|good (?:job|answer))\b",
            re.I,
        ),
        key="feedback.positive",
        mem_type=MemoryType.FEEDBACK,
        confidence=0.75,
        tags=["feedback"],
    ),
    ExtractionRule(
        pattern=re.compile(
            r"\b(?:that(?:'s| was) (?:wrong|incorrect|not right)|no[,.]?\s+(?:that|it)(?:'s| is) (?:not|wrong))\b",
            re.I,
        ),
        key="feedback.correction",
        mem_type=MemoryType.FEEDBACK,
        confidence=0.80,
        tags=["feedback", "correction"],
    ),
]


class MemoryExtractor:
    LLM_CONFIDENCE_MULTIPLIER: float = 0.6

    def __init__(
        self,
        store: MemoryStore,
        rules: Optional[list[ExtractionRule]] = None,
    ):
        self.store = store
        self.rules = rules if rules is not None else _EXTRACTION_RULES

    def extract_from_user(self, text: str) -> list[MemoryRecord]:
        return self._run_rules(text, source="user", confidence_scale=1.0)

    def extract_from_llm_response(self, text: str) -> list[MemoryRecord]:
        return self._run_rules(
            text,
            source="llm_response",
            confidence_scale=self.LLM_CONFIDENCE_MULTIPLIER,
        )

    def store_context_signal(
        self,
        key: str,
        value: str,
        ttl: float = 300,
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=_new_id(),
            type=MemoryType.CONTEXT,
            key=key,
            value=value,
            confidence=1.0,
            source="brain",
            ttl=ttl,
            tags=["context"],
        )
        return self.store.upsert(record)

    def _run_rules(
        self,
        text: str,
        source: str,
        confidence_scale: float,
    ) -> list[MemoryRecord]:
        extracted: list[MemoryRecord] = []

        for rule in self.rules:
            match = rule.pattern.search(text)
            if not match:
                continue

            try:
                value = match.group("value").strip()
            except IndexError:
                value = match.group(0).strip()

            if not value:
                continue

            ttl = rule.ttl if rule.ttl is not None else _DEFAULT_TTL.get(rule.mem_type)

            record = MemoryRecord(
                id=_new_id(),
                type=rule.mem_type,
                key=rule.key,
                value=value,
                confidence=round(rule.confidence * confidence_scale, 4),
                source=source,
                ttl=ttl,
                tags=list(rule.tags),
            )

            saved = self.store.upsert(record)
            extracted.append(saved)

            logger.debug(
                "Extracted [%s] %s = %r  (conf=%.2f, src=%s)",
                record.type.name,
                record.key,
                record.value,
                record.confidence,
                source,
            )

        return extracted
