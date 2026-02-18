from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .store import MemoryStore, MemoryRecord, MemoryType, _DEFAULT_TTL, _new_id

logger = logging.getLogger("Brain")


# ═══════════════════════════════════════════════════════════════════
# EXTRACTION RULE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ExtractionRule:
    """
    A single pattern -> memory mapping rule.

    pattern:    Compiled regex. Use named group ?P<value> to capture the fact.
    key:        Normalized memory key (e.g. "user.name").
    mem_type:   Semantic type to assign.
    confidence: Base confidence for this rule (0.0 - 1.0).
    tags:       Optional tags for categorization.
    ttl:        Override TTL in seconds. None = use type default.
    """
    pattern:    re.Pattern
    key:        str
    mem_type:   MemoryType
    confidence: float           = 0.9
    tags:       list[str]       = field(default_factory=list)
    ttl:        Optional[float] = None


# ═══════════════════════════════════════════════════════════════════
# EXTRACTION RULE SET
# Add new rules here freely — no Brain logic changes needed.
# ═══════════════════════════════════════════════════════════════════

_EXTRACTION_RULES: list[ExtractionRule] = [

    # Identity — name
    ExtractionRule(
        pattern    = re.compile(r"\b(?:my name is|i am|i'm|call me)\s+(?P<value>[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", re.I),
        key        = "user.name",
        mem_type   = MemoryType.FACT,
        confidence = 0.95,
        tags       = ["personal", "identity"],
    ),

    # Location
    ExtractionRule(
        pattern    = re.compile(r"\b(?:i(?:'m| am) (?:from|in|based in)|i live in)\s+(?P<value>[A-Z][a-zA-Z\s,]+?)(?:\.|$)", re.I),
        key        = "user.location",
        mem_type   = MemoryType.FACT,
        confidence = 0.88,
        tags       = ["personal", "location"],
    ),

    # Occupation
    ExtractionRule(
        pattern    = re.compile(r"\b(?:i(?:'m| am) a|i work as a?n?)\s+(?P<value>[a-z][a-z\s]+?)(?:\.|,|$)", re.I),
        key        = "user.occupation",
        mem_type   = MemoryType.FACT,
        confidence = 0.85,
        tags       = ["personal"],
    ),

    # Age
    ExtractionRule(
        pattern    = re.compile(r"\bi(?:'m| am)\s+(?P<value>\d{1,3})\s+years?\s+old\b", re.I),
        key        = "user.age",
        mem_type   = MemoryType.FACT,
        confidence = 0.92,
        tags       = ["personal"],
    ),

    # Language preference
    ExtractionRule(
        pattern    = re.compile(r"\b(?:i prefer|please (?:use|speak|write(?: in)?))\s+(?P<value>[a-zA-Z]+)(?:\s+language)?", re.I),
        key        = "pref.language",
        mem_type   = MemoryType.PREF,
        confidence = 0.90,
        tags       = ["preference", "language"],
    ),

    # Response style preference
    ExtractionRule(
        pattern    = re.compile(r"\b(?:keep (?:it|your (?:answers?|responses?)) (?P<value>short|brief|concise|detailed|simple|formal|casual))\b", re.I),
        key        = "pref.response_style",
        mem_type   = MemoryType.PREF,
        confidence = 0.88,
        tags       = ["preference", "style"],
    ),

    # Goal
    ExtractionRule(
        pattern    = re.compile(r"\b(?:i want to|i need to|i(?:'m| am) trying to|my goal is to)\s+(?P<value>.+?)(?:\.|$)", re.I),
        key        = "user.goal",
        mem_type   = MemoryType.GOAL,
        confidence = 0.80,
        tags       = ["intent"],
        ttl        = 86_400 * 7,  # 7 days
    ),

    # Feedback — positive
    ExtractionRule(
        pattern    = re.compile(r"\b(?:that(?:'s| was) (?:great|perfect|correct|right|helpful)|good (?:job|answer))\b", re.I),
        key        = "feedback.positive",
        mem_type   = MemoryType.FEEDBACK,
        confidence = 0.75,
        tags       = ["feedback"],
    ),

    # Feedback — correction
    ExtractionRule(
        pattern    = re.compile(r"\b(?:that(?:'s| was) (?:wrong|incorrect|not right)|no[,.]?\s+(?:that|it)(?:'s| is) (?:not|wrong))\b", re.I),
        key        = "feedback.correction",
        mem_type   = MemoryType.FEEDBACK,
        confidence = 0.80,
        tags       = ["feedback", "correction"],
    ),
]


# ═══════════════════════════════════════════════════════════════════
# MEMORY EXTRACTOR
# ═══════════════════════════════════════════════════════════════════

class MemoryExtractor:
    """
    Brain-owned extraction pipeline. The SOLE gateway for writing to MemoryStore.

    Responsibilities:
      - Run all ExtractionRules against user input and optionally LLM output.
      - Produce MemoryRecord objects and write them to MemoryStore.
      - LLM output is treated as a lower-trust signal (0.6x confidence).
        The LLM never writes directly — this extractor is the controlled
        channel and sole arbiter of what enters memory.
    """

    # AI output is trusted less than direct user statements
    LLM_CONFIDENCE_MULTIPLIER: float = 0.6

    def __init__(
        self,
        store: MemoryStore,
        rules: Optional[list[ExtractionRule]] = None,
    ):
        self.store = store
        self.rules = rules if rules is not None else _EXTRACTION_RULES

    # ----------------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------------

    def extract_from_user(self, text: str) -> list[MemoryRecord]:
        """Extract memories from direct user input (full confidence)."""
        return self._run_rules(text, source="user", confidence_scale=1.0)

    def extract_from_llm_response(self, text: str) -> list[MemoryRecord]:
        """
        Extract signals from LLM output at reduced confidence.

        The LLM never writes to memory directly — this is the controlled
        channel through which AI output may influence the store.
        """
        return self._run_rules(
            text,
            source="llm_response",
            confidence_scale=self.LLM_CONFIDENCE_MULTIPLIER,
        )

    def store_context_signal(
        self,
        key:   str,
        value: str,
        ttl:   float = 300,
    ) -> MemoryRecord:
        """
        Directly store a transient context signal from Brain logic.
        Examples: current topic, active intent, last command.
        """
        record = MemoryRecord(
            id         = _new_id(),
            type       = MemoryType.CONTEXT,
            key        = key,
            value      = value,
            confidence = 1.0,
            source     = "brain",
            ttl        = ttl,
            tags       = ["context"],
        )
        return self.store.upsert(record)

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _run_rules(
        self,
        text:             str,
        source:           str,
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
                id         = _new_id(),
                type       = rule.mem_type,
                key        = rule.key,
                value      = value,
                confidence = round(rule.confidence * confidence_scale, 4),
                source     = source,
                ttl        = ttl,
                tags       = list(rule.tags),
            )

            saved = self.store.upsert(record)
            extracted.append(saved)

            logger.debug(
                "Extracted [%s] %s = %r  (conf=%.2f, src=%s)",
                record.type.name, record.key, record.value,
                record.confidence, source,
            )

        return extracted