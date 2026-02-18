from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol, runtime_checkable

logger = logging.getLogger("Brain")


class ResponseType(str, Enum):
    SPEECH = "SPEECH"
    ACTION = "ACTION"
    SYSTEM = "SYSTEM"
    ERROR = "ERROR"
    SILENT = "SILENT"


@dataclass
class BrainResponse:
    type: ResponseType
    reply: str = ""
    action: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {"type": self.type.value, "reply": self.reply}
        if self.action:
            out["action"] = self.action
        if self.meta:
            out["meta"] = self.meta
        return out


@runtime_checkable
class IntentEngine(Protocol):
    def detect_intent(self, text: str) -> dict: ...


@runtime_checkable
class LLMEngine(Protocol):
    def ask(self, text: str) -> str: ...


STATIC_RESPONSES: dict[frozenset, str] = {
    frozenset({"hi", "hello", "hey", "howdy", "sup", "yo"}): "Hello.",
    frozenset({"bye", "goodbye", "see you", "later", "cya"}): "Goodbye.",
    frozenset({"thanks", "thank you", "thx", "ty"}): "You're welcome.",
    frozenset({"ok", "okay", "alright", "got it", "sure"}): "Understood.",
}


def _lookup_static(text: str) -> Optional[str]:
    normalized = text.lower().strip()
    for key_set, reply in STATIC_RESPONSES.items():
        if normalized in key_set:
            return reply
    return None


SYSTEM_REPLIES: dict[str, str] = {
    "SHUTDOWN": "Shutting down.",
    "RESTART": "Restarting.",
    "SLEEP": "Going to sleep.",
    "WAKE": "I'm awake.",
}


def _system_reply(action: str) -> str:
    return SYSTEM_REPLIES.get(action.upper(), "System command received.")


class Brain:
    def __init__(
        self,
        intent_engine: IntentEngine,
        llm_engine: LLMEngine,
        fallback_reply: str = "I'm not sure how to respond to that.",
        hooks: Optional[dict[ResponseType, Callable[[BrainResponse], None]]] = None,
    ):
        if not isinstance(intent_engine, IntentEngine):
            raise TypeError("intent_engine must implement IntentEngine protocol.")
        if not isinstance(llm_engine, LLMEngine):
            raise TypeError("llm_engine must implement LLMEngine protocol.")

        self.intent_engine = intent_engine
        self.llm_engine = llm_engine
        self.fallback_reply = fallback_reply
        self.hooks = hooks or {}

    def process(self, user_text: str) -> dict:
        start = time.monotonic()

        try:
            response = self._route(user_text)
        except Exception as exc:
            logger.exception("Unhandled exception in Brain.process: %s", exc)
            response = BrainResponse(
                type=ResponseType.ERROR,
                reply="Something went wrong on my end.",
                meta={"error": str(exc)},
            )

        elapsed = round((time.monotonic() - start) * 1000, 2)
        response.meta["latency_ms"] = elapsed

        self._fire_hook(response)
        logger.debug(
            "Brain response [%s] in %sms: %s",
            response.type,
            elapsed,
            response.reply,
        )

        return response.to_dict()

    def _route(self, user_text: str) -> BrainResponse:
        user_text = user_text.strip()

        if not user_text:
            return BrainResponse(type=ResponseType.SILENT)

        intent_data = self._safe_detect_intent(user_text)
        intent = intent_data.get("intent", "UNKNOWN")
        action = intent_data.get("action", "")

        if intent == "SYSTEM":
            return self._handle_system(action)

        if intent == "COMMAND":
            return self._handle_command(action, intent_data)

        static = _lookup_static(user_text)
        if static:
            return BrainResponse(type=ResponseType.SPEECH, reply=static)

        return self._handle_llm(user_text)

    def _handle_system(self, action: str) -> BrainResponse:
        reply = _system_reply(action)
        return BrainResponse(
            type=ResponseType.SYSTEM,
            reply=reply,
            action=action or "SHUTDOWN",
        )

    def _handle_command(self, action: str, intent_data: dict) -> BrainResponse:
        if not action or action == "UNKNOWN":
            logger.warning("COMMAND intent with no valid action: %s", intent_data)
            return BrainResponse(
                type=ResponseType.ERROR,
                reply="I understood a command, but couldn't identify the action.",
            )
        return BrainResponse(
            type=ResponseType.ACTION,
            reply=f"Running: {action}.",
            action=action,
            meta={k: v for k, v in intent_data.items() if k not in ("intent", "action")},
        )

    def _handle_llm(self, user_text: str) -> BrainResponse:
        try:
            ai_reply = self.llm_engine.ask(user_text)
        except Exception as exc:
            logger.error("LLM engine failed: %s", exc)
            return BrainResponse(
                type=ResponseType.ERROR,
                reply="My AI module hit an error. Please try again.",
                meta={"error": str(exc)},
            )

        reply = ai_reply.strip() if ai_reply and ai_reply.strip() else self.fallback_reply
        return BrainResponse(type=ResponseType.SPEECH, reply=reply)

    def _safe_detect_intent(self, text: str) -> dict:
        try:
            result = self.intent_engine.detect_intent(text)
            if not isinstance(result, dict):
                raise ValueError(f"IntentEngine returned non-dict: {type(result)}")
            return result
        except Exception as exc:
            logger.error("IntentEngine failed: %s", exc)
            return {"intent": "UNKNOWN"}

    def _fire_hook(self, response: BrainResponse) -> None:
        hook = self.hooks.get(response.type)
        if hook:
            try:
                hook(response)
            except Exception as exc:
                logger.warning("Hook for %s raised: %s", response.type, exc)
