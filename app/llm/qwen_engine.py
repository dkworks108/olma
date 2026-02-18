# qwen_engine.py
# Advanced, production-grade "dumb transport" LLM engine for llama.cpp /completion
# - Accepts context=... (ignored) to match Brain contract
# - NEVER injects objects, roles, or system prompts
# - Optional safe_context_text (string only) if Brain explicitly provides it
# - Robust retries (exp backoff + jitter), session reuse, circuit breaker
# - Deterministic prompt truncation, clean error types
# - Works smoothly with Brain.ask(text, context=MemoryContext)

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import requests


logger = logging.getLogger("QwenEngine")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# -----------------------------
# Config + Errors
# -----------------------------

@dataclass(frozen=True)
class GenerationConfig:
    n_predict: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.15
    stop: Optional[list[str]] = None


class LLMError(RuntimeError):
    """Base class for LLM engine errors."""


class LLMTimeoutError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMHTTPError(LLMError):
    pass


class LLMCircuitOpenError(LLMError):
    pass


# -----------------------------
# Engine
# -----------------------------

class QwenEngine:
    """
    Pure text completion engine.
    - It is NOT a chatbot.
    - It does NOT manage history.
    - It does NOT add System/User/Assistant roles.
    - It does NOT inspect or stringify context objects.

    Expected usage from Brain:
        engine.ask(text, context=memory_ctx)

    Optional safe injection:
        engine.ask(prompt, safe_context_text=rendered_context)
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        timeout: float = 60.0,
        max_prompt_chars: int = 24_000,
        debug: bool = False,
        # circuit breaker
        breaker_fail_threshold: int = 5,
        breaker_cooldown_sec: int = 20,
        # retry/backoff
        base_backoff_sec: float = 0.7,
        max_backoff_sec: float = 6.0,
        jitter_sec: float = 0.25,
        # transport
        endpoint: str = "/completion",
        user_agent: str = "QwenEngine/2.0",
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_prompt_chars = int(max_prompt_chars)

        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})

        # breaker state
        self._fail_count = 0
        self._breaker_open_until = 0.0
        self._breaker_fail_threshold = int(breaker_fail_threshold)
        self._breaker_cooldown_sec = float(breaker_cooldown_sec)

        # retry/backoff tuning
        self._base_backoff_sec = float(base_backoff_sec)
        self._max_backoff_sec = float(max_backoff_sec)
        self._jitter_sec = float(jitter_sec)

        if debug:
            logger.setLevel(logging.DEBUG)

    # -----------------------------
    # Public API
    # -----------------------------

    def ask(
        self,
        text: str,
        context: Optional[Any] = None,  # accepted, intentionally ignored
        *,
        safe_context_text: Optional[str] = None,
        cfg: Optional[GenerationConfig] = None,
        retries: int = 2,
        stream: bool = False,
        request_meta: Optional[Dict[str, Any]] = None,  # logging only
        **_ignored_kwargs,  # forward-compat so new Brain args won't crash engine
    ) -> str:
        prompt = (text or "").strip()
        if not prompt:
            return ""

        # Circuit breaker: fail fast if server is in a bad state
        now = time.time()
        if now < self._breaker_open_until:
            raise LLMCircuitOpenError("Circuit breaker open: server recently failing.")

        if safe_context_text:
            safe_context_text = safe_context_text.strip()
            if safe_context_text:
                # IMPORTANT: safe_context_text must be plain text rendered by Brain.
                prompt = f"{safe_context_text}\n\n{prompt}"

        prompt = self._truncate_prompt(prompt)

        cfg = cfg or GenerationConfig()

        if stream:
            return self._stream_completion(prompt, cfg, retries=retries, request_meta=request_meta)
        return self._completion(prompt, cfg, retries=retries, request_meta=request_meta)

    def healthcheck(self) -> bool:
        """
        Best-effort quick check that server is reachable.
        Returns True/False. Does not raise.
        """
        try:
            _ = self._session.get(self.server_url, timeout=min(3.0, self.timeout))
            return True
        except Exception:
            return False

    # -----------------------------
    # Internals: completion calls
    # -----------------------------

    def _completion(
        self,
        prompt: str,
        cfg: GenerationConfig,
        *,
        retries: int,
        request_meta: Optional[Dict[str, Any]],
    ) -> str:
        payload = self._build_payload(prompt, cfg, stream=False)

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                logger.debug("LLM request meta=%s", request_meta)
                r = self._session.post(
                    f"{self.server_url}{self.endpoint}",
                    json=payload,
                    timeout=self.timeout,
                )

                # Treat 5xx as transient, 4xx as permanent (usually bad payload)
                if 500 <= r.status_code <= 599:
                    raise LLMHTTPError(f"Server error {r.status_code}")
                if 400 <= r.status_code <= 499:
                    # Permanent: don't retry much, but allow one retry if asked
                    r.raise_for_status()

                r.raise_for_status()

                data = self._safe_json(r)
                text_out = (data.get("content") or "").strip()

                self._mark_success()
                return text_out

            except requests.exceptions.Timeout as e:
                last_exc = e
                self._mark_failure()
                if attempt >= retries:
                    raise LLMTimeoutError("LLM request timed out.") from e

            except requests.exceptions.ConnectionError as e:
                last_exc = e
                self._mark_failure()
                if attempt >= retries:
                    raise LLMConnectionError("LLM connection failed.") from e

            except requests.exceptions.HTTPError as e:
                # 4xx tends to be permanent, but keep one retry path if configured
                last_exc = e
                self._mark_failure()
                if attempt >= retries:
                    raise LLMHTTPError(f"HTTP error: {e}") from e

            except LLMHTTPError as e:
                # Transient 5xx path
                last_exc = e
                self._mark_failure()
                if attempt >= retries:
                    raise

            except Exception as e:
                last_exc = e
                self._mark_failure()
                if attempt >= retries:
                    raise LLMError(f"Unexpected LLM error: {e}") from e

            self._sleep_backoff(attempt)

        raise LLMError(f"LLM request failed: {last_exc}")

    def _stream_completion(
        self,
        prompt: str,
        cfg: GenerationConfig,
        *,
        retries: int,
        request_meta: Optional[Dict[str, Any]],
    ) -> str:
        """
        Streaming parser for llama.cpp SSE-style "data: {json}" lines.
        If your server streams differently, adjust parse logic.
        """
        payload = self._build_payload(prompt, cfg, stream=True)

        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            collected: list[str] = []
            try:
                logger.debug("LLM stream request meta=%s", request_meta)
                with self._session.post(
                    f"{self.server_url}{self.endpoint}",
                    json=payload,
                    timeout=self.timeout,
                    stream=True,
                ) as r:
                    if 500 <= r.status_code <= 599:
                        raise LLMHTTPError(f"Server error {r.status_code}")
                    r.raise_for_status()

                    for line in self._iter_sse_lines(r.iter_lines()):
                        # Expect {"content": "...", "stop": bool} chunks
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        token = chunk.get("content", "")
                        if token:
                            collected.append(token)

                        if chunk.get("stop"):
                            break

                text_out = "".join(collected).strip()
                self._mark_success()
                return text_out

            except requests.exceptions.Timeout as e:
                last_exc = e
                self._mark_failure()
                if attempt >= retries:
                    raise LLMTimeoutError("LLM stream timed out.") from e

            except requests.exceptions.ConnectionError as e:
                last_exc = e
                self._mark_failure()
                if attempt >= retries:
                    raise LLMConnectionError("LLM stream connection failed.") from e

            except requests.exceptions.HTTPError as e:
                last_exc = e
                self._mark_failure()
                if attempt >= retries:
                    raise LLMHTTPError(f"HTTP error: {e}") from e

            except LLMHTTPError as e:
                last_exc = e
                self._mark_failure()
                if attempt >= retries:
                    raise

            except Exception as e:
                last_exc = e
                self._mark_failure()
                if attempt >= retries:
                    raise LLMError(f"Unexpected LLM stream error: {e}") from e

            self._sleep_backoff(attempt)

        raise LLMError(f"LLM stream request failed: {last_exc}")

    # -----------------------------
    # Helpers
    # -----------------------------

    def _build_payload(self, prompt: str, cfg: GenerationConfig, *, stream: bool) -> Dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": int(cfg.n_predict),
            "temperature": float(cfg.temperature),
            "top_p": float(cfg.top_p),
            "top_k": int(cfg.top_k),
            "repeat_penalty": float(cfg.repeat_penalty),
            "stop": cfg.stop or [],
            "stream": bool(stream),
        }

    def _truncate_prompt(self, prompt: str) -> str:
        if len(prompt) <= self.max_prompt_chars:
            return prompt
        # Deterministic truncation: keep end, where the actual user request usually is.
        return prompt[-self.max_prompt_chars :]

    def _safe_json(self, response: requests.Response) -> Dict[str, Any]:
        try:
            data = response.json()
            if isinstance(data, dict):
                return data
            return {"content": str(data)}
        except Exception:
            # fallback: treat raw as content
            return {"content": (response.text or "")}

    def _iter_sse_lines(self, lines: Iterable[bytes]) -> Iterable[str]:
        """
        Converts iter_lines() output into JSON strings by stripping "data:" prefixes.
        Supports both bytes and str lines.
        """
        for raw in lines:
            if not raw:
                continue
            if isinstance(raw, bytes):
                line = raw.decode("utf-8", errors="ignore").strip()
            else:
                line = str(raw).strip()

            # Common SSE: "data: {...}"
            if line.startswith("data:"):
                line = line[5:].strip()

            # ignore keepalive / empty
            if not line or line == "[DONE]":
                continue

            yield line

    def _sleep_backoff(self, attempt: int) -> None:
        # exponential backoff with jitter, clamped
        base = self._base_backoff_sec * (2 ** attempt)
        base = min(base, self._max_backoff_sec)
        jitter = random.uniform(0.0, self._jitter_sec)
        time.sleep(base + jitter)

    def _mark_success(self) -> None:
        self._fail_count = 0
        self._breaker_open_until = 0.0

    def _mark_failure(self) -> None:
        self._fail_count += 1
        if self._fail_count >= self._breaker_fail_threshold:
            self._breaker_open_until = time.time() + self._breaker_cooldown_sec
            logger.warning("Circuit breaker opened for %.1fs", self._breaker_cooldown_sec)


# Convenience
def quick_ask(text: str, **kwargs) -> str:
    return QwenEngine(**kwargs).ask(text)


if __name__ == "__main__":
    engine = QwenEngine(debug=True)
    print("QwenEngine ready. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        try:
            out = engine.ask(user_input)
            print(f"AI: {out}\n")
        except LLMCircuitOpenError as e:
            print(f"AI: (server busy) {e}\n")
        except LLMTimeoutError:
            print("AI: (timeout) Try a shorter prompt.\n")
        except LLMError as e:
            print(f"AI: (error) {e}\n")
