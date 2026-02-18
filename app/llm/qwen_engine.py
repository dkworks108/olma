import re
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("QwenEngine")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@dataclass
class Message:
    role: str
    content: str


@dataclass
class GenerationConfig:
    n_predict: int = 256
    temperature: float = 0.3
    top_p: float = 0.90
    top_k: int = 40
    repeat_penalty: float = 1.15
    stop: list = field(default_factory=lambda: [
        "User:", "Assistant:", "\nUser", "\nAssistant"
    ])


def _analyze(text: str) -> dict:
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "has_question": text.strip().endswith("?") or text.lower().startswith(
            ("what", "who", "where", "when", "why", "how", "is ", "are ", "can ", "does ")
        ),
        "has_numbers": bool(re.search(r"\d", text)),
        "is_code_request": any(
            kw in text.lower()
            for kw in ["code", "function", "script", "write", "implement", "debug", "class", "def ", "import"]
        ),
        "is_math": bool(re.search(r"[\d\+\-\*/=\^%]", text)) and len(words) <= 12,
        "is_short": len(words) <= 5,
        "is_very_short": len(words) <= 2,
    }


def _build_config(signals: dict) -> GenerationConfig:
    cfg = GenerationConfig()

    if signals["is_code_request"]:
        cfg.n_predict = 512
    elif signals["is_short"] or signals["is_math"]:
        cfg.n_predict = 128
    else:
        cfg.n_predict = min(64 + signals["word_count"] * 12, 384)

    if signals["is_math"] or signals["has_numbers"]:
        cfg.temperature = 0.05
    elif signals["is_code_request"]:
        cfg.temperature = 0.15
    elif signals["is_short"]:
        cfg.temperature = 0.25
    else:
        cfg.temperature = 0.35

    if signals["is_code_request"]:
        cfg.top_p = 0.95
        cfg.top_k = 50
    elif signals["is_math"]:
        cfg.top_p = 0.80
        cfg.top_k = 20
    else:
        cfg.top_p = 0.90
        cfg.top_k = 40

    return cfg


SYSTEM_PROMPT = (
    "You are a concise, accurate assistant. "
    "Answer directly without unnecessary preamble. "
    "For code, output only the code block. "
    "For math, show the answer and a brief explanation."
)


def _build_messages_prompt(history: list[Message], user_text: str) -> str:
    lines = [f"System: {SYSTEM_PROMPT}\n"]
    for msg in history:
        role = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    lines.append(f"User: {user_text.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _clean(raw: str, signals: dict) -> str:
    if not raw:
        return "Okay."

    for marker in ("Assistant:", "User:", "Output:", "Input:"):
        if marker in raw:
            raw = raw.split(marker)[-1]

    raw = raw.strip()

    if signals.get("is_code_request"):
        return raw[:2000].strip()

    raw = raw.strip('"\'')

    raw = re.sub(r"\n{3,}", "\n\n", raw)

    max_chars = 600
    if len(raw) > max_chars:
        cut = raw[:max_chars]
        for end in (".", "?", "!"):
            idx = cut.rfind(end)
            if idx > max_chars // 2:
                raw = cut[: idx + 1]
                break
        else:
            raw = cut.rsplit(" ", 1)[0] + "…"

    return raw.strip()


class QwenEngine:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        timeout: int = 60,
        max_history: int = 10,
        debug: bool = False,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_history = max_history
        self.history: list[Message] = []

        if debug:
            logger.setLevel(logging.DEBUG)

    def ask(
        self,
        user_text: str,
        *,
        stream: bool = False,
        retries: int = 2,
        override_config: Optional[GenerationConfig] = None,
    ) -> str:
        user_text = user_text.strip()
        if not user_text:
            return ""

        signals = _analyze(user_text)
        cfg = override_config or _build_config(signals)
        prompt = _build_messages_prompt(self.history, user_text)

        logger.debug("Prompt:\n%s", prompt)
        logger.debug("Config: %s", cfg)

        raw = self._call_with_retry(prompt, cfg, stream=stream, retries=retries)
        response = _clean(raw, signals)

        self._add_to_history(Message("user", user_text))
        self._add_to_history(Message("assistant", response))

        return response

    def clear_history(self):
        self.history.clear()
        logger.debug("History cleared.")

    def show_history(self) -> str:
        if not self.history:
            return "(No history)"
        return "\n".join(
            f"[{m.role.upper()}] {m.content}" for m in self.history
        )

    def _add_to_history(self, msg: Message):
        self.history.append(msg)
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def _call_with_retry(
        self,
        prompt: str,
        cfg: GenerationConfig,
        stream: bool,
        retries: int,
    ) -> str:
        last_error = None
        for attempt in range(retries + 1):
            try:
                if stream:
                    return self._stream_completion(prompt, cfg)
                else:
                    return self._completion(prompt, cfg)
            except requests.exceptions.Timeout:
                last_error = "timeout"
                logger.warning("Attempt %d timed out.", attempt + 1)
            except requests.exceptions.ConnectionError:
                last_error = "connection"
                logger.warning("Attempt %d: connection error.", attempt + 1)
            except requests.exceptions.HTTPError:
                return "The model returned an error."
            except Exception:
                return "Something went wrong."

            if attempt < retries:
                time.sleep(2 ** attempt)

        if last_error == "timeout":
            return "The model is taking too long. Try a shorter prompt."
        return "I can't reach the model. Is the server running?"

    def _completion(self, prompt: str, cfg: GenerationConfig) -> str:
        payload = {
            "prompt": prompt,
            "n_predict": cfg.n_predict,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repeat_penalty": cfg.repeat_penalty,
            "stop": cfg.stop,
            "stream": False,
        }
        r = requests.post(
            f"{self.server_url}/completion",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("content", "").strip()

    def _stream_completion(self, prompt: str, cfg: GenerationConfig) -> str:
        payload = {
            "prompt": prompt,
            "n_predict": cfg.n_predict,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "repeat_penalty": cfg.repeat_penalty,
            "stop": cfg.stop,
            "stream": True,
        }
        collected = []
        with requests.post(
            f"{self.server_url}/completion",
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data:"):
                    import json
                    try:
                        chunk = json.loads(line[5:].strip())
                        token = chunk.get("content", "")
                        print(token, end="", flush=True)
                        collected.append(token)
                        if chunk.get("stop"):
                            break
                    except json.JSONDecodeError:
                        continue
        print()
        return "".join(collected).strip()


def quick_ask(text: str, **kwargs) -> str:
    return QwenEngine(**kwargs).ask(text)


if __name__ == "__main__":
    engine = QwenEngine(debug=True)
    print("QwenEngine ready. Type 'quit' to exit, 'history' to show chat.\n")
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
        if user_input.lower() == "history":
            print(engine.show_history())
            continue
        if user_input.lower() == "clear":
            engine.clear_history()
            print("History cleared.")
            continue

        response = engine.ask(user_input)
        print(f"Assistant: {response}\n")
