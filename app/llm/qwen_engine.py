import requests
import re


class QwenEngine:
    def __init__(self, server_url="http://127.0.0.1:8080", timeout=60):
        self.server_url = server_url
        self.timeout = timeout

    def _analyze_input(self, text: str) -> dict:
        words = text.split()
        length = len(text)
        word_count = len(words)

        return {
            "has_question": "?" in text,
            "has_numbers": bool(re.search(r"\d", text)),
            "word_count": word_count,
            "char_count": length,
            "short": word_count <= 3,
            "very_short": word_count <= 1,
        }

    def _build_prompt(self, user_text: str) -> str:
        return (
            "Give a clear, useful answer.\n"
            "Be concise.\n"
            "Do not add explanations or commentary.\n\n"
            f"Input: {user_text.strip()}\n"
            "Output:"
        )

    def ask(self, user_text: str) -> str:
        if not user_text or not user_text.strip():
            return ""

        signals = self._analyze_input(user_text)

        n_predict = min(
            32 + signals["word_count"] * 8,
            128
        )

        if signals["has_numbers"]:
            temperature = 0.1
        elif signals["short"]:
            temperature = 0.2
        else:
            temperature = 0.3

        top_p = 0.85

        payload = {
            "prompt": self._build_prompt(user_text),
            "n_predict": n_predict,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": 1.2,
            "stop": ["Input:", "Output:", "\nInput", "\nOutput"],
        }

        try:
            r = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()

            raw = r.json().get("content", "").strip()
            if not raw:
                return "Okay."

            raw = raw.split("\n")[0]
            raw = re.sub(r"[^\w\s.,?!]", "", raw)

            for end in [".", "?", "!"]:
                if end in raw:
                    raw = raw.split(end)[0] + end
                    break

            if len(raw) > 140:
                raw = raw[:140].rsplit(" ", 1)[0]

            return raw.strip()

        except requests.exceptions.Timeout:
            return "One moment."

        except requests.exceptions.ConnectionError:
            return "I can't reach the model."

        except Exception:
            return "Something went wrong."
