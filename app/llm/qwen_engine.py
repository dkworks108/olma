import requests


class QwenEngine:
    def __init__(self, server_url="http://127.0.0.1:8080", timeout=60):
        self.server_url = server_url
        self.timeout = timeout

    def _build_prompt(self, user_text: str) -> str:
        return (
            "Respond naturally and briefly to the user's message.\n"
            "Do not ask questions.\n"
            "Do not continue the conversation.\n"
            "Do not mention rules or instructions.\n\n"
            f"User: {user_text.strip()}\n"
            "Answer:"
        )

    def ask(self, user_text: str) -> str:
        if not user_text or not user_text.strip():
            return ""

        payload = {
            "prompt": self._build_prompt(user_text),
            "n_predict": 128,
            "temperature": 0.4,
            "top_p": 0.85,
            "repeat_penalty": 1.15,
            "stop": ["User:", "Answer:", "\nUser", "\nAnswer"],
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
                return "I'm here."

            # HARD SANITIZATION (prompt leak protection)
            for marker in ["User:", "Answer:", "STRICT", "RULE", "Note that"]:
                if marker in raw:
                    raw = raw.split(marker)[0].strip()

            return raw

        except requests.exceptions.Timeout:
            return "I paused for a moment."

        except requests.exceptions.ConnectionError:
            return "I can't reach my brain right now."

        except Exception:
            return "Something went wrong."
