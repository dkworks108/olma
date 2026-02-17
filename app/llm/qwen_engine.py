import requests
import time


class QwenEngine:
    def __init__(self, server_url="http://127.0.0.1:8080", timeout=60):
        self.server_url = server_url
        self.timeout = timeout

    def _build_prompt(self, user_text: str) -> str:
        """
        Phase-1 SAFE PROMPT
        Single-turn response only.
        """

        return (
            "You are a friendly human companion.\n"
            "Respond naturally to the user's message.\n\n"

            "STRICT RULES:\n"
            "- Write ONLY one reply.\n"
            "- Do NOT continue the conversation.\n"
            "- Do NOT ask follow-up questions.\n"
            "- Do NOT write dialogue or names.\n"
            "- Do NOT simulate another speaker.\n"
            "- End your response clearly.\n\n"

            f"User: {user_text.strip()}\n"
            "Response:"
        )

    def ask(self, user_text: str) -> str:
        if not user_text or not user_text.strip():
            return ""

        payload = {
            "prompt": self._build_prompt(user_text),
            "n_predict": 128,        # HARD LIMIT
            "temperature": 0.5,      # Stability > creativity
            "top_p": 0.85,
            "repeat_penalty": 1.15,
            "stop": ["User:", "Response:", "\nUser", "\nResponse"],
        }

        try:
            response = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            text = response.json().get("content", "").strip()

            if not text:
                return "I'm here."

            # FINAL SAFETY TRIM
            for bad in ["User:", "Doresh:", "Friend:", "Doosh:"]:
                if bad in text:
                    text = text.split(bad)[0].strip()

            return text

        except requests.exceptions.Timeout:
            return "I paused for a second."

        except requests.exceptions.ConnectionError:
            return "I can't reach my brain right now."

        except Exception:
            return "Something went wrong."
