import requests


class QwenEngine:
    def __init__(self, server_url="http://127.0.0.1:8080"):
        self.server_url = server_url

    def ask(self, user_text: str) -> str:
        if not user_text.strip():
            return ""

        prompt = (
    "You are my close childhood friend of over 10 years. "
    "We talk casually and comfortably, like real friends do. "
    "This conversation is ongoing and never resets. "
    "My name is Doresh.\n\n"

    "RULES:\n"
    "- You speak ONLY as my friend.\n"
    "- You reply to my last message only.\n"
    "- You never describe yourself as an AI.\n"
    
    "- You never generate dialogue for me.\n"
    "- You never interview me.\n"
    "- You talk naturally, like a real person.\n\n"

    "STYLE:\n"
    "- Short, relaxed sentences.\n"
    "- Friendly, familiar tone.\n"
    "- No emojis.\n"
    "- No generic chatbot phrases.\n\n"

    f"Doresh: {user_text}\n"
    "Friend:"
)


        payload = {
            "prompt": prompt,
            "n_predict": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "repeat_penalty": 1.05,
        }

        try:
            r = requests.post(
                f"{self.server_url}/completion",
                json=payload,
                timeout=90,
            )
            r.raise_for_status()
            return r.json().get("content", "").strip() or "I'm here—say that again."

        except requests.exceptions.Timeout:
            return "Give me a second—something stalled."

        except Exception:
            return "Something went wrong on my side."
