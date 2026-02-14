class IntentEngine:
    def __init__(self):
        self.command_keywords = {
            "MOVE_FORWARD": ["आगे", "चलो", "आगे बढ़ो"],
            "MOVE_BACKWARD": ["पीछे", "वापस"],
            "STOP": ["रुको", "बंद"],
            "TURN_LEFT": ["बाएं", "बाईं"],
            "TURN_RIGHT": ["दाएं", "दाईं"]
        }

        self.system_keywords = ["बंद हो जाओ", "shutdown", "exit"]

    def detect_intent(self, text: str) -> dict:
        text = text.strip()

        # 1️⃣ System intent
        for word in self.system_keywords:
            if word in text:
                return {
                    "intent": "SYSTEM",
                    "action": "EXIT"
                }

        # 2️⃣ Command intent
        for action, keywords in self.command_keywords.items():
            for word in keywords:
                if word in text:
                    return {
                        "intent": "COMMAND",
                        "action": action
                    }

        # 3️⃣ Question intent
        if text.endswith("क्या") or "क्या" in text:
            return {
                "intent": "QUESTION",
                "text": text
            }

        # 4️⃣ Default: Chat
        return {
            "intent": "CHAT",
            "text": text
        }
