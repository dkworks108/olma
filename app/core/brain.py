class Brain:
    def __init__(self, intent_engine, llm_engine):
        self.intent_engine = intent_engine
        self.llm_engine = llm_engine

    def process(self, text: str) -> dict:
        intent = self.intent_engine.detect_intent(text)

        if intent["intent"] == "SYSTEM":
            return {
                "type": "SYSTEM",
                "reply": "सिस्टम बंद किया जा रहा है।"
            }

        if intent["intent"] == "COMMAND":
            action = intent.get("action", "UNKNOWN")

            reply_map = {
                "MOVE_FORWARD": "आदेश समझ लिया। आगे बढ़ रहा हूँ।",
                "MOVE_BACKWARD": "ठीक है, पीछे जा रहा हूँ।",
                "TURN_LEFT": "बाईं ओर मुड़ रहा हूँ।",
                "TURN_RIGHT": "दाईं ओर मुड़ रहा हूँ।",
                "STOP": "रुक गया हूँ।"
            }

            return {
                "type": "ACTION",
                "action": action,
                "reply": reply_map.get(action, "आदेश समझ लिया।")
            }

        reply = self.llm_engine.ask(text)

        return {
            "type": "SPEECH",
            "reply": reply
        }
