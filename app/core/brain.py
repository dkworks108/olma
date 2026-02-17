class Brain:
    def __init__(self, intent_engine, llm_engine):
        self.intent_engine = intent_engine
        self.llm_engine = llm_engine

    def process(self, text: str) -> dict:
        intent = self.intent_engine.detect_intent(text)

        # SYSTEM COMMAND (Phase-1 English only)
        if intent["intent"] == "SYSTEM":
            return {
                "type": "SYSTEM",
                "reply": "Shutting down the system."
            }

        # ACTION COMMANDS (future hardware layer)
        if intent["intent"] == "COMMAND":
            action = intent.get("action", "UNKNOWN")

            reply_map = {
                "MOVE_FORWARD": "Moving forward.",
                "MOVE_BACKWARD": "Moving backward.",
                "TURN_LEFT": "Turning left.",
                "TURN_RIGHT": "Turning right.",
                "STOP": "Stopped."
            }

            return {
                "type": "ACTION",
                "action": action,
                "reply": reply_map.get(action, "Command received.")
            }

        # NORMAL SPEECH
        reply = self.llm_engine.ask(text)

        return {
            "type": "SPEECH",
            "reply": reply
        }
