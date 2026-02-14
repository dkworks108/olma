from app.intent.intent_engine import IntentEngine

engine = IntentEngine()

test_inputs = [
    "आगे बढ़ो",
    "रुको",
    "तुम क्या कर सकते हो",
    "नमस्ते दोस्त"
]

for text in test_inputs:
    result = engine.detect_intent(text)
    print(f"INPUT: {text}")
    print(f"INTENT: {result}\n")
