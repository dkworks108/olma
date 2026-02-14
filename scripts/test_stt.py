from app.stt.vosk_listener import VoskListener

VOSK_MODEL_PATH = "models/vosk/vosk-model-small-hi-0.22"

listener = VoskListener(VOSK_MODEL_PATH)

while True:
    text = listener.listen()
    print(f" Recognized Text: {text}")
