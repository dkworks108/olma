from app.llm.qwen_engine import QwenEngine
from app.tts.piper_engine import PiperEngine
import os


def main():
    print("\nOffline AI Brain Started (Text to LLM to Voice)")
    print("Type 'exit' to quit.\n")

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    QWEN_MODEL = os.path.join(
        BASE_DIR,
        "models/qwen/qwen2.5-0.5b-instruct-q4_0.gguf"
    )

    LLAMA_BIN = os.path.join(
        BASE_DIR,
        "models/llama.cpp/build/bin/llama-cli"
    )

    llm = QwenEngine(
        model_path=QWEN_MODEL,
        llama_bin=LLAMA_BIN,
        ctx=2048
    )

    tts = PiperEngine(
        "models/piper/hi_IN-priyamvada-medium.onnx"
    )

    while True:
        user_text = input("You: ").strip()

        if not user_text:
            continue

        if user_text.lower() in ("exit", "quit"):
            print("Exiting brain.")
            break

        print("Thinking...")
        reply = llm.ask(user_text)

        print(f"AI: {reply}\n")
        tts.speak(reply)


if __name__ == "__main__":
    main()
