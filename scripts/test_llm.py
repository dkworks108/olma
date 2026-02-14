import os
from app.llm.qwen_engine import QwenEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LLAMA_BIN = os.path.join(
    BASE_DIR,
    "models/llama.cpp/build/bin/llama-cli"
)

QWEN_MODEL = os.path.join(
    BASE_DIR,
    "models/qwen/qwen2.5-0.5b-instruct-q4_0.gguf"
)

llm = QwenEngine(QWEN_MODEL, LLAMA_BIN)

print(" Qwen LLM test started. Type 'exit' to quit.\n")

while True:
    text = input("You: ")
    if text.lower() in ["exit", "quit"]:
        break

    reply = llm.ask(text)
    print("AI:", reply, "\n")
