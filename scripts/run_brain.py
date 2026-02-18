from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from app.core.brain import Brain, ResponseType
from app.core.memory.store import MemoryStore
from app.intent.intent_engine import IntentEngine
from app.llm.qwen_engine import QwenEngine
from app.tts.piper_engine import PiperEngine


PIPER_MODEL = os.path.join(BASE_DIR, "models/piper/en_US-amy-medium.onnx")
QWEN_SERVER = "http://127.0.0.1:8080"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_brain",
        description="SUNNY_AI — Offline AI Brain (Text → LLM → Voice)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--no-memory-dump", action="store_true")
    return parser.parse_args()


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _build_engines(args: argparse.Namespace):
    print("  Loading intent engine...", end=" ", flush=True)
    try:
        intent = IntentEngine()
        print("OK")
    except Exception as exc:
        print(f"FAILED\n  ✗ IntentEngine: {exc}")
        sys.exit(1)

    print("  Loading LLM engine...", end=" ", flush=True)
    try:
        llm = QwenEngine(server_url=QWEN_SERVER)
        print("OK")
    except Exception as exc:
        print(f"FAILED\n  ✗ QwenEngine: {exc}")
        sys.exit(1)

    tts = None
    if not args.no_tts:
        print("  Loading TTS engine...", end=" ", flush=True)
        try:
            tts = PiperEngine(PIPER_MODEL)
            print("OK")
        except Exception as exc:
            print(f"FAILED (voice disabled)\n  ✗ PiperEngine: {exc}")

    return intent, llm, tts


def _dump_memory(brain: Brain) -> None:
    snapshot = brain.memory_snapshot()
    if not snapshot:
        print("\n  (No memories recorded this session.)")
        return

    print(f"\n{'─' * 48}")
    print(f"  SESSION MEMORY  ({len(snapshot)} records)")
    print(f"{'─' * 48}")
    for rec in snapshot:
        ttl_label = f"  ttl={int(rec['ttl'])}s" if rec.get("ttl") else "  permanent"
        print(
            f"  [{rec['type']:8}] {rec['key']:30} = {rec['value']}"
            f"  (conf={rec['confidence']:.2f}{ttl_label})"
        )
    print(f"{'─' * 48}\n")


def _handle_response(resp: dict, tts: PiperEngine | None) -> bool:
    rtype = resp.get("type")
    reply = resp.get("reply", "")

    if rtype == ResponseType.SILENT:
        return True

    if rtype == ResponseType.SYSTEM:
        print(f"\n  [{rtype}] {reply}")
        if tts and reply:
            tts.speak(reply)
        return False

    if rtype == ResponseType.ERROR:
        print(f"\n  [!] {reply}")
        return True

    if rtype == ResponseType.ACTION:
        action = resp.get("action", "UNKNOWN")
        print(f"\n  [ACTION → {action}] {reply}")
        if tts and reply:
            tts.speak(reply)
        return True

    print(f"\n  AI : {reply}\n")
    if tts and reply:
        try:
            tts.speak(reply)
        except Exception as exc:
            logging.getLogger("run_brain").warning("TTS failed: %s", exc)

    return True


def main() -> None:
    args = _parse_args()
    _setup_logging(args.debug)

    print("\n" + "═" * 50)
    print("  SUNNY_AI  —  Offline Brain")
    print("  Type 'exit' or press Ctrl+C to quit.")
    print("═" * 50 + "\n")

    print("  Initialising engines...")
    intent_engine, llm_engine, tts_engine = _build_engines(args)
    print()

    store = MemoryStore(max_records=1_000)

    brain = Brain(
        intent_engine=intent_engine,
        llm_engine=llm_engine,
        memory_store=store,
        extract_llm_output=True,
        purge_interval=300.0,
    )

    running = True

    def _on_sigint(sig, frame):
        nonlocal running
        print("\n\n  Interrupted. Shutting down...\n")
        running = False

    signal.signal(signal.SIGINT, _on_sigint)

    print("  Ready. Listening...\n")

    while running:
        try:
            user_text = input("  You : ").strip()
        except EOFError:
            break

        if not user_text:
            continue

        if user_text.lower() in ("exit", "quit", "bye"):
            print("\n  Exiting. Goodbye!\n")
            if tts_engine:
                try:
                    tts_engine.speak("Goodbye.")
                except Exception:
                    pass
            break

        print("  Thinking...", flush=True)

        resp = brain.process(user_text)

        should_continue = _handle_response(resp, tts_engine)
        if not should_continue:
            break

    if not args.no_memory_dump:
        _dump_memory(brain)

    print("  Brain offline.\n")


if __name__ == "__main__":
    main()
