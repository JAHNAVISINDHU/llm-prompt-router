#!/usr/bin/env python3
"""
LLM-Powered Prompt Router — Interactive CLI
Run: python main.py
"""

import sys
from router import process_message, VALID_INTENTS, LOG_FILE

BANNER = """
╔══════════════════════════════════════════════════════╗
║         🤖  LLM-Powered Prompt Router  🤖           ║
║  Intelligently routes your query to the right expert ║
╚══════════════════════════════════════════════════════╝

Supported expert domains:
  • code          — Software engineering & debugging
  • data_analysis — Statistics, ML & data science
  • writing       — Drafting, editing & communication
  • career        — Resume, interviews & job search

Type 'exit' or 'quit' to stop.
Type 'log'  to view the last 5 log entries.
Type 'help' to see this menu again.
──────────────────────────────────────────────────────
"""


def show_log(n: int = 5) -> None:
    """Print the last n entries from the log file."""
    import json
    if not LOG_FILE.exists():
        print("  (No log file found yet.)\n")
        return
    lines = LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        print("  (Log is empty.)\n")
        return
    recent = lines[-n:]
    print(f"\n── Last {len(recent)} log entries ──────────────────────")
    for line in recent:
        try:
            entry = json.loads(line)
            print(f"  [{entry.get('timestamp','')}]")
            print(f"  User   : {entry.get('user_message','')[:80]}")
            print(f"  Intent : {entry.get('intent','')}  (confidence: {entry.get('confidence',0):.2f})")
            print(f"  Reply  : {entry.get('final_response','')[:100]}...")
            print()
        except Exception:
            print(f"  (Malformed entry: {line[:80]})")
    print("────────────────────────────────────────────────────\n")


def main() -> None:
    print(BANNER)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nBye! 👋")
            sys.exit(0)

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit"):
            print("Bye! 👋")
            break
        if cmd == "help":
            print(BANNER)
            continue
        if cmd == "log":
            show_log()
            continue

        print("\n⏳ Classifying intent…", end="", flush=True)

        try:
            result = process_message(user_input)
        except EnvironmentError as e:
            print(f"\n\n❌ Configuration error: {e}\n")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n❌ Unexpected error: {e}\n")
            continue

        intent_label = result["intent"].upper()
        confidence_pct = result["confidence"] * 100

        print(f"\r✅ Intent: {intent_label:<16} Confidence: {confidence_pct:.0f}%")
        print("──────────────────────────────────────────────────────")
        print(f"🤖 Assistant:\n\n{result['final_response']}")
        print("──────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
