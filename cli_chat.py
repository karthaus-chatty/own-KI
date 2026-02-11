#!/usr/bin/env python3
"""
cli_chat.py – kleines Terminal-Interface für deinen Chatbot.

Nutzung:
    python3 cli_chat.py

Features:
- nutzt dieselbe generate_response-Logik wie die Web-App
- nutzt tweak_answer_with_context für etwas Kontext-Verhalten
- schreibt Logs in data/chatlog.csv (wie die Web-App)
"""

import uuid

from chatbot import (
    generate_response,
    tweak_answer_with_context,
    log_message,
)


def main():
    print("Own-KI CLI 🧠")
    print("Tippe deine Nachrichten. Mit 'exit', 'quit' oder ':q' beenden.\n")

    history = []
    session_id = str(uuid.uuid4())

    while True:
        try:
            user_text = input("Du: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBeende CLI-Chat. 👋")
            break

        if not user_text:
            continue

        if user_text.lower() in ("exit", "quit", ":q"):
            print("Okay, bis zum nächsten Mal 👋")
            break

        # Antwort vom Bot holen
        answer, intent, confidence = generate_response(user_text)
        # Kontext-Tweak
        answer = tweak_answer_with_context(answer, intent, history)

        # Logging (analog zur Web-App)
        note = f"cli_session={session_id};conf={confidence:.3f}"
        try:
            log_message(user_text, answer, intent, note=note)
        except Exception as e:
            print(f"[WARN] Konnte Log nicht schreiben: {e}")

        # Historie updaten
        history.append(
            {
                "role": "user",
                "text": user_text,
                "intent": intent,
                "confidence": confidence,
            }
        )
        history.append(
            {
                "role": "bot",
                "text": answer,
                "intent": intent,
                "confidence": confidence,
            }
        )
        history = history[-20:]

        # Ausgabe
        print(f"Bot [{intent} @ {confidence*100:.1f}%]: {answer}\n")


if __name__ == "__main__":
    main()