# manage_bot.py

import subprocess
import sys


def print_help():
    print("Verwendung:")
    print("  python3 manage_bot.py chat              # Chatbot starten")
    print("  python3 manage_bot.py train             # Modell aus Logs + unknowns trainieren")
    print("  python3 manage_bot.py analyze           # Logs analysieren")
    print("  python3 manage_bot.py export_unknowns   # unknown-Intents in unknowns.csv exportieren")
    print()
    print("Beispiel-Workflow:")
    print("  1) python3 manage_bot.py chat")
    print("  2) python3 manage_bot.py export_unknowns")
    print("  3) unknowns.csv in Excel/Numbers labeln (Spalte 'new_intent')")
    print("  4) python3 manage_bot.py train")
    print("  5) python3 manage_bot.py chat  # mit neuem Modell")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_help()
        return

    cmd = sys.argv[1]

    if cmd == "chat":
        subprocess.run(["python3", "chatbot.py"])
    elif cmd == "train":
        subprocess.run(["python3", "train_from_logs.py"])
    elif cmd == "analyze":
        subprocess.run(["python3", "analyze_logs.py"])
    elif cmd == "export_unknowns":
        subprocess.run(["python3", "export_unknowns.py"])
    else:
        print(f"Unbekannter Befehl: {cmd}")
        print_help()


if __name__ == "__main__":
    main()