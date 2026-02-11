# export_unknowns.py

import os
import csv

# === Pfade ===

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = os.path.join(DATA_DIR, "chatlog.csv")
UNKNOWN_FILE = os.path.join(DATA_DIR, "unknowns.csv")


def export_unknowns():
    """
    Liest data/chatlog.csv und schreibt alle Zeilen mit Intent 'unknown'
    nach data/unknowns.csv.

    Rückgabe:
      {
        "ok": bool,
        "message": str
      }
    """
    if not os.path.exists(LOG_FILE):
        return {
            "ok": False,
            "message": f"Keine Log-Datei '{LOG_FILE}' gefunden. "
                       "Bitte zuerst im Chat ein paar Nachrichten schreiben.",
        }

    count = 0

    try:
        with open(LOG_FILE, mode="r", encoding="utf-8", newline="") as f_in, \
             open(UNKNOWN_FILE, mode="w", encoding="utf-8", newline="") as f_out:

            reader = csv.reader(f_in)
            writer = csv.writer(f_out)

            for row in reader:
                # Erwartetes Schema:
                # [timestamp, user_text, bot_answer, intent, note]
                if len(row) >= 4 and row[3] == "unknown":
                    writer.writerow(row)
                    count += 1

    except Exception as e:
        return {
            "ok": False,
            "message": f"Fehler beim Lesen oder Schreiben: {e}",
        }

    if count == 0:
        return {
            "ok": True,
            "message": "Keine Unknown-Intents in den Logs gefunden.",
        }

    return {
        "ok": True,
        "message": f"{count} Unknown-Intents nach '{UNKNOWN_FILE}' exportiert.",
    }


if __name__ == "__main__":
    result = export_unknowns()
    print(result)