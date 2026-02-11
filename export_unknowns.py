# export_unknowns.py
import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = os.path.join(DATA_DIR, "chatlog.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "unknowns.csv")


def export_unknowns():
    if not os.path.exists(LOG_FILE):
        msg = f"Keine Log-Datei '{LOG_FILE}' gefunden."
        print(msg)
        return {
            "ok": False,
            "message": msg,
            "total_rows": 0,
            "unknown_rows": 0,
            "output_file": OUTPUT_FILE,
        }

    total_rows = 0
    unknown_rows = 0

    with open(LOG_FILE, mode="r", encoding="utf-8") as f_in, open(
        OUTPUT_FILE, mode="w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        writer.writerow(
            ["timestamp", "user_text", "bot_answer", "intent", "note", "new_intent"]
        )

        for row in reader:
            total_rows += 1
            if len(row) < 4:
                continue
            timestamp, user_text, bot_answer, intent = (row + ["", "", ""])[:4]
            note = row[4] if len(row) > 4 else ""
            if (intent or "").strip() == "unknown":
                writer.writerow([timestamp, user_text, bot_answer, intent, note, ""])
                unknown_rows += 1

    msg = (
        f"Unbekannte Intents exportiert nach '{OUTPUT_FILE}': {unknown_rows} "
        f"(Gesamtzeilen: {total_rows})"
    )
    print(msg)
    return {
        "ok": True,
        "message": msg,
        "total_rows": total_rows,
        "unknown_rows": unknown_rows,
        "output_file": OUTPUT_FILE,
    }


if __name__ == "__main__":
    export_unknowns()