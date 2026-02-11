# analyze_logs.py
import os
import csv
import re
from collections import Counter, defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = os.path.join(DATA_DIR, "chatlog.csv")
WORD_RE = re.compile(r"[a-zA-ZäöüÄÖÜß]+", re.UNICODE)


def load_logs():
    if not os.path.exists(LOG_FILE):
        return []

    rows = []
    with open(LOG_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            timestamp, user_text, bot_answer, intent = (row + ["", "", ""])[:4]
            note = row[4] if len(row) > 4 else ""
            rows.append(
                {
                    "timestamp": timestamp.strip(),
                    "user_text": user_text.strip(),
                    "bot_answer": bot_answer.strip(),
                    "intent": intent.strip(),
                    "note": note.strip(),
                }
            )
    return rows


def tokenize(text: str):
    text = text.lower()
    return WORD_RE.findall(text)


def compute_basic_stats(rows):
    total = len(rows)
    msg_rows = [r for r in rows if r["user_text"]]
    return {
        "total_rows": total,
        "rows_with_user_text": len(msg_rows),
    }


def compute_intent_stats(rows):
    intents = [r["intent"] or "EMPTY" for r in rows]
    counter = Counter(intents)
    total = sum(counter.values()) or 1

    stats = []
    for intent, count in counter.most_common():
        stats.append(
            {
                "intent": intent,
                "count": count,
                "percent": (count / total) * 100.0,
            }
        )

    unknown_count = counter.get("unknown", 0)
    return {
        "by_intent": stats,
        "unknown_count": unknown_count,
        "total": total,
        "unknown_percent": (unknown_count / total * 100.0) if total else 0.0,
    }


def compute_day_stats(rows):
    per_day = Counter()
    for r in rows:
        ts = r["timestamp"]
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
            day_str = dt.date().isoformat()
            per_day[day_str] += 1
        except ValueError:
            continue

    return [{"day": day, "count": count} for day, count in sorted(per_day.items())]


def examples_per_intent(rows, max_per_intent=5):
    examples = defaultdict(list)
    for r in rows:
        intent = r["intent"] or "EMPTY"
        if len(examples[intent]) < max_per_intent:
            examples[intent].append(r["user_text"])
    return examples


def top_words_overall(rows, top_n=20):
    counter = Counter()
    for r in rows:
        for token in tokenize(r["user_text"]):
            counter[token] += 1
    return counter.most_common(top_n)


def top_words_per_intent(rows, top_n=10, min_count=2):
    words_by_intent = defaultdict(Counter)
    for r in rows:
        intent = r["intent"] or "EMPTY"
        tokens = tokenize(r["user_text"])
        for token in tokens:
            words_by_intent[intent][token] += 1

    result = {}
    for intent, counter in words_by_intent.items():
        filtered = [(w, c) for w, c in counter.items() if c >= min_count]
        filtered.sort(key=lambda x: x[1], reverse=True)
        result[intent] = filtered[:top_n]
    return result


# CLI-Test
if __name__ == "__main__":
    rows = load_logs()
    if not rows:
        print(f"Keine Logs in {LOG_FILE}")
    else:
        basic = compute_basic_stats(rows)
        print("Gesamtzeilen:", basic["total_rows"])
        print("Mit User-Text:", basic["rows_with_user_text"])