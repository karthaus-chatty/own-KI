# train_from_logs.py
import os
import csv
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from chatbot import base_train_data  # nutzt dieselben Basis-Daten

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = os.path.join(DATA_DIR, "chatlog.csv")
UNKNOWNS_FILE = os.path.join(DATA_DIR, "unknowns.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(DATA_DIR, "vectorizer.pkl")
TRAINING_JSON = os.path.join(DATA_DIR, "training_data.json")


def load_json_training_data():
    if not os.path.exists(TRAINING_JSON):
        print(f"[train] Keine '{TRAINING_JSON}' gefunden – überspringe JSON-Daten.")
        return []

    with open(TRAINING_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("data", [])
    result = []
    for item in items:
        text = (item.get("text") or "").strip()
        intent = (item.get("intent") or "").strip()
        if text and intent and intent != "unknown":
            result.append((text, intent))

    print(f"[train] {len(result)} Beispiele aus training_data.json geladen.")
    return result


def load_log_data():
    if not os.path.exists(LOG_FILE):
        print(f"[train] Keine Log-Datei '{LOG_FILE}' gefunden – überspringe Logs.")
        return []

    result = []
    with open(LOG_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            _ts, user_text, _bot_answer, intent = row[:4]
            user_text = (user_text or "").strip()
            intent = (intent or "").strip()
            if not user_text or not intent or intent == "unknown":
                continue
            result.append((user_text, intent))

    print(f"[train] {len(result)} Beispiele aus chatlog.csv geladen.")
    return result


def load_labeled_unknowns():
    if not os.path.exists(UNKNOWNS_FILE):
        print(f"[train] Keine Datei '{UNKNOWNS_FILE}' gefunden – überspringe gelabelte Unknowns.")
        return []

    result = []
    with open(UNKNOWNS_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        has_new_intent = False
        if header and "new_intent" in header:
            new_intent_idx = header.index("new_intent")
            text_idx = header.index("user_text") if "user_text" in header else 1
            has_new_intent = True
        else:
            # Fallback: feste Spaltenpositionen
            new_intent_idx = 5
            text_idx = 1

        if not has_new_intent:
            print("[train] Warnung: Header ohne 'new_intent' – nehme Spalte 5 als new_intent an.")

        for row in reader:
            if len(row) <= max(new_intent_idx, text_idx):
                continue
            user_text = (row[text_idx] or "").strip()
            new_intent = (row[new_intent_idx] or "").strip()
            if not user_text or not new_intent or new_intent == "unknown":
                continue
            result.append((user_text, new_intent))

    print(f"[train] {len(result)} gelabelte Unknowns aus unknowns.csv geladen.")
    return result


def train_model(train_data):
    texts = [t for t, _ in train_data]
    labels = [i for _, i in train_data]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return vectorizer, model


def train_model_from_all_data():
    train_data = list(base_train_data)

    json_data = load_json_training_data()
    train_data.extend(json_data)

    log_data = load_log_data()
    train_data.extend(log_data)

    labeled_unknowns = load_labeled_unknowns()
    train_data.extend(labeled_unknowns)

    if not train_data:
        raise RuntimeError("Keine Trainingsdaten vorhanden!")

    print(f"[train] Gesamtanzahl Trainingsbeispiele: {len(train_data)}")

    vectorizer, model = train_model(train_data)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print(f"[train] Modell und Vektorisierer gespeichert in:\n  {MODEL_FILE}\n  {VECTORIZER_FILE}")


if __name__ == "__main__":
    train_model_from_all_data()