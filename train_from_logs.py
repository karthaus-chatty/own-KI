# train_from_logs.py
#
# Trainiert das Intent-Modell aus:
# - data/training_data.json (falls vorhanden)
# - Basisdaten aus chatbot.base_train_data
# - data/chatlog.csv (nur Zeilen mit Intent != 'unknown')
# - data/unknowns.csv (nur Zeilen mit Intent != 'unknown', z.B. manuell gelabelte Unknowns)
#
# Ergebnis:
# - Neues Modell + Vectorizer werden in data/model.pkl / data/vectorizer.pkl gespeichert
# - Es wird eine Summary mit Größen je Quelle zurückgegeben

import os
import csv
import json
from typing import List, Tuple, Dict, Any

import joblib

from chatbot import (
    base_train_data,
    train_model,
    MODEL_FILE,
    VECTORIZER_FILE,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_JSON = os.path.join(DATA_DIR, "training_data.json")
LOG_FILE = os.path.join(DATA_DIR, "chatlog.csv")
UNKNOWN_FILE = os.path.join(DATA_DIR, "unknowns.csv")


def load_from_training_json() -> List[Tuple[str, str]]:
    if not os.path.exists(TRAINING_JSON):
        print("[train_from_logs] Keine training_data.json gefunden – nutze nur Basisdaten + Logs.")
        return []

    try:
        with open(TRAINING_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[train_from_logs] Fehler beim Lesen von training_data.json: {e}")
        return []

    items = []
    for item in data.get("data", []):
        text = (item.get("text") or "").strip()
        intent = (item.get("intent") or "").strip()
        if text and intent:
            items.append((text, intent))

    print(f"[train_from_logs] {len(items)} Beispiele aus training_data.json geladen.")
    return items


def load_from_csv(path: str, source_name: str) -> List[Tuple[str, str]]:
    if not os.path.exists(path):
        print(f"[train_from_logs] {source_name}: Datei {path} nicht gefunden.")
        return []

    items: List[Tuple[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                user_text = (row[1] or "").strip()
                intent = (row[3] or "").strip()
                if not user_text or not intent:
                    continue
                if intent == "unknown":
                    continue
                items.append((user_text, intent))
    except Exception as e:
        print(f"[train_from_logs] Fehler beim Lesen von {source_name}: {e}")
        return []

    print(f"[train_from_logs] {source_name}: {len(items)} gültige Beispiele geladen.")
    return items


def collect_all_training_data() -> tuple[List[Tuple[str, str]], Dict[str, Any]]:
    all_data: List[Tuple[str, str]] = []

    # 1) Basisdaten aus chatbot.py
    base_count = len(base_train_data)
    print(f"[train_from_logs] Basisdaten aus chatbot.base_train_data: {base_count}")
    all_data.extend(base_train_data)

    # 2) training_data.json
    json_data = load_from_training_json()
    json_count = len(json_data)
    all_data.extend(json_data)

    # 3) chatlog.csv
    log_data = load_from_csv(LOG_FILE, "chatlog.csv")
    log_count = len(log_data)
    all_data.extend(log_data)

    # 4) unknowns.csv
    unknown_data = load_from_csv(UNKNOWN_FILE, "unknowns.csv")
    unknown_count = len(unknown_data)
    all_data.extend(unknown_data)

    total_before = len(all_data)

    # Dubletten entfernen (Text+Intent)
    unique = {}
    for text, intent in all_data:
        key = (text, intent)
        unique[key] = (text, intent)

    deduped = list(unique.values())
    total_after = len(deduped)

    summary: Dict[str, Any] = {
        "base": base_count,
        "json": json_count,
        "logs": log_count,
        "unknowns": unknown_count,
        "total_before_dedup": total_before,
        "total_after_dedup": total_after,
    }

    print(f"[train_from_logs] Gesamt vor Dubletten: {total_before}, nach Dubletten: {total_after}")

    return deduped, summary


def train_model_from_all_data():
    train_data, summary = collect_all_training_data()
    if not train_data:
        raise RuntimeError("Keine Trainingsdaten gefunden – Training abgebrochen.")

    print(f"[train_from_logs] Starte Training mit {len(train_data)} Beispielen...")
    vectorizer, model = train_model(train_data)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print(f"[train_from_logs] Training abgeschlossen.")
    print(f"[train_from_logs] Modell gespeichert in: {MODEL_FILE}")
    print(f"[train_from_logs] Vectorizer gespeichert in: {VECTORIZER_FILE}")

    return summary


if __name__ == "__main__":
    summary = train_model_from_all_data()
    print("[train_from_logs] Summary:", summary)