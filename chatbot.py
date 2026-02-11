# chatbot.py
import os
import csv
import random
from datetime import datetime
from typing import List, Tuple, Sequence, Dict, Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# =========================
# Pfade & Dateien
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_FILE = os.path.join(DATA_DIR, "chatlog.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(DATA_DIR, "vectorizer.pkl")

# =========================
# Basis-Trainingsdaten (Minimalfallback)
# =========================

base_train_data: List[Tuple[str, str]] = [
    ("Hallo", "greeting"),
    ("Guten Morgen", "greeting"),
    ("Tschüss", "goodbye"),
    ("Danke", "thanks"),
    ("Was ist künstliche Intelligenz?", "ki_general"),
    ("Wie trainiert man eine KI?", "ki_training"),
    ("Wie programmiere ich meine eigene KI?", "ki_programming"),
    ("Wie installiere ich ein Paket mit pip?", "python_help"),
    ("Was hältst du von Dubstep?", "music"),
    ("Wie geht's dir?", "smalltalk"),
    ("Ich bin gestresst von der Arbeit", "work"),
    ("Mir geht es heute nicht so gut", "feelings"),
    ("Ich komme mit einer Kollegin richtig gut klar", "personal_relations"),
]

# =========================
# Antworten pro Intent
# =========================

responses = {
    "greeting": [
        "Hey! 😊 Wie kann ich dir heute helfen?",
        "Hi! Was steht an?",
        "Moin! Bereit für ein bisschen KI & Code?",
    ],
    "goodbye": [
        "Mach's gut! 👋",
        "Bis später – ich bin hier, wenn du mich brauchst.",
        "Ciao, hab einen entspannten Tag!",
    ],
    "thanks": [
        "Gerne! 🙌",
        "Immer doch 😄",
        "Freut mich, dass ich helfen konnte!",
    ],
    "meta_bot": [
        "Ich bin dein lokaler Chatbot – spezialisiert auf KI, Python und alles drumherum.",
        "Ich helfe dir beim Bauen und Trainieren deiner eigenen KI, plus ein bisschen Smalltalk.",
    ],
    "ki_general": [
        "Künstliche Intelligenz ist ein Sammelbegriff für Systeme, die Aufgaben lösen, für die man normalerweise menschliche Intelligenz bräuchte.",
        "KI umfasst Dinge wie Bilderkennung, Sprachverarbeitung, Vorhersagen und viele andere Bereiche.",
    ],
    "ki_training": [
        "Um eine KI zu trainieren, brauchst du Daten, ein Modell und eine Trainingsprozedur – typischerweise mit Train/Test-Split.",
        "Wichtig beim Training: genug Daten, gute Features und vernünftige Evaluation (z.B. Accuracy, F1).",
    ],
    "ki_programming": [
        "Ein typischer Start: Python, scikit-learn, ein Datensatz, TF-IDF und ein Klassifikator wie Logistic Regression.",
        "Du kannst z.B. Textdaten mit TfidfVectorizer featurisieren und dann mit LogisticRegression trainieren.",
    ],
    "python_help": [
        "Python-Frage incoming 😄 Erzähl, woran du hängst.",
        "Python ist da recht flexibel – beschreib mir dein Problem, dann gehen wir es Schritt für Schritt durch.",
    ],
    "music": [
        "Dubstep, EDM und Metal? Klingt nach einem soliden Setup zum Coden 🤘",
        "Für Fokus empfehlen viele instrumentale EDM- oder LoFi-Tracks, für Motivation gerne auch Metal.",
    ],
    "smalltalk": [
        "Klar, lass ein bisschen quatschen 😄",
        "Okay, Pause-Modus aktiviert. Was geht dir gerade durch den Kopf?",
    ],
    "work": [
        "Arbeit kann echt ziehen. Magst du eher über Organisation, Grenzen setzen oder Motivation reden?",
        "Klingt, als wäre im Job gut was los. Erzähl mir ein bisschen mehr, dann sortieren wir das gemeinsam.",
    ],
    "feelings": [
        "Danke, dass du das teilst. ❤️ Willst du eher venten, ablenken oder eine kleine Strategie überlegen?",
        "Gefühle sind kein Bug, sondern Feature – wir können gern kurz sortieren, was gerade los ist.",
    ],
    "personal_relations": [
        "Zwischenmenschliche Themen sind immer spannend, aber auch komplex. Wir können das Schritt für Schritt durchgehen – ohne dass ich dir irgendwas aufdränge.",
        "Klingt, als ginge es um eine besondere Person 😉 Ich kann dir helfen, deine Gedanken zu sortieren, keine „Tipps aus der Gießkanne“.",
    ],
    "unknown": [
        "Da bin ich mir noch nicht sicher 🤔 Magst du das anders formulieren?",
        "Das kann ich noch nicht richtig einordnen – vielleicht später ein eigener Intent?",
    ],
}

known_intents = sorted(set(intent for _, intent in base_train_data) | set(responses.keys()))

# =========================
# Logging
# =========================

def log_message(user_text: str, bot_answer: str, intent: str, note: str = ""):
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(LOG_FILE, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, user_text, bot_answer, intent, note])


# =========================
# Modell laden / trainieren
# =========================

def train_model(train_data: List[Tuple[str, str]]):
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


def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        print(f"[chatbot] Lade Modell aus {MODEL_FILE}")
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return vectorizer, model

    print("[chatbot] Kein gespeichertes Modell gefunden – trainiere Basis-Modell.")
    vectorizer, model = train_model(base_train_data)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    return vectorizer, model


vectorizer, model = load_or_train_model()

# =========================
# Prediction + Response
# =========================

def predict_intent_with_confidence(user_text: str):
    X = vectorizer.transform([user_text])
    probs = model.predict_proba(X)[0]
    best_idx = int(np.argmax(probs))
    intent = model.classes_[best_idx]
    confidence = float(probs[best_idx])
    return intent, confidence


def generate_response(user_text: str):
    """
    Gibt (answer, intent, confidence) zurück.
    """
    intent, confidence = predict_intent_with_confidence(user_text)

    if intent in responses:
        answer = random.choice(responses[intent])
    else:
        answer = random.choice(responses["unknown"])

    return answer, intent, confidence


# =========================
# Kontext-Tweaks
# =========================

def tweak_answer_with_context(answer: str, intent: str, history: Sequence[Dict[str, Any]]) -> str:
    """
    Nimmt die vom Modell gewählte Antwort und passt sie leicht an,
    basierend auf der bisherigen Session-Historie.

    history: Liste von Dicts wie
      {"role": "user"/"bot", "text": str, "intent": str, "confidence": float}
    """
    if not history:
        return answer

    # letztes User-Intent aus der Historie finden
    last_user_intent = None
    for entry in reversed(history):
        if entry.get("role") == "user":
            last_user_intent = entry.get("intent")
            break

    # 1) Wiederbegrüßung
    if intent == "greeting" and history:
        variants = [
            "Hey, da bist du wieder 😄 Was liegt an?",
            "Willkommen zurück! Was möchtest du diesmal angehen?",
            "Hi again! Wollen wir da weitermachen, wo wir aufgehört haben?",
        ]
        return random.choice(variants)

    # 2) Smalltalk nach Work/Feelings/Relations – weicher Übergang
    if intent == "smalltalk" and last_user_intent in ("work", "feelings", "personal_relations"):
        return (
            answer
            + " Und falls du über das Thema von eben noch weiterreden willst, bin ich auch dabei."
        )

    # 3) Work nach Feelings – etwas empathischer Einstieg
    if intent == "work" and last_user_intent == "feelings":
        return (
            "Okay, schauen wir mal auf die Arbeit, wenn sie so viel Raum einnimmt. "
            + answer
        )

    # 4) Personal Relations nach Work/Feelings – Hinweis auf Komplexität
    if intent == "personal_relations" and last_user_intent in ("work", "feelings"):
        return (
            "Zwischenmenschliches hängt oft mit Stimmung und Alltag zusammen – "
            + answer
        )

    return answer


# Optional: CLI-Chat zum Testen
if __name__ == "__main__":
    print("Local Chatbot (CLI). Zum Beenden 'quit' eingeben.")
    session_history: List[Dict[str, Any]] = []
    while True:
        msg = input("Du: ").strip()
        if msg.lower() in ("quit", "exit"):
            break
        ans, intent, conf = generate_response(msg)
        ans = tweak_answer_with_context(ans, intent, session_history)
        print(f"Bot [{intent} ({conf:.2f})]: {ans}")
        session_history.append(
            {"role": "user", "text": msg, "intent": intent, "confidence": conf}
        )
        session_history.append(
            {"role": "bot", "text": ans, "intent": intent, "confidence": conf}
        )
        log_message(msg, ans, intent, note=f"conf={conf:.3f};cli=1")