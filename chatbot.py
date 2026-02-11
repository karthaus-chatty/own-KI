# chatbot.py
#
# Zentrales Modul für:
# - Intent-Erkennung (TF-IDF + LogisticRegression)
# - Hybrid-Logik (ML + Keywords + Knowledge-Base)
# - Antwortgenerierung pro Intent
# - Logging von Gesprächen

import os
import csv
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# Pfade & globale Konfiguration
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_FILE = os.path.join(DATA_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(DATA_DIR, "vectorizer.pkl")
TRAINING_JSON = os.path.join(DATA_DIR, "training_data.json")
LOG_FILE = os.path.join(DATA_DIR, "chatlog.csv")
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge_base.json")

UNKNOWN_INTENT = "unknown"
UNKNOWN_CONFIDENCE_THRESHOLD = 0.45  # darunter → unknown

# Wird später nach Laden des Modells gefüllt
known_intents: List[str] = []

# ==============================
# Basis-Trainingsdaten (Seeds)
# ==============================

# Diese Daten sind fest im Code eingebaut und geben dem Modell Grundverständnis.
# Sie werden später durch training_data.json und Logs ergänzt.
base_train_data: List[Tuple[str, str]] = [
    # greeting
    ("Hallo", "greeting"),
    ("Hi", "greeting"),
    ("Hey", "greeting"),
    ("Guten Morgen", "greeting"),
    ("Guten Abend", "greeting"),
    ("Servus", "greeting"),

    # goodbye
    ("Tschüss", "goodbye"),
    ("Ciao", "goodbye"),
    ("Bis später", "goodbye"),
    ("Bis morgen", "goodbye"),
    ("Ich gehe jetzt", "goodbye"),

    # thanks
    ("Danke", "thanks"),
    ("Vielen Dank", "thanks"),
    ("Mega hilfreich, danke dir", "thanks"),
    ("Dankeschön", "thanks"),

    # how_are_you
    ("Wie geht es dir?", "how_are_you"),
    ("Alles gut bei dir?", "how_are_you"),

    # feelings / stress / work
    ("Ich bin gestresst", "stress"),
    ("Ich habe zu viel zu tun", "stress"),
    ("Ich fühle mich überfordert", "stress"),
    ("Ich weiß nicht weiter", "feelings"),
    ("Mir geht es nicht gut", "feelings"),
    ("Ich bin unsicher wegen meiner Arbeit", "work_situation"),
    ("Ich bin unzufrieden im Job", "work_situation"),

    # motivation / decision_help
    ("Motiviere mich", "motivation"),
    ("Ich brauche Motivation", "motivation"),
    ("Ich kann mich nicht entscheiden", "decision_help"),
    ("Hilf mir bei einer Entscheidung", "decision_help"),

    # relationship
    ("Ich mag eine Kollegin", "relationship"),
    ("Ich bin verliebt in jemanden", "relationship"),
    ("Beziehungstipps", "relationship"),

    # tech / coding
    ("Hilf mir mit Python", "python_help"),
    ("Wie schreibe ich eine Funktion in Python?", "python_help"),
    ("Hilf mir bei Webentwicklung", "web_dev"),
    ("Wie baue ich eine Website?", "web_dev"),
    ("Was ist Flask?", "web_dev"),

    ("Erkläre mir künstliche Intelligenz", "ai_explanation"),
    ("Was ist Machine Learning?", "ai_explanation"),
    ("Wie funktioniert ein neuronales Netz?", "ai_explanation"),

    ("Ich habe einen Fehler in meinem Code", "error_debug"),
    ("Fehlermeldung in Python", "error_debug"),
    ("Git push funktioniert nicht", "git_help"),
    ("Hilf mir mit Git", "git_help"),

    ("Wie deploye ich eine App?", "deployment_help"),
    ("Deployment auf Railway", "deployment_help"),

    # explanation / definition / comparison
    ("Erkläre es wie für ein Kind", "explain_like_5"),
    ("Einfach erklärt bitte", "explain_like_5"),
    ("Definition von KI", "definition"),
    ("Was bedeutet Machine Learning?", "definition"),
    ("Was ist der Unterschied zwischen Flask und Django?", "comparison"),
    ("Vergleiche Python und Java", "comparison"),
    ("Vor- und Nachteile von Docker", "pros_cons"),
    ("Was sind die Nachteile von KI?", "pros_cons"),
    ("Erkläre mir das Schritt für Schritt", "step_by_step"),

    # meta
    ("Was kannst du alles?", "what_can_you_do"),
    ("Was sind deine Fähigkeiten?", "what_can_you_do"),
    ("Wie bist du trainiert?", "training_info"),
    ("Wie funktionierst du?", "training_info"),

    # ki_programming (dein Projekt)
    ("Wie kann ich meine eigene KI programmieren?", "ki_programming"),
    ("Hilf mir, eine eigene KI zu bauen", "ki_programming"),
    ("Wie erstelle ich einen eigenen Chatbot?", "ki_programming"),
]


# ==============================
# Knowledge Base laden
# ==============================

def load_knowledge_base() -> Dict[str, str]:
    if not os.path.exists(KNOWLEDGE_FILE):
        return {}
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[chatbot] Konnte knowledge_base.json nicht lesen: {e}")
    return {}


knowledge_base = load_knowledge_base()


# ==============================
# Training & Modell laden
# ==============================

def train_model(train_data: List[Tuple[str, str]]) -> Tuple[TfidfVectorizer, LogisticRegression]:
    texts = [t for t, _ in train_data]
    intents = [i for _, i in train_data]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
    )
    X = vectorizer.fit_transform(texts)

    # Kompatible Variante für ältere scikit-learn Versionen
    model = LogisticRegression(
        max_iter=1000
        # keine multi_class / n_jobs explizit setzen → Default verwenden
    )
    model.fit(X, intents)
    return vectorizer, model


def load_additional_training_data() -> List[Tuple[str, str]]:
    """Lädt zusätzliche Trainingsdaten aus data/training_data.json, falls vorhanden."""
    if not os.path.exists(TRAINING_JSON):
        return []
    try:
        with open(TRAINING_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = []
        for item in data.get("data", []):
            text = (item.get("text") or "").strip()
            intent = (item.get("intent") or "").strip()
            if text and intent:
                items.append((text, intent))
        print(f"[chatbot] Zusätzliche Trainingsdaten aus training_data.json: {len(items)} Beispiele.")
        return items
    except Exception as e:
        print(f"[chatbot] Konnte training_data.json nicht lesen: {e}")
        return []


def initial_train_if_needed():
    """
    Wenn kein Modell vorhanden ist, trainiere eines aus base_train_data + training_data.json.
    Wird beim Import von chatbot.py einmal aufgerufen.
    """
    global known_intents

    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            known_intents = sorted(list(model.classes_))
            print(f"[chatbot] Modell geladen. Intents: {known_intents}")
            return
        except Exception as e:
            print(f"[chatbot] Modell konnte nicht geladen werden: {e}")
            # fällt unten auf Neu-Training zurück

    print("[chatbot] Kein gültiges Modell gefunden – initiales Training...")
    train_data = list(base_train_data)
    train_data.extend(load_additional_training_data())

    vectorizer, model = train_model(train_data)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    known_intents = sorted(list(model.classes_))
    print(f"[chatbot] Initiales Modell trainiert. Intents: {known_intents}")


# beim Import direkt sicherstellen, dass ein Modell da ist
initial_train_if_needed()


def load_model():
    vectorizer = joblib.load(VECTORIZER_FILE)
    model = joblib.load(MODEL_FILE)
    return vectorizer, model


def get_intent_example_counts() -> Dict[str, int]:
    """
    Zählt, wie viele Trainingsbeispiele es pro Intent gibt.
    Basis:
    - base_train_data (im Code)
    - data/training_data.json (falls vorhanden)

    (Logs und unknowns.csv werden hier bewusst nicht berücksichtigt,
    es ist eher ein Gefühl für die "explizit" gelabelten Beispiele.)
    """
    counts: Dict[str, int] = {}

    # Basisdaten aus dem Code
    for _, intent in base_train_data:
        counts[intent] = counts.get(intent, 0) + 1

    # Zusätzliche Daten aus training_data.json
    if os.path.exists(TRAINING_JSON):
        try:
            with open(TRAINING_JSON, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            for item in payload.get("data", []):
                intent = (item.get("intent") or "").strip()
                if intent:
                    counts[intent] = counts.get(intent, 0) + 1
        except Exception as e:
            print(f"[chatbot] Konnte training_data.json für Counts nicht lesen: {e}")

    return counts


# ==============================
# Intent-Vorhersage
# ==============================

def predict_intent(user_text: str) -> Tuple[str, float]:
    """
    Gibt (intent, confidence) zurück. Wenn Confidence unter Threshold, intent=unknown.
    """
    vectorizer, model = load_model()
    X = vectorizer.transform([user_text])
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    max_idx = probs.argmax()
    predicted_intent = classes[max_idx]
    confidence = float(probs[max_idx])

    # Unknown-Handling
    if confidence < UNKNOWN_CONFIDENCE_THRESHOLD:
        return UNKNOWN_INTENT, confidence

    return predicted_intent, confidence

def debug_intent_analysis(user_text: str, top_n: int = 5) -> Dict[str, Any]:
    """
    Liefert Debug-Infos zur Intent-Vorhersage:
    - predicted_intent (mit Unknown-Handling)
    - predicted_confidence (roh, ohne Unknown-Override)
    - top: Liste der Top-N Intents mit Confidence
    - predicted_train_count: wie viele Trainingsbeispiele es für den vorhergesagten Intent gibt
    - top[i].train_count: Beispiele pro Intent in der Top-Liste
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return {
            "predicted_intent": UNKNOWN_INTENT,
            "predicted_confidence": 0.0,
            "predicted_train_count": 0,
            "top": [],
        }

    vectorizer, model = load_model()
    X = vectorizer.transform([user_text])
    probs = model.predict_proba(X)[0]
    intents = model.classes_

    pairs = sorted(
        zip(intents, probs),
        key=lambda x: x[1],
        reverse=True,
    )

    top_list = [
        {"intent": intent, "confidence": float(conf)}
        for intent, conf in pairs[:top_n]
    ]

    if not top_list:
        return {
            "predicted_intent": UNKNOWN_INTENT,
            "predicted_confidence": 0.0,
            "predicted_train_count": 0,
            "top": [],
        }

    best_raw_intent = top_list[0]["intent"]
    best_raw_conf = top_list[0]["confidence"]

    predicted_intent = best_raw_intent
    if best_raw_conf < UNKNOWN_CONFIDENCE_THRESHOLD:
        predicted_intent = UNKNOWN_INTENT

    # Trainings-Beispiel-Counts ergänzen
    counts = get_intent_example_counts()
    for item in top_list:
        item_intent = item["intent"]
        item["train_count"] = int(counts.get(item_intent, 0))

    predicted_train_count = int(counts.get(predicted_intent, 0))

    return {
        "predicted_intent": predicted_intent,
        "predicted_confidence": best_raw_conf,
        "predicted_train_count": predicted_train_count,
        "top": top_list,
    }


# ==============================
# Logging
# ==============================

def log_message(user_text: str, bot_answer: str, intent: str, note: str = ""):
    """
    Schreibt eine Zeile in data/chatlog.csv:
    [timestamp, user_text, bot_answer, intent, note]
    """
    ts = datetime.utcnow().isoformat()
    row = [ts, user_text, bot_answer, intent, note]

    # sicherstellen, dass Datenordner existiert
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print(f"[chatbot] Konnte Log nicht schreiben: {e}")


# ==============================
# Kontext-Nachbearbeitung
# ==============================

def tweak_answer_with_context(answer: str, intent: str, history: List[Dict[str, Any]]) -> str:
    """
    Optionale Kontextanpassungen auf Basis der letzten Messages.
    history: Liste von Dicts wie {"role": "user"/"bot", "text": "...", "intent": "...", "confidence": ...}
    """
    # Beispiel: Wenn der User gerade sehr gestresst ist, ton etwas softer machen
    last_user_msgs = [h for h in history if h.get("role") == "user"]
    if intent in ("stress", "feelings", "work_situation", "relationship") and last_user_msgs:
        answer = (
            "Ich höre, dass dich das wirklich beschäftigt. "
            + answer
        )

    # Bei wiederholten Unknowns: aktiv nachfragen
    last_bot_unknowns = [
        h for h in history[-6:]
        if h.get("role") == "bot" and h.get("intent") == UNKNOWN_INTENT
    ]
    if intent == UNKNOWN_INTENT and len(last_bot_unknowns) >= 1:
        answer += (
            " Wenn du möchtest, kannst du mir kurz in einem Satz sagen, "
            "worum es dir vor allem geht (z.B. Arbeit, Gefühle, Technik)."
        )

    return answer


# ==============================
# Antwortbausteine / Knowledge-Base
# ==============================

def answer_greeting() -> str:
    return "Hey 👋 Was liegt an? Frag mich einfach, womit ich dir helfen soll – Technik, Arbeit, Gefühle, irgendwas."


def answer_goodbye() -> str:
    return "Alles klar, wir hören uns! 👋 Wenn du wieder was brauchst – einfach schreiben."


def answer_thanks() -> str:
    return "Gerne doch! 😊 Wenn du magst, kannst du direkt mit der nächsten Frage weitermachen."


def answer_how_are_you() -> str:
    return "Ich funktioniere tadellos 😄 Wichtiger: Was geht bei dir gerade ab?"


def answer_stress() -> str:
    return (
        "Okay, Stresslevel scheint hoch zu sein 😅 Lass uns das sortieren:\n"
        "1. Worum geht es hauptsächlich (Arbeit, privat, etwas anderes)?\n"
        "2. Was wäre heute ein kleiner, realistischer Schritt, den du schaffen könntest?\n"
        "Erzähl mir kurz Punkt 1, dann überlegen wir zusammen die nächsten Schritte."
    )


def answer_feelings() -> str:
    return (
        "Danke, dass du das teilst. ❤️ Ich bin keine Therapeutin, aber ich kann dir helfen, "
        "deine Gedanken zu sortieren, Optionen zu sehen oder Formulierungen zu finden. "
        "Magst du mir in 1–2 Sätzen sagen, was dich gerade am meisten belastet?"
    )


def answer_relationship() -> str:
    return (
        "Herzthemen sind immer tricky 🫠 Ich kann dir z.B. helfen bei:\n"
        "- Formulierungen für Nachrichten\n"
        "- Sortieren, was du willst\n"
        "- Wie du etwas respektvoll ansprichst\n\n"
        "Erzähl mir kurz: Was ist die Situation, sehr grob zusammengefasst?"
    )


def answer_work_situation() -> str:
    return (
        "Arbeit kann echt Energie fressen. Lass uns kurz strukturieren:\n"
        "1. Was nervt dich am meisten?\n"
        "2. Was läuft noch okay oder sogar gut?\n"
        "3. Was könntest du konkret innerhalb der nächsten Woche verändern?\n"
        "Schreib mir erst mal Punkt 1, dann bauen wir darauf auf."
    )


def answer_motivation() -> str:
    return (
        "Okay, wir werfen den Motivationsmotor an 🔥\n"
        "Sag mir eine Sache, die du heute schaffen willst. "
        "Nur eine. Dann brechen wir die in Mini-Schritte runter."
    )


def answer_decision_help() -> str:
    return (
        "Entscheidungen sind nervig, weil man immer irgendwas verliert. "
        "Wir können es so angehen:\n"
        "1. Welche 2–3 Optionen hast du?\n"
        "2. Was spricht jeweils dafür und dagegen?\n"
        "3. Was wäre die Worst-Case-Folge jeder Option?\n\n"
        "Schreib mir kurz deine Optionen, dann füllen wir die Liste gemeinsam."
    )


def answer_python_help(user_text: str) -> str:
    return (
        "Klar, Python geht immer 🐍\n"
        "Schick mir am besten:\n"
        "- den relevanten Codeausschnitt und\n"
        "- die Fehlermeldung (falls vorhanden).\n\n"
        "Dann erkläre ich dir Schritt für Schritt, was da schief läuft oder wie du es bauen kannst."
    )


def answer_web_dev() -> str:
    kb = knowledge_base.get("flask", "")
    extra = f"\n\nKleiner Kontext zu Flask:\n{kb}" if kb else ""
    return (
        "Webentwicklung kann vieles bedeuten – Frontend, Backend oder beides.\n"
        "Sag mir kurz, ob es um HTML/CSS/JS, Flask/Backend oder Deployment geht – "
        "dann kann ich gezielter helfen."
        + extra
    )


def answer_ai_explanation() -> str:
    kb = knowledge_base.get("künstliche intelligenz", "") or knowledge_base.get("ki", "")
    if kb:
        return kb
    return (
        "Künstliche Intelligenz (KI) bedeutet, dass Computer Aufgaben lösen, "
        "für die man normalerweise menschliche Intelligenz bräuchte – z.B. Sprache verstehen, "
        "Bilder erkennen oder Entscheidungen treffen. Oft lernt ein Modell aus Beispielen "
        "und generalisiert dann auf neue Situationen."
    )


def answer_definition(user_text: str) -> str:
    lower = user_text.lower()
    for key, value in knowledge_base.items():
        if key.lower() in lower:
            return f"Defintion von **{key}**:\n{value}"
    return (
        "Sag mir bitte konkret, wovon du eine Definition möchtest "
        "(z.B. 'Definition von Flask', 'Definition von Machine Learning')."
    )


def answer_explain_like_5(user_text: str) -> str:
    return (
        "Okay, wir gehen auf 'Erklär es wie für ein Kind'-Level 😄\n"
        "Schreib mir bitte nochmal genau, welchen Begriff oder welches Thema ich erklären soll."
    )


def answer_comparison(user_text: str) -> str:
    return (
        "Vergleiche kann ich gut machen. Schreib mir bitte nochmal im Format:\n"
        "'Vergleiche X und Y' – z.B. 'Vergleiche Flask und Django' – "
        "dann stelle ich dir die wichtigsten Unterschiede gegenüber."
    )


def answer_pros_cons(user_text: str) -> str:
    return (
        "Vor- und Nachteile lassen sich gut in eine Liste packen. "
        "Sag mir, wovon du die Pros & Cons möchtest (z.B. 'Remote-Arbeit', 'KI im Unternehmen', 'Docker'), "
        "dann bastle ich dir eine Übersicht."
    )


def answer_step_by_step(user_text: str) -> str:
    return (
        "Gerne Schritt für Schritt ✅\n"
        "Schreib mir bitte, welche Aufgabe oder welches Ziel du genau hast, "
        "dann packe ich dir das in nummerierte Schritte."
    )


def answer_what_can_you_do() -> str:
    return (
        "Ich kann dir helfen bei:\n"
        "- Fragen zu Technik (Python, Web, KI, Deployment)\n"
        "- Strukturieren von Gedanken (Arbeit, Gefühle, Entscheidungen)\n"
        "- Erklärungen (Definitionen, Vergleiche, Schritt-für-Schritt-Anleitungen)\n"
        "- deinem eigenen KI-/Chatbot-Projekt (so wie dieser Bot 😉)\n\n"
        "Frag mich einfach konkret nach deinem nächsten Thema."
    )


def answer_training_info() -> str:
    return (
        "Ich bin ein klassifikationsbasierter Chatbot:\n"
        "- Ich nutze TF-IDF + Logistic Regression für Intenterkennung.\n"
        "- Meine Trainingsdaten kommen aus festen Beispielen, einer training_data.json "
        "und – wenn du das nutzt – aus geloggten Gesprächen.\n"
        "- Alles läuft lokal bzw. in deiner Railway-Umgebung ohne externe KI-API."
    )


def answer_ki_programming() -> str:
    kb = knowledge_base.get("eigene ki programmieren", "")
    base = (
        "Eigene KI programmieren läuft grob so ab:\n"
        "1. Problem definieren (z.B. Text klassifizieren, Chatbot, Empfehlungssystem).\n"
        "2. Daten sammeln und beschriften.\n"
        "3. Ein Modell wählen (z.B. Logistic Regression, Random Forest, Neurales Netz).\n"
        "4. Features bauen (z.B. TF-IDF bei Texten).\n"
        "5. Modell trainieren, validieren, verbessern.\n"
        "6. Modell in eine App einbauen (z.B. Flask-API).\n\n"
        "Du hast mit diesem Projekt schon wirklich viel davon umgesetzt 👌\n"
        "Wenn du magst, können wir Schritt für Schritt das nächste Level planen "
        "(z.B. besseres Modell, mehr Intents, Daten-Cleanup, CI/CD)."
    )
    if kb:
        return kb + "\n\n" + base
    return base


def answer_unknown(user_text: str, confidence: float) -> str:
    return (
        "Ich bin mir da gerade nicht sicher 🤔\n"
        "Formulier deine Frage bitte etwas konkreter oder sag mir kurz, "
        "ob es um Technik, Arbeit oder etwas Persönliches geht."
    )


def get_training_examples_for_similarity() -> List[Dict[str, str]]:
    """
    Liefert eine Liste von Trainingsbeispielen für Similarity-Suche.
    Quellen:
    - base_train_data (Code-basis)
    - data/training_data.json (manuell gelabelt im Admin)
    - data/chatlog.csv (geloggte, bekannte Intents – unbekannte/unknown werden ignoriert)

    Struktur je Eintrag:
    { "text": "...", "intent": "..." }
    """
    examples: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_example(text: str, intent: str):
        t = (text or "").strip()
        i = (intent or "").strip()
        if not t or not i:
            return
        key = (t, i)
        if key in seen:
            return
        seen.add(key)
        examples.append({"text": t, "intent": i})

    # 1) Basisdaten aus dem Code
    for text, intent in base_train_data:
        add_example(text, intent)

    # 2) Zusätzliche Daten aus training_data.json
    if os.path.exists(TRAINING_JSON):
        try:
            with open(TRAINING_JSON, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            for item in payload.get("data", []):
                add_example(item.get("text") or "", item.get("intent") or "")
        except Exception as e:
            print(f"[similarity] Konnte training_data.json nicht lesen: {e}")

    # 3) Geloggte Daten aus chatlog.csv (nur bekannte Intents)
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = (row.get("user_text") or "").strip()
                    i = (row.get("intent") or "").strip()
                    if not t or not i:
                        continue
                    if i == UNKNOWN_INTENT:
                        continue
                    add_example(t, i)
        except Exception as e:
            print(f"[similarity] Konnte chatlog.csv nicht lesen: {e}")

    return examples


# ==============================
# Hauptfunktion: generate_response
# ==============================

def generate_response(user_text: str) -> Tuple[str, str, float]:
    """
    Nimmt den User-Text, bestimmt Intent + Confidence und
    erzeugt eine passende Antwort.

    Rückgabe: (answer, intent, confidence)
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return "Sag ruhig was, dann kann ich antworten 😊", UNKNOWN_INTENT, 0.0

    intent, confidence = predict_intent(user_text)

    # Router: Intent → Antwortlogik
    if intent == "greeting":
        answer = answer_greeting()
    elif intent == "goodbye":
        answer = answer_goodbye()
    elif intent == "thanks":
        answer = answer_thanks()
    elif intent == "how_are_you":
        answer = answer_how_are_you()
    elif intent in ("stress",):
        answer = answer_stress()
    elif intent in ("feelings",):
        answer = answer_feelings()
    elif intent in ("relationship",):
        answer = answer_relationship()
    elif intent in ("work_situation",):
        answer = answer_work_situation()
    elif intent in ("motivation",):
        answer = answer_motivation()
    elif intent in ("decision_help",):
        answer = answer_decision_help()
    elif intent in ("python_help",):
        answer = answer_python_help(user_text)
    elif intent in ("web_dev",):
        answer = answer_web_dev()
    elif intent in ("ai_explanation",):
        answer = answer_ai_explanation()
    elif intent in ("definition",):
        answer = answer_definition(user_text)
    elif intent in ("explain_like_5",):
        answer = answer_explain_like_5(user_text)
    elif intent in ("comparison",):
        answer = answer_comparison(user_text)
    elif intent in ("pros_cons",):
        answer = answer_pros_cons(user_text)
    elif intent in ("step_by_step",):
        answer = answer_step_by_step(user_text)
    elif intent in ("what_can_you_do",):
        answer = answer_what_can_you_do()
    elif intent in ("training_info",):
        answer = answer_training_info()
    elif intent in ("ki_programming",):
        answer = answer_ki_programming()
    else:
        # Fallback: unknown oder nicht explizit gemappter Intent
        answer = answer_unknown(user_text, confidence)
        intent = UNKNOWN_INTENT

    return answer, intent, confidence