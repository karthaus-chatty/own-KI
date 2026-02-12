# app.py – Voll-App mit Login, Registrierung, Admin, Logging & Session-Kontext

import csv
import os
import base64
import uuid
import json
from functools import wraps
from typing import Optional
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
import csv

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    session,
    redirect,
    send_file,
)
from export_unknowns import export_unknowns, UNKNOWN_FILE
from werkzeug.security import generate_password_hash, check_password_hash

from chatbot import (
    generate_response,
    known_intents,
    log_message,
    tweak_answer_with_context,
    debug_intent_analysis,
    get_intent_example_counts,
    #find_similar_examples,
)
from analyze_logs import (
    load_logs,
    compute_basic_stats,
    compute_intent_stats,
    compute_day_stats,
    examples_per_intent,
    top_words_overall,
    top_words_per_intent,
    recent_unknowns,
)
from export_unknowns import export_unknowns
from train_from_logs import train_model_from_all_data

# ===============================
# Pfade & Konfiguration
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
TRAINING_JSON = os.path.join(DATA_DIR, "training_data.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.csv")
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge_base.json")

USERS_FILE = os.path.join(DATA_DIR, "users.json")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or "test123"

# Login für Frontend (Default-User, wird beim ersten Start angelegt)
APP_USER = os.getenv("APP_USER") or "user"
APP_PASSWORD = os.getenv("APP_PASSWORD") or "changeme"

print(f"[DEBUG] Voll-App gestartet.")
print(f"[DEBUG] ADMIN_PASSWORD gesetzt (Länge): {len(ADMIN_PASSWORD)}")
print(f"[DEBUG] APP_USER: {APP_USER!r}")

def apply_style(answer: str, mode: str) -> str:
    """
    Passt den Antwort-Stil an den gewählten Modus an.
    Modi:
    - friendly: lockerer Ton, Emojis okay
    - focus: sachlich, kein unnötiger Schnickschnack
    - coach: etwas reflektierender Stil
    """
    answer = answer or ""

    if mode == "focus":
        # Sehr sachlich, eher knapp – hier einfach als Platzhalter:
        # Emojis entfernen und ggf. etwas kürzen, wenn extrem lang.
        for emo in ["😊", "😄", "😅", "😉", "😎", "❤️", "💙", "😃"]:
            answer = answer.replace(emo, "")
        # optional leichte Kürzung bei extrem langem Text
        if len(answer) > 1500:
            answer = answer[:1500].rstrip() + " …"
        return answer

    if mode == "coach":
        # Sanfter Coach-Ton am Ende
        extra = "\n\nWenn du magst, erzähl mir gern noch ein bisschen mehr dazu – " \
                "dann kann ich dir gezielter helfen."
        # Nicht doppelt anhängen
        if extra.strip() not in answer:
            answer = answer.rstrip() + extra
        return answer

    # friendly (default)
    return answer


def load_knowledge_base() -> dict:
    """
    Lädt die Knowledge-Base aus data/knowledge_base.json.
    """
    if not os.path.exists(KNOWLEDGE_FILE):
        return {}
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            kb = json.load(f) or {}
        if not isinstance(kb, dict):
            return {}
        return kb
    except Exception as e:
        print(f"[knowledge] Fehler beim Lesen von knowledge_base.json: {e}")
        return {}


def get_knowledge_snippet(intent: str, max_chars: int = 800) -> str | None:
    """
    Versucht, für einen Intent einen Wissens-Snippet zu laden.
    Nutzt Pfade aus knowledge_base.json und liest die erste existierende Datei.
    """
    intent = (intent or "").strip()
    if not intent:
        return None

    kb = load_knowledge_base()
    files = kb.get(intent)
    if not files:
        return None

    if isinstance(files, str):
        files = [files]

    for rel_path in files:
        if not rel_path:
            continue
        if os.path.isabs(rel_path):
            full_path = rel_path
        else:
            full_path = os.path.join(DATA_DIR, rel_path)

        if not os.path.exists(full_path):
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"[knowledge] Fehler beim Lesen von {full_path}: {e}")
            continue

        if not text:
            continue

        # Kürzen auf max_chars, möglichst an Absatzgrenze
        if len(text) > max_chars:
            cut = text[:max_chars]
            last_nl = cut.rfind("\n\n")
            if last_nl > 100:
                cut = cut[:last_nl]
            text = cut.rstrip() + " …"

        return text

    return None


def apply_knowledge(answer: str, intent: str) -> str:
    """
    Hängt – falls vorhanden – einen Wissens-Snippet für den Intent an.
    """
    answer = answer or ""
    snippet = get_knowledge_snippet(intent)
    if not snippet:
        return answer

    appended = (
        answer.rstrip()
        + "\n\n"
        + "Zusatzinfo zu diesem Thema:\n"
        + snippet
    )
    return appended

def load_feedback_stats() -> dict:
    """
    Liest data/feedback.csv ein und berechnet:
    - total: Gesamtanzahl Feedbacks
    - up: Anzahl Daumen hoch
    - down: Anzahl Daumen runter
    - by_intent: Liste von Dicts mit Intent-Stats:
        {
          "intent": "feelings",
          "up": 10,
          "down": 2,
          "total": 12,
          "up_percent": 83.3
        }
      sortiert nach up_percent aufsteigend (schlechteste zuerst).
    """
    stats = {
        "total": 0,
        "up": 0,
        "down": 0,
        "by_intent": [],
    }

    if not os.path.exists(FEEDBACK_FILE):
        return stats

    per_intent: dict[str, dict[str, int]] = {}

    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                # ts = row[0]
                # text = row[1]
                intent = (row[2] or "").strip() or "unknown"
                rating = (row[3] or "").strip().lower()

                if rating not in ("up", "down"):
                    continue

                stats["total"] += 1
                stats[rating] += 1

                bucket = per_intent.setdefault(intent, {"up": 0, "down": 0})
                bucket[rating] += 1
    except Exception as e:
        print(f"[feedback] Fehler beim Lesen von feedback.csv: {e}")
        return stats

    for intent, d in per_intent.items():
        total = d["up"] + d["down"]
        if total <= 0:
            continue
        up_percent = (d["up"] / total) * 100.0
        stats["by_intent"].append(
            {
                "intent": intent,
                "up": d["up"],
                "down": d["down"],
                "total": total,
                "up_percent": up_percent,
            }
        )

    # Schlechteste zuerst (niedrigste up_percent)
    stats["by_intent"].sort(key=lambda x: x["up_percent"])

    return stats

# ===============================
# User-Verwaltung (Registrierung)
# ===============================

def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("users", [])
    except Exception as e:
        print(f"[DEBUG] Konnte users.json nicht lesen: {e}")
        return []


def save_users(users):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump({"users": users}, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG] {len(users)} User in users.json gespeichert.")
    except Exception as e:
        print(f"[DEBUG] Konnte users.json nicht schreiben: {e}")


def ensure_default_user():
    users = load_users()
    for u in users:
        if u.get("username") == APP_USER:
            print("[DEBUG] Default-User existiert bereits.")
            return
    # Default-User anlegen
    hashed = generate_password_hash(APP_PASSWORD)
    users.append({"username": APP_USER, "password_hash": hashed})
    save_users(users)
    print(f"[DEBUG] Default-User {APP_USER!r} angelegt.")


def find_user(username: str):
    users = load_users()
    for u in users:
        if u.get("username", "").lower() == username.lower():
            return u
    return None


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if not username or not password:
        return False, "Benutzername und Passwort dürfen nicht leer sein."

    if len(username) < 3:
        return False, "Benutzername muss mindestens 3 Zeichen lang sein."
    if len(password) < 4:
        return False, "Passwort muss mindestens 4 Zeichen lang sein."

    users = load_users()
    for u in users:
        if u.get("username", "").lower() == username.lower():
            return False, "Benutzername ist bereits vergeben."

    hashed = generate_password_hash(password)
    users.append({"username": username, "password_hash": hashed})
    save_users(users)
    return True, "Benutzer erfolgreich angelegt."


# ===============================
# Admin-Auth (Basic Auth für /admin)
# ===============================

def check_auth(auth_header: Optional[str]) -> bool:
    if not auth_header:
        print("[DEBUG] Kein Authorization-Header vorhanden")
        return False
    if not auth_header.startswith("Basic "):
        print("[DEBUG] Authorization-Header nicht 'Basic ...'")
        return False

    try:
        b64 = auth_header.split(" ", 1)[1]
        decoded = base64.b64decode(b64).decode("utf-8")
        username, password = decoded.split(":", 1)
        print(f"[DEBUG] Basic-Auth empfangen: user={username!r}, pass_len={len(password)}")
    except Exception as e:
        print(f"[DEBUG] Fehler beim Decoden des Auth-Headers: {e}")
        return False

    ok = password == ADMIN_PASSWORD
    print(f"[DEBUG] Admin-Passwort korrekt? {ok}")
    return ok


def authenticate():
    print("[DEBUG] Sende 401 für Admin-Login")
    return Response(
        "Admin-Login erforderlich.",
        401,
        {"WWW-Authenticate": 'Basic realm="Chatbot Admin"'},
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not check_auth(auth_header):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


# ===============================
# Frontend-Login (Session-basiert)
# ===============================

def is_logged_in() -> bool:
    return bool(session.get("logged_in"))


def login_user(username: str):
    session["logged_in"] = True
    session["username"] = username


def logout_user():
    session.pop("logged_in", None)
    session.pop("history", None)
    session.pop("username", None)


def requires_login(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            return redirect("/login")
        return view_func(*args, **kwargs)

    return wrapper


# ===============================
# Flask-App & Session
# ===============================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or "change-me-dev-secret"

ensure_default_user()


@app.before_request
def ensure_session():
    # Session-ID & History einmalig anlegen
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session["history"] = []
        print(f"[DEBUG] Neue Session: {session['session_id']}")


# ===============================
# Login / Logout / Registrierung
# ===============================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        print(f"[DEBUG] Login-Versuch: username={username!r}")

        user = find_user(username)
        if user and check_password_hash(user.get("password_hash", ""), password):
            login_user(username)
            print(f"[DEBUG] Login erfolgreich für {username!r}")
            return redirect("/")
        else:
            print(f"[DEBUG] Login FEHLGESCHLAGEN für {username!r}")
            return render_template("login.html", error="Falscher Benutzername oder Passwort."), 401

    # GET
    return render_template("login.html", error=None)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        ok, msg = register_user(username, password)
        if ok:
            print(f"[DEBUG] Neuer User registriert: {username!r}")
            login_user(username)
            return redirect("/")
        else:
            print(f"[DEBUG] Registrierung fehlgeschlagen für {username!r}: {msg}")
            return render_template("register.html", error=msg), 400

    # GET
    return render_template("register.html", error=None)


@app.route("/logout")
def logout():
    print("[DEBUG] Logout aufgerufen")
    logout_user()
    return redirect("/login")


# ===============================
# Routen – Chat & API
# ===============================

@app.route("/admin/download_unknowns", methods=["GET"])
@requires_login
@requires_auth
def admin_download_unknowns():
    if not os.path.exists(UNKNOWN_FILE):
        return (
            "Die Datei unknowns.csv existiert noch nicht. "
            "Bitte zuerst Unknown-Intents exportieren.",
            404,
        )
    return send_file(
        UNKNOWN_FILE,
        as_attachment=True,
        download_name="unknowns.csv",
        mimetype="text/csv",
    )

@app.route("/")
@requires_login
def index():
    username = session.get("username", "")
    return render_template("index.html", username=username)



@app.route("/api/intents", methods=["GET"])
@requires_login
def api_intents():
    return jsonify({"intents": sorted(known_intents)})


# ===============================
# Admin-Routen
# ===============================

@app.route("/admin")
@requires_login
@requires_auth
def admin_dashboard():
    print("[DEBUG] /admin aufgerufen – Auth ok")
    rows = load_logs()
    username = session.get("username", "")

    # Feedback-Auswertung unabhängig von Logs laden
    feedback_stats = load_feedback_stats()

    if not rows:
        return render_template(
            "admin.html",
            no_logs=True,
            basic=None,
            intents=None,
            per_day=None,
            examples=None,
            top_overall=None,
            top_per_intent=None,
            recent_unknowns_list=[],
            intent_health=[],
            feedback_stats=feedback_stats,
            username=username,
        )

    basic = compute_basic_stats(rows)
    intents = compute_intent_stats(rows)
    per_day = compute_day_stats(rows)
    examples = examples_per_intent(rows, max_per_intent=5)
    top_overall = top_words_overall(rows, top_n=20)
    top_per_intent = top_words_per_intent(rows, top_n=10, min_count=2)
    recent_unknowns_list = recent_unknowns(rows, limit=20)

    # === Intent-Health: Training- und Log-Zahlen zusammenführen ===
    train_counts = get_intent_example_counts()
    # Log-Zahlen aus compute_intent_stats.by_intent in ein Dict bringen
    log_counts = {}
    for item in intents["by_intent"]:
        name = item["intent"]
        log_counts[name] = item["count"]

    all_intent_names = sorted(set(list(train_counts.keys()) + list(log_counts.keys())))

    health = []
    for name in all_intent_names:
        if name == "unknown":
            # Unknown ist kein "trainbarer" Intent
            continue

        t_count = int(train_counts.get(name, 0))
        l_count = int(log_counts.get(name, 0))

        # Status-Kategorien (kannst du nach Geschmack anpassen)
        if t_count >= 20:
            status = "good"
            label = "stark trainiert"
        elif t_count >= 5:
            status = "medium"
            label = "okay, aber ausbaubar"
        elif t_count == 0 and l_count > 0:
            status = "needs_training"
            label = "in Logs, aber ohne Trainingsdaten"
        else:
            status = "low"
            label = "wenig Trainingsdaten"

        health.append(
            {
                "intent": name,
                "train_count": t_count,
                "log_count": l_count,
                "status": status,
                "status_label": label,
            }
        )

    return render_template(
        "admin.html",
        no_logs=False,
        basic=basic,
        intents=intents,
        per_day=per_day,
        examples=examples,
        top_overall=top_overall,
        top_per_intent=top_per_intent,
        recent_unknowns_list=recent_unknowns_list,
        intent_health=health,
        feedback_stats=feedback_stats,
        username=username,
    )


@app.route("/admin/export_unknowns", methods=["POST"])
@requires_login
@requires_auth
def admin_export_unknowns():
    print("[DEBUG] /admin/export_unknowns aufgerufen")
    result = export_unknowns()
    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code


@app.route("/admin/train", methods=["POST"])
@requires_login
@requires_auth
def admin_train():
    print("[DEBUG] /admin/train aufgerufen")
    try:
        summary = train_model_from_all_data()
        return jsonify(
            {
                "ok": True,
                "message": "Training abgeschlossen. Neues Modell wurde gespeichert.",
                "summary": summary,
            }
        )
    except Exception as e:
        print(f"[DEBUG] Training fehlgeschlagen: {e}")
        return jsonify(
            {
                "ok": False,
                "message": f"Training fehlgeschlagen: {e}",
            }
        ), 500
    
@app.route("/admin/debug_intent", methods=["POST"])
@requires_login
@requires_auth
def admin_debug_intent():
    print("[DEBUG] /admin/debug_intent aufgerufen")
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"ok": False, "message": "Leerer Text."}), 400

    try:
        analysis = debug_intent_analysis(text, top_n=5)
        return jsonify({"ok": True, "analysis": analysis})
    except Exception as e:
        print(f"[DEBUG] admin_debug_intent Fehler: {e}")
        return jsonify(
            {
                "ok": False,
                "message": f"Fehler bei der Intent-Analyse: {e}",
            }
        ), 500

@app.route("/admin/annotate", methods=["POST"])
@requires_login
@requires_auth
def admin_annotate():
    """
    Nimmt einen Text + Intent aus dem Admin entgegen
    und hängt ihn an data/training_data.json an.
    """
    from flask import current_app  # optional, nur falls du loggen willst

    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    intent = (data.get("intent") or "").strip()

    if not text:
        return jsonify({"ok": False, "message": "Kein Text übergeben."}), 400
    if not intent:
        return jsonify({"ok": False, "message": "Kein Intent angegeben."}), 400

    # Datei einlesen oder Grundstruktur erzeugen
    payload = {"data": []}
    if os.path.exists(TRAINING_JSON):
        try:
            with open(TRAINING_JSON, "r", encoding="utf-8") as f:
                payload = json.load(f) or {"data": []}
        except Exception:
            # falls kaputt, fangen wir lieber sauber neu an
            payload = {"data": []}

    if "data" not in payload or not isinstance(payload["data"], list):
        payload["data"] = []

    payload["data"].append({"text": text, "intent": intent})

    try:
        with open(TRAINING_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[admin_annotate] Schreibfehler: {e}")
        return jsonify({"ok": False, "message": f"Konnte training_data.json nicht schreiben: {e}"}), 500

    return jsonify(
        {
            "ok": True,
            "message": f"Beispiel gespeichert (Intent: {intent}). "
                       "Starte danach ein Re-Training, damit es aktiv wird.",
        }
    )

@app.route("/admin/similar_examples", methods=["POST"])
@requires_login
@requires_auth
def admin_similar_examples():
    """
    Nimmt einen Text entgegen und liefert ähnliche Trainingsbeispiele zurück.
    Wird im Admin-Unknown-Panel genutzt, um beim Labeln zu helfen.
    """
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"ok": False, "message": "Kein Text übergeben."}), 400

    try:
        examples = find_similar_examples(text, top_n=5)
    except Exception as e:
        print(f"[admin_similar_examples] Fehler: {e}")
        return jsonify({"ok": False, "message": "Fehler bei der Ähnlichkeitssuche."}), 500

    return jsonify({"ok": True, "examples": examples})

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """
    Nimmt Feedback vom Frontend entgegen:
    { "text": "...Bot-Antwort...", "intent": "ki_programming", "rating": "up"|"down" }
    und loggt es in data/feedback.csv.
    """
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    intent = (data.get("intent") or "").strip()
    rating = (data.get("rating") or "").strip().lower()

    if rating not in ("up", "down"):
        return jsonify({"ok": False, "message": "Ungültiges Rating."}), 400

    # Minimal-Logging, Intent/Text können leer sein – aber hilfreich, wenn vorhanden
    from datetime import datetime

    ts = datetime.utcnow().isoformat()
    row = [ts, text, intent, rating]

    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print(f"[api_feedback] Konnte Feedback nicht schreiben: {e}")
        return jsonify({"ok": False, "message": "Fehler beim Speichern des Feedbacks."}), 500

    return jsonify({"ok": True})

@app.route("/api/mode", methods=["GET", "POST"])
def api_mode():
    """
    GET: gibt den aktuellen Modus zurück
    POST: setzt den Modus (friendly, focus, coach)
    """
    if request.method == "GET":
        mode = session.get("mode", "friendly")
        return jsonify({"ok": True, "mode": mode})

    data = request.get_json() or {}
    mode = (data.get("mode") or "").strip().lower()

    allowed = {"friendly", "focus", "coach"}
    if mode not in allowed:
        return jsonify({"ok": False, "message": "Ungültiger Modus."}), 400

    session["mode"] = mode
    return jsonify({"ok": True, "mode": mode})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Chat-Endpoint:
    - nimmt User-Text entgegen
    - ruft generate_response auf
    - reichert Antwort mit Knowledge-Snippet an (falls vorhanden)
    - passt Stil an gewählten Modus an
    - liefert answer, intent, confidence
    """
    data = request.get_json() or {}
    user_text = (data.get("message") or data.get("text") or "").strip()

    if not user_text:
        return jsonify({"ok": False, "message": "Leere Nachricht."}), 400

    # Modus aus Session (Default friendly)
    mode = session.get("mode", "friendly")

    # generate_response kann bei dir je nach Version etwas anderes zurückgeben.
    # Wir sind tolerant und unterstützen tuple oder dict oder plain string.
    result = generate_response(user_text)

    answer = ""
    intent = "unknown"
    confidence = 0.0

    # tuple: (answer, intent, confidence) o.Ä.
    if isinstance(result, tuple):
        if len(result) >= 1:
            answer = result[0]
        if len(result) >= 2:
            intent = result[1] or intent
        if len(result) >= 3:
            try:
                confidence = float(result[2])
            except Exception:
                confidence = 0.0

    # dict-Variante
    elif isinstance(result, dict):
        answer = result.get("answer", "") or ""
        intent = result.get("intent") or intent
        try:
            confidence = float(result.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0

    # plain string
    else:
        answer = str(result)

    # Wissens-Snippet anhängen
    answer = apply_knowledge(answer, intent)

    # Stil anwenden
    answer = apply_style(answer, mode)

    # Optional: Logging – falls nicht schon in generate_response enthalten
    try:
        log_message(
            user_text=user_text,
            bot_answer=answer,
            intent=intent,
            confidence=confidence,
        )
    except Exception as e:
        print(f"[api_chat] Konnte Log nicht schreiben: {e}")

    return jsonify(
        {
            "ok": True,
            "answer": answer,
            "intent": intent,
            "confidence": confidence,
            "mode": mode,
        }
    )
def get_training_examples_for_similarity() -> List[Dict[str, str]]:
    """
    Liefert eine Liste von Trainingsbeispielen für Similarity-Suche.
    Quelle:
    - base_train_data (Code-basis)
    - data/training_data.json (manuell gelabelt im Admin)

    Struktur je Eintrag:
    { "text": "...", "intent": "..." }
    """
    examples: List[Dict[str, str]] = []

    # Basisdaten aus dem Code
    for text, intent in base_train_data:
        t = (text or "").strip()
        i = (intent or "").strip()
        if not t or not i:
            continue
        examples.append({"text": t, "intent": i})

    # Zusätzliche Daten aus training_data.json
    if os.path.exists(TRAINING_JSON):
        try:
            with open(TRAINING_JSON, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            for item in payload.get("data", []):
                t = (item.get("text") or "").strip()
                i = (item.get("intent") or "").strip()
                if not t or not i:
                    continue
                examples.append({"text": t, "intent": i})
        except Exception as e:
            print(f"[similarity] Konnte training_data.json nicht lesen: {e}")

    # Du könntest hier optional auch gelabelte Logs ergänzen.
    # Für den Anfang reicht Basis + JSON meist völlig.

    return examples


def find_similar_examples(user_text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Findet die Top-N ähnlichsten Trainingsbeispiele für einen gegebenen Text.
    Nutzt den selben Vektorraum (vectorizer) wie das Modell.
    Rückgabe: Liste von Dicts mit:
        {
          "text": "...",
          "intent": "...",
          "similarity": float  # 0..1
        }
    """
    user_text = (user_text or "").strip()
    if not user_text:
        return []

    examples = get_training_examples_for_similarity()
    if not examples:
        return []

    vectorizer, model = load_model()

    corpus_texts = [e["text"] for e in examples]
    # Query + Corpus in EINEM Rutsch vektorisieren
    X = vectorizer.transform([user_text] + corpus_texts)
    q_vec = X[0:1]
    corpus_vecs = X[1:]

    sims = cosine_similarity(q_vec, corpus_vecs)[0]  # shape: (len(corpus),)

    indexed = list(enumerate(sims))
    indexed.sort(key=lambda x: x[1], reverse=True)

    results: List[Dict[str, Any]] = []
    for idx, score in indexed[:top_n]:
        ex = examples[idx]
        results.append(
            {
                "text": ex["text"],
                "intent": ex["intent"],
                "similarity": float(score),
            }
        )

    return results

# ===============================
# Main
# ===============================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"[DEBUG] Starte Voll-Flask-App auf http://0.0.0.0:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)