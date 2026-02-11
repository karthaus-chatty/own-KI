# app.py – Voll-App mit Login, Registrierung, Admin, Logging & Session-Kontext

import os
import base64
import uuid
import json
from functools import wraps
from typing import Optional

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

USERS_FILE = os.path.join(DATA_DIR, "users.json")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or "test123"

# Login für Frontend (Default-User, wird beim ersten Start angelegt)
APP_USER = os.getenv("APP_USER") or "user"
APP_PASSWORD = os.getenv("APP_PASSWORD") or "changeme"

print(f"[DEBUG] Voll-App gestartet.")
print(f"[DEBUG] ADMIN_PASSWORD gesetzt (Länge): {len(ADMIN_PASSWORD)}")
print(f"[DEBUG] APP_USER: {APP_USER!r}")


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


@app.route("/api/chat", methods=["POST"])
@requires_login
def api_chat():
    data = request.get_json() or {}
    user_text = (data.get("message") or "").strip()

    if not user_text:
        return jsonify({"error": "Leere Nachricht"}), 400

    history = session.get("history", [])

    answer, intent, confidence = generate_response(user_text)
    answer = tweak_answer_with_context(answer, intent, history)

    session_id = session.get("session_id", "unknown")
    note = f"conf={confidence:.3f};session={session_id}"

    try:
        log_message(user_text, answer, intent, note=note)
    except Exception as e:
        print(f"[DEBUG] Konnte Log nicht schreiben: {e}")

    # Historie aktualisieren (letzte 20 Einträge)
    history.append(
        {"role": "user", "text": user_text, "intent": intent, "confidence": confidence}
    )
    history.append(
        {"role": "bot", "text": answer, "intent": intent, "confidence": confidence}
    )
    session["history"] = history[-20:]

    return jsonify(
        {
            "answer": answer,
            "intent": intent,
            "confidence": confidence,
        }
    )


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
            username=username,
        )

    basic = compute_basic_stats(rows)
    intents = compute_intent_stats(rows)
    per_day = compute_day_stats(rows)
    examples = examples_per_intent(rows, max_per_intent=5)
    top_overall = top_words_overall(rows, top_n=20)
    top_per_intent = top_words_per_intent(rows, top_n=10, min_count=2)
    recent_unknowns_list = recent_unknowns(rows, limit=20)

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


# ===============================
# Main
# ===============================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"[DEBUG] Starte Voll-Flask-App auf http://0.0.0.0:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)