# app.py – Voll-App mit Auth, Admin, Logging & Session-Kontext

import os
import base64
import uuid

APP_USER = os.getenv("APP_USER") or "user"
APP_PASSWORD = os.getenv("APP_PASSWORD") or "changeme"


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
)

from chatbot import (
    generate_response,
    known_intents,
    log_message,
    tweak_answer_with_context,
)
from analyze_logs import (
    load_logs,
    compute_basic_stats,
    compute_intent_stats,
    compute_day_stats,
    examples_per_intent,
    top_words_overall,
    top_words_per_intent,
)
from export_unknowns import export_unknowns
from train_from_logs import train_model_from_all_data

# ===============================
# Admin-Auth
# ===============================

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or "test123"
print(f"[DEBUG] Voll-App gestartet. ADMIN_PASSWORD = {ADMIN_PASSWORD!r}")


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
    print(f"[DEBUG] Passwort korrekt? {ok}")
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
# Frontend-Login (für Chat & Admin)
# ===============================

def is_logged_in() -> bool:
    return bool(session.get("logged_in"))

def login_user():
    session["logged_in"] = True

def logout_user():
    session.pop("logged_in", None)
    session.pop("history", None)  # optional: History löschen beim Logout

def requires_login(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            return render_template("login.html", error=None), 401
        return view_func(*args, **kwargs)
    return wrapper


# ===============================
# Flask-App & Session
# ===============================

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or "change-me-dev-secret"


@app.before_request
def ensure_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session["history"] = []
        print(f"[DEBUG] Neue Session: {session['session_id']}")


# ===============================
# Routen
# ===============================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        if username == APP_USER and password == APP_PASSWORD:
            login_user()
            return redirect("/")  # nach erfolgreichem Login zum Chat
        else:
            return render_template("login.html", error="Falscher Benutzername oder Passwort."), 401

    # GET
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    logout_user()
    return redirect("/login")

@app.route("/")
@requires_login
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
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
def api_intents():
    return jsonify({"intents": sorted(known_intents)})


# ---------- Admin-Routen ----------

@app.route("/admin")
@requires_login
@requires_auth
def admin_dashboard():
    print("[DEBUG] /admin aufgerufen – Auth ok")
    rows = load_logs()
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
        )

    basic = compute_basic_stats(rows)
    intents = compute_intent_stats(rows)
    per_day = compute_day_stats(rows)
    examples = examples_per_intent(rows, max_per_intent=5)
    top_overall = top_words_overall(rows, top_n=20)
    top_per_intent = top_words_per_intent(rows, top_n=10, min_count=2)

    return render_template(
        "admin.html",
        no_logs=False,
        basic=basic,
        intents=intents,
        per_day=per_day,
        examples=examples,
        top_overall=top_overall,
        top_per_intent=top_per_intent,
    )


@app.route("/admin/export_unknowns", methods=["POST"])
@requires_auth
def admin_export_unknowns():
    print("[DEBUG] /admin/export_unknowns aufgerufen")
    result = export_unknowns()
    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code


@app.route("/admin/train", methods=["POST"])
@requires_auth
def admin_train():
    print("[DEBUG] /admin/train aufgerufen")
    try:
        train_model_from_all_data()
        return jsonify(
            {
                "ok": True,
                "message": "Training abgeschlossen. Neues Modell wurde gespeichert.",
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


if __name__ == "__main__":
    print("[DEBUG] Starte Voll-Flask-App auf http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)