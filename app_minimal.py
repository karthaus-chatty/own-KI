# app.py – Minimalversion nur zum Test der Admin-Auth

import base64
from functools import wraps
from typing import Optional

from flask import Flask, request, Response


ADMIN_PASSWORD = "test123"
print(f"[DEBUG] Minimal-App gestartet. ADMIN_PASSWORD = {ADMIN_PASSWORD!r}")

app = Flask(__name__)


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
        print(f"[DEBUG] Basic-Auth empfangen: user={username!r}, pass={password!r}")
    except Exception as e:
        print(f"[DEBUG] Fehler beim Decoden des Auth-Headers: {e}")
        return False

    ok = password == ADMIN_PASSWORD
    print(f"[DEBUG] Passwort korrekt? {ok}")
    return ok


def authenticate():
    print("[DEBUG] Sende 401 für Admin-Login")
    return Response(
        "Admin-Login erforderlich (Minimal-Test).",
        401,
        {"WWW-Authenticate": 'Basic realm="Chatbot Admin Test"'},
    )


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not check_auth(auth_header):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


@app.route("/")
def index():
    return "OK: Index-Seite ohne Auth (Minimal-Test)."


@app.route("/admin")
@requires_auth
def admin():
    return "OK: Du bist im geschützten Admin-Bereich der Minimal-App 🎉"


if __name__ == "__main__":
    print("[DEBUG] Starte Minimal-Flask-App auf http://0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)