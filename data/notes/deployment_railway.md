Deployment mit Railway – Kurzsummary

- Repository mit Flask-App (app.py) auf GitHub.
- Railway-Projekt erstellen und mit GitHub-Repo verbinden.
- Build-Umgebung: Python-Version per environment variable oder Procfile definieren.
- Start-Command: z.B. gunicorn app:app oder python app.py (je nach Setup).

Tipps:
- .env-Variablen für Secrets nutzen (z.B. SECRET_KEY).
- Logging in Dateien vermeiden oder klein halten (Railway-Storage).
- Debug-Modus in Produktion ausschalten.