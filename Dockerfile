FROM python:3.11-slim

# Basisverzeichnis
WORKDIR /app

# System-Dependencies (optional minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Projektcode
COPY . .

# Default-Port (wird von $PORT überschrieben)
ENV PORT=8000

# Gunicorn als WSGI-Server nutzen (statt app.run)
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-8000} app:app"]