# Schlankes Python-Image
FROM python:3.11-slim

# System-Dependencies für scikit-learn / numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# Requirements zuerst kopieren (für besseren Cache)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Restliche Dateien
COPY . .

# Standard-Port für Flask
EXPOSE 5000

# Environment, damit Flask im Container korrekt läuft
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# data-Verzeichnis sicherstellen (zur Sicherheit)
RUN mkdir -p /app/data

EXPOSE 5000
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"] 