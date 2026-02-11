# 1. Daten vorbereiten
texte = [
    "Das ist super, ich liebe es",
    "Das war richtig schlecht",
    "Total genial, hat Spaß gemacht",
    "Ich hasse das",
    "Das gefällt mir gar nicht",
    "Wirklich toll und beeindruckend"
]

labels = [
    "positiv",
    "negativ",
    "positiv",
    "negativ",
    "negativ",
    "positiv"
]

# 2. Texte in Zahlen umwandeln (Features)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texte)  # Matrix aus Wort-Häufigkeiten

# 3. Modell wählen und trainieren
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, labels)

# 4. Neue Texte testen
neue_texte = [
    "Ich finde das großartig",
    "Das war eine Katastrophe",
    "Ganz okay, aber nicht perfekt"
]

X_neu = vectorizer.transform(neue_texte)
vorhersagen = model.predict(X_neu)

for text, label in zip(neue_texte, vorhersagen):
    print(f"Text: {text} -> Einschätzung der KI: {label}")