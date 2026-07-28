import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BASE = _os.path.join(_HERE, "..")

MODEL_PATH = _os.path.join(_BASE, "model", "sentiment_model.pkl")
VECTORIZER_PATH = _os.path.join(_BASE, "model", "sentiment_vectorizer.pkl")
LABELED_DATA_PATH = _os.path.join(_BASE, "data", "labeled_data.csv")

# Classifies a list of headlines using trained model and returns predictions + sentiment score
def classify_headlines(headlines, model, vectorizer):
    if not headlines:
        return pd.DataFrame()
    
    X_vec = vectorizer.transform(headlines)
    predictions = model.predict(X_vec)
    prediction_probs = model.predict_proba(X_vec)

    results = []
    for i, headline in enumerate(headlines):
        sentiment_score = TextBlob(headline).sentiment.polarity
        confidence = round(max(prediction_probs[i]), 3)
        label = '📈 Positive' if predictions[i] == 1 else '📉 Negative'

        results.append({
            'headline': headline,
            'sentiment_score': round(sentiment_score, 3),
            'prediction': label,
            'confidence_score': confidence
        })

    return pd.DataFrame(results)

# Trains a new RandomForest model using labeled headline data
def train_model():
    df = pd.read_csv(LABELED_DATA_PATH, encoding='utf-8')
    X = df['headline']
    y = df['price_movement']

    vectorizer = TfidfVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_vec, y)

    return model, vectorizer

# Loads the model and vectorizer from disk or trains new ones if not found
def train_or_load_model():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        model, vectorizer = train_model()
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
    return model, vectorizer