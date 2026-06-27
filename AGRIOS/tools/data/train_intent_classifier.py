"""
Train a quick intent classifier (TF-IDF + LogisticRegression)
Saves: agrisense_app/backend/models/intent_vectorizer.joblib
       agrisense_app/backend/models/intent_classifier.joblib
Produces a small report printed to stdout and saved to models/intent_training_report.json
"""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

DATA = Path("agrisense_app/backend/data/chatbot_intents.csv")
MODELS_DIR = Path("agrisense_app/backend/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    X = df['utterance'].astype(str)
    y = df['intent'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
    Xt = vec.fit_transform(X_train)
    Xv = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver='saga')
    clf.fit(Xt, y_train)

    preds = clf.predict(Xv)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    # Save artifacts
    joblib.dump(vec, MODELS_DIR / 'intent_vectorizer.joblib')
    joblib.dump(clf, MODELS_DIR / 'intent_classifier.joblib')

    out = {
        'accuracy': acc,
        'num_classes': int(len(df['intent'].unique())),
        'num_samples': int(len(df)),
        'report': report
    }
    with open(MODELS_DIR / 'intent_training_report.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(f"✅ Trained intent classifier — accuracy: {acc:.4f}")
    print(f"✅ Saved vectorizer to: {MODELS_DIR / 'intent_vectorizer.joblib'}")
    print(f"✅ Saved classifier to: {MODELS_DIR / 'intent_classifier.joblib'}")
    print(f"✅ Report saved to: {MODELS_DIR / 'intent_training_report.json'}")

if __name__ == '__main__':
    main()
