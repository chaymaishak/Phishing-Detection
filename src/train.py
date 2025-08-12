import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd

from src.data_loader import load_csv
from src.preprocess import build_tfidf

def train_pipeline(csv_path, model_out_path="model.joblib", vec_out_path="vectorizer.joblib"):
    df = load_csv(csv_path)
    df = df.dropna(subset=['text','label'])
    X_texts = df['text'].astype(str).tolist()
    y = df['label'].apply(lambda x: 1 if x.lower()=='phishing' else 0).values

    vectorizer, X = build_tfidf(X_texts)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Save artifacts
    joblib.dump(clf, model_out_path)
    joblib.dump(vectorizer, vec_out_path)
    print(f"Saved model -> {model_out_path}")
    print(f"Saved vectorizer -> {vec_out_path}")

if __name__ == "__main__":
    # default train on sample data
    train_pipeline("data/sample_emails.csv")
