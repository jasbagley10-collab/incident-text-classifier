```python
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "incidents.csv"
MODEL_PATH = BASE_DIR / "incident_classifier.joblib"

def main():
    # Load the dataset
    data = pd.read_csv(DATA_PATH)
    # Separate input text from target labels
    texts = data["text"]
    labels = data["label"]
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    # Build the machine-learning pipeline
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2)
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                ),
            ),
        ]
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on data the model did not train on
    predictions = model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.3f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save the trained pipeline
    joblib.dump(model, MODEL_PATH)

    print()
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
