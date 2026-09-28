from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "incidents.csv"
MODEL_PATH = BASE_DIR / "incident_classifier.joblib"
RESULTS_DIR = BASE_DIR / "results"

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load dataset and trained model
    data = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    texts = data["text"]
    labels = data["label"]

    # Recreate the same deterministic test split used during training
    _, X_test, _, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    # Generate predictions
    predictions = model.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, predictions)

    report_path = RESULTS_DIR / "classification_report.txt"
    report_path.write_text(report)

    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    labels_sorted = sorted(labels.unique())

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels_sorted,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels_sorted,
    )

    display.plot(xticks_rotation=45)
    plt.tight_layout()

    confusion_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=200)
    plt.close()

    print(f"Saved classification report to: {report_path}")
    print(f"Saved confusion matrix to: {confusion_path}")

if __name__ == "__main__":
    main()
