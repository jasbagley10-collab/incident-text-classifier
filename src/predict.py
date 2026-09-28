from pathlib import Path
import sys

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "incident_classifier.joblib"

def main():
    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Allow the user to provide an incident directly from the command line.
    if len(sys.argv) > 1:
        incident = " ".join(sys.argv[1:])
    else:
        incident = input("Enter an incident report: ")

    # Make a prediction
    prediction = model.predict([incident])[0]

    # Get probability estimates for each category
    probabilities = model.predict_proba([incident])[0]
    classes = model.classes_

    print()
    print(f"Predicted category: {prediction}")
    print()
    print("Class probabilities:")

    for category, probability in zip(classes, probabilities):
        print(f"{category}: {probability:.2f}")

if __name__ == "__main__":
    main()
