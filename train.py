import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_PATH = os.path.join("..", "data", "features.csv")
MODEL_DIR = os.path.join("..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "svm.pkl")
REPORT_PATH = os.path.join(MODEL_DIR, "report.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # SVM pipeline with class_weight="balanced"
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced"))
    ])

    # Expanded hyperparameter search
    params = {
        "svm__C": [0.1, 1, 5, 10, 50],
        "svm__gamma": ["scale", "auto", 0.1, 0.01, 0.001]
    }

    grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    # Predictions
    y_pred = grid.predict(X_test)

    # Print results
    print("[INFO] Best params:", grid.best_params_)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(grid.best_estimator_, MODEL_PATH)
    print(f"[INFO] Saved model to {MODEL_PATH}")

    # Save report to file
    with open(REPORT_PATH, "w") as f:
        f.write(f"Best Params: {grid.best_params_}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(y_test, y_pred)))

    print(f"[INFO] Saved report to {REPORT_PATH}")

if __name__ == "__main__":
    main()
