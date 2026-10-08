from pathlib import Path
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

PROCESSED_PATH = MODEL_DIR / "processed_experiment_B.joblib"
MODEL_PATH = MODEL_DIR / "svm_experiment_B.joblib"


def evaluate_model(model, X, y, split_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)

    print(f"\n{'=' * 60}")
    print(f"SVM Experiment B - {split_name} Results")
    print(f"{'=' * 60}")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            y_pred,
            target_names=["Not Depressed", "Depressed"],
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


def main():
    print("=" * 60)
    print("SVM - EXPERIMENT B")
    print("Risk Factors Only")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load processed Experiment B data
    # ---------------------------------------------------------
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed Experiment B data not found:\n{PROCESSED_PATH}\n"
            "Run preprocess_experiment_b.py first."
        )

    data = joblib.load(PROCESSED_PATH)

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    print("\nData loaded successfully.")
    print(f"Training data   : {X_train.shape}")
    print(f"Validation data : {X_val.shape}")
    print(f"Test data       : {X_test.shape}")

    # ---------------------------------------------------------
    # Validate data
    # ---------------------------------------------------------
    for name, X in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ]:
        if not np.isfinite(X).all():
            raise ValueError(f"{name} contains NaN or infinite values.")

    # ---------------------------------------------------------
    # Build SVM
    # ---------------------------------------------------------
    print("\nCreating SVM model...")

    model = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    print("\nSVM configuration:")
    params = model.get_params()

    for key in [
        "C",
        "kernel",
        "gamma",
        "probability",
        "class_weight",
        "random_state",
    ]:
        print(f"{key}: {params[key]}")

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    print("\nTraining SVM...")
    model.fit(X_train, y_train)

    print("Training completed.")

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------
    validation_results = evaluate_model(
        model,
        X_val,
        y_val,
        "Validation",
    )

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "Test",
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------
    model_package = {
        "model": model,
        "experiment": "B",
        "description": "SVM using antenatal risk factors only",
        "feature_count": X_train.shape[1],
        "validation_results": validation_results,
        "test_results": test_results,
        "random_state": 42,
    }

    joblib.dump(model_package, MODEL_PATH)

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)
    print(MODEL_PATH)

    print("\nSVM Experiment B completed successfully.")


if __name__ == "__main__":
    main()