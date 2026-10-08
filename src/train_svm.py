# ============================================================
# Sneha Antenatal Detection
# SVM Training
# ============================================================

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

from config import MODEL_DIR


# ============================================================
# PATHS
# ============================================================

PROCESSED_PATH = (
    MODEL_DIR / "processed_experiment_A.joblib"
)

MODEL_PATH = (
    MODEL_DIR / "svm_experiment_A.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_processed_data():

    if not PROCESSED_PATH.exists():

        raise FileNotFoundError(
            f"Processed data not found:\n"
            f"{PROCESSED_PATH}\n\n"
            "Run first:\n"
            "python src\\preprocessing.py"
        )

    data = joblib.load(
        PROCESSED_PATH
    )

    print("=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)

    print(
        f"Processed package: {PROCESSED_PATH}"
    )

    return data


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name,
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    cm = confusion_matrix(
        y,
        predictions,
    )

    print("\n")
    print("=" * 70)
    print(f"{dataset_name.upper()} RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Not Depressed",
                "Depressed",
            ],
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


# ============================================================
# TRAIN SVM
# ============================================================

def train_svm():

    data = load_processed_data()

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    print("\nDataset shapes:")

    print(
        f"Train      : {X_train.shape}"
    )

    print(
        f"Validation : {X_val.shape}"
    )

    print(
        f"Test       : {X_test.shape}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING SVM")
    print("=" * 70)

    model = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    params = model.get_params()

    print("\nSVM configuration:")

    print(
        f"C              : {params['C']}"
    )

    print(
        f"Kernel          : {params['kernel']}"
    )

    print(
        f"Gamma           : {params['gamma']}"
    )

    print(
        f"Class weight    : {params['class_weight']}"
    )

    print(
        f"Probability     : {params['probability']}"
    )

    print(
        f"Random state    : {params['random_state']}"
    )

    print("\nTraining...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_results = evaluate_model(
        model,
        X_val,
        y_val,
        "Validation",
    )

    # ========================================================
    # TEST
    # ========================================================

    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        "Test",
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_package = {

        "model": model,

        "model_name": "SVM",

        "experiment": "Experiment A",

        "feature_names": data.get(
            "feature_names",
            None,
        ),

        "validation_results": validation_results,

        "test_results": test_results,

        "random_state": 42,
    }

    joblib.dump(
        model_package,
        MODEL_PATH,
    )

    print("\n")
    print("=" * 70)
    print("SVM SAVED")
    print("=" * 70)

    print(
        f"Model path:\n{MODEL_PATH}"
    )

    print("\nSUCCESS: SVM training complete.")

    return model_package


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_svm()