# ============================================================
# Sneha Antenatal Detection
# Random Forest - Experiment B
# Risk Factors Only
# ============================================================

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
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
    MODEL_DIR / "processed_experiment_B.joblib"
)

MODEL_PATH = (
    MODEL_DIR / "random_forest_experiment_B.joblib"
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
            "python src\\preprocess_experiment_b.py"
        )

    data = joblib.load(PROCESSED_PATH)

    print("=" * 70)
    print("LOADING EXPERIMENT B DATA")
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

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

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
# TRAIN RANDOM FOREST
# ============================================================

def train_random_forest():

    data = load_processed_data()

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    print("\nDataset shapes:")

    print(f"Train      : {X_train.shape}")
    print(f"Validation : {X_val.shape}")
    print(f"Test       : {X_test.shape}")

    # ========================================================
    # MODEL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING RANDOM FOREST - EXPERIMENT B")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    params = model.get_params()

    print("\nRandom Forest configuration:")

    print(
        f"Estimators    : {params['n_estimators']}"
    )

    print(
        f"Max depth     : {params['max_depth']}"
    )

    print(
        f"Max features  : {params['max_features']}"
    )

    print(
        f"Class weight  : {params['class_weight']}"
    )

    print(
        f"Random state  : {params['random_state']}"
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
    # FEATURE IMPORTANCE
    # ========================================================

    feature_names = data.get(
        "feature_names",
        None,
    )

    if feature_names is not None:

        importances = model.feature_importances_

        sorted_indices = np.argsort(
            importances
        )[::-1]

        print("\n")
        print("=" * 70)
        print("TOP 20 FEATURE IMPORTANCES")
        print("=" * 70)

        for rank, index in enumerate(
            sorted_indices[:20],
            start=1,
        ):

            print(
                f"{rank:2d}. "
                f"{feature_names[index]} "
                f"-> "
                f"{importances[index]:.6f}"
            )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_package = {

        "model": model,

        "model_name": "Random Forest",

        "experiment": "Experiment B",

        "description": "Risk factors only",

        "feature_names": feature_names,

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
    print("RANDOM FOREST EXPERIMENT B SAVED")
    print("=" * 70)

    print(
        f"Model path:\n{MODEL_PATH}"
    )

    print(
        "\nSUCCESS: Random Forest Experiment B complete."
    )

    return model_package


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_random_forest()