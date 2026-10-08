from pathlib import Path
import joblib
import torch
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

RF_A_PATH = MODEL_DIR / "random_forest_experiment_A.joblib"
SVM_A_PATH = MODEL_DIR / "svm_experiment_A.joblib"

RF_B_PATH = MODEL_DIR / "random_forest_experiment_B.joblib"
SVM_B_PATH = MODEL_DIR / "svm_experiment_B.joblib"

TRANSFORMER_A_PATH = (
    MODEL_DIR / "tabular_transformer_experiment_A.pt"
)

TRANSFORMER_B_PATH = (
    MODEL_DIR / "tabular_transformer_experiment_B.pt"
)

OUTPUT_PATH = (
    MODEL_DIR / "model_comparison_results.csv"
)


# ============================================================
# HELPERS
# ============================================================

def extract_sklearn_results(path, model_name, experiment):

    if not path.exists():
        print(f"WARNING: Missing {path}")
        return None

    package = joblib.load(path)

    validation = package["validation_results"]
    test = package["test_results"]

    return {
        "Experiment": experiment,
        "Model": model_name,

        "Validation Accuracy":
            validation["accuracy"],

        "Validation Precision":
            validation["precision"],

        "Validation Recall":
            validation["recall"],

        "Validation F1":
            validation["f1"],

        "Validation ROC-AUC":
            validation["roc_auc"],

        "Test Accuracy":
            test["accuracy"],

        "Test Precision":
            test["precision"],

        "Test Recall":
            test["recall"],

        "Test F1":
            test["f1"],

        "Test ROC-AUC":
            test["roc_auc"],
    }


def extract_transformer_results(
    path,
    model_name,
    experiment,
):

    if not path.exists():
        print(f"WARNING: Missing {path}")
        return None

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    validation = checkpoint[
        "validation_metrics"
    ]

    test = checkpoint[
        "test_metrics"
    ]

    return {
        "Experiment": experiment,
        "Model": model_name,

        "Validation Accuracy":
            validation["accuracy"],

        "Validation Precision":
            validation["precision"],

        "Validation Recall":
            validation["recall"],

        "Validation F1":
            validation["f1"],

        "Validation ROC-AUC":
            validation["roc_auc"],

        "Test Accuracy":
            test["accuracy"],

        "Test Precision":
            test["precision"],

        "Test Recall":
            test["recall"],

        "Test F1":
            test["f1"],

        "Test ROC-AUC":
            test["roc_auc"],
    }


def print_results(df):

    percentage_columns = [
        "Validation Accuracy",
        "Validation Precision",
        "Validation Recall",
        "Validation F1",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Test F1",
    ]

    display_df = df.copy()

    for column in percentage_columns:
        display_df[column] = (
            display_df[column] * 100
        ).map(
            lambda value: f"{value:.2f}%"
        )

    display_df["Validation ROC-AUC"] = (
        display_df["Validation ROC-AUC"]
        .map(lambda value: f"{value:.4f}")
    )

    display_df["Test ROC-AUC"] = (
        display_df["Test ROC-AUC"]
        .map(lambda value: f"{value:.4f}")
    )

    print("\n" + "=" * 120)
    print("FINAL MODEL COMPARISON")
    print("=" * 120)

    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SNEHA ANTENATAL DETECTION")
    print("RF vs SVM vs TABULAR TRANSFORMER")
    print("=" * 80)

    results = []

    # --------------------------------------------------------
    # Experiment A
    # --------------------------------------------------------

    result = extract_sklearn_results(
        RF_A_PATH,
        "Random Forest",
        "A",
    )

    if result is not None:
        results.append(result)

    result = extract_sklearn_results(
        SVM_A_PATH,
        "SVM",
        "A",
    )

    if result is not None:
        results.append(result)

    result = extract_transformer_results(
        TRANSFORMER_A_PATH,
        "Tabular Transformer",
        "A",
    )

    if result is not None:
        results.append(result)

    # --------------------------------------------------------
    # Experiment B
    # --------------------------------------------------------

    result = extract_sklearn_results(
        RF_B_PATH,
        "Random Forest",
        "B",
    )

    if result is not None:
        results.append(result)

    result = extract_sklearn_results(
        SVM_B_PATH,
        "SVM",
        "B",
    )

    if result is not None:
        results.append(result)

    result = extract_transformer_results(
        TRANSFORMER_B_PATH,
        "Tabular Transformer",
        "B",
    )

    if result is not None:
        results.append(result)

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "No trained model results were found."
        )

    df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Sort by experiment and test ROC-AUC
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "Experiment",
            "Test ROC-AUC",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_results(df)

    # --------------------------------------------------------
    # Best model for each experiment
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "BEST MODEL BY EXPERIMENT"
    )

    print(
        "=" * 80
    )

    for experiment in ["A", "B"]:

        experiment_df = df[
            df["Experiment"] == experiment
        ]

        if experiment_df.empty:
            continue

        best_index = experiment_df[
            "Test ROC-AUC"
        ].idxmax()

        best = experiment_df.loc[
            [best_index]
        ].iloc[0]

        print(
            f"\nExperiment {experiment}: "
            f"{best['Model']}"
        )

        print(
            f"Test Accuracy : "
            f"{best['Test Accuracy']:.4f}"
        )

        print(
            f"Test F1       : "
            f"{best['Test F1']:.4f}"
        )

        print(
            f"Test ROC-AUC   : "
            f"{best['Test ROC-AUC']:.4f}"
        )

    # --------------------------------------------------------
    # Final file
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "COMPARISON COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSaved comparison:"
        f"\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()