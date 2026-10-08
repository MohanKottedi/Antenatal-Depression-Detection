from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    DATASET_PATH,
    MODEL_DIR,
    TARGET_COLUMN,
    FEATURES_EXPERIMENT_A,
    FEATURES_EXPERIMENT_B,
)

from data_cleaning import clean_dataset


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD + CLEAN
# ============================================================

def load_and_clean_data():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(
        f"Raw dataset shape: {df.shape}"
    )

    df_clean = clean_dataset(df)

    return df_clean


# ============================================================
# CREATE SPLIT
# ============================================================

def create_data_split(df):

    # --------------------------------------------------------
    # Remove target and Scalling from model inputs
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            TARGET_COLUMN,
            "Scalling",
        ]
    )

    y = df[TARGET_COLUMN].astype(int)

    # --------------------------------------------------------
    # 70% TRAIN
    # 30% TEMPORARY
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    # --------------------------------------------------------
    # 15% VALIDATION
    # 15% TEST
    # --------------------------------------------------------

    X_val, X_test, y_val, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=y_temp,
        )
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# PRINT SPLIT INFORMATION
# ============================================================

def print_split_information(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):

    total = (
        len(X_train)
        + len(X_val)
        + len(X_test)
    )

    print("\n" + "=" * 70)
    print("FINAL DATA SPLIT")
    print("=" * 70)

    print(
        f"Total:      {total}"
    )

    print(
        f"Train:      {len(X_train)} "
        f"({len(X_train) / total:.2%})"
    )

    print(
        f"Validation: {len(X_val)} "
        f"({len(X_val) / total:.2%})"
    )

    print(
        f"Test:       {len(X_test)} "
        f"({len(X_test) / total:.2%})"
    )

    print("\nFeature count:")
    print(
        f"Train:      {X_train.shape[1]}"
    )
    print(
        f"Validation: {X_val.shape[1]}"
    )
    print(
        f"Test:       {X_test.shape[1]}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TARGET DISTRIBUTION")
    print("-" * 70)

    for name, y in [
        ("TRAIN", y_train),
        ("VALIDATION", y_val),
        ("TEST", y_test),
    ]:

        counts = (
            y.value_counts()
            .sort_index()
        )

        proportions = (
            y.value_counts(
                normalize=True
            )
            .sort_index()
        )

        print(f"\n{name}")

        print(
            f"Not Depressed (0): "
            f"{counts.get(0, 0)} "
            f"({proportions.get(0, 0):.2%})"
        )

        print(
            f"Depressed (1):     "
            f"{counts.get(1, 0)} "
            f"({proportions.get(1, 0):.2%})"
        )


# ============================================================
# CHECK OVERLAP
# ============================================================

def get_record_hashes(X, y):

    combined = X.copy()

    combined["__target__"] = (
        y.to_numpy()
    )

    return set(
        pd.util.hash_pandas_object(
            combined,
            index=False,
        ).values
    )


def verify_no_overlap(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):

    print("\n" + "=" * 70)
    print("OVERLAP CHECK")
    print("=" * 70)

    train_hashes = get_record_hashes(
        X_train,
        y_train,
    )

    val_hashes = get_record_hashes(
        X_val,
        y_val,
    )

    test_hashes = get_record_hashes(
        X_test,
        y_test,
    )

    train_val = (
        train_hashes & val_hashes
    )

    train_test = (
        train_hashes & test_hashes
    )

    val_test = (
        val_hashes & test_hashes
    )

    print(
        "Train ↔ Validation:",
        len(train_val),
    )

    print(
        "Train ↔ Test:",
        len(train_test),
    )

    print(
        "Validation ↔ Test:",
        len(val_test),
    )

    assert len(train_val) == 0
    assert len(train_test) == 0
    assert len(val_test) == 0

    print(
        "\nSUCCESS: No exact record overlap."
    )


# ============================================================
# CHECK EXPERIMENT FEATURES
# ============================================================

def verify_experiment_features(
    X_train,
    X_val,
    X_test,
):

    print("\n" + "=" * 70)
    print("EXPERIMENT FEATURE CHECK")
    print("=" * 70)

    experiments = {
        "Experiment A": FEATURES_EXPERIMENT_A,
        "Experiment B": FEATURES_EXPERIMENT_B,
    }

    for name, features in experiments.items():

        for dataset_name, X in [
            ("Train", X_train),
            ("Validation", X_val),
            ("Test", X_test),
        ]:

            missing = [
                column
                for column in features
                if column not in X.columns
            ]

            if missing:

                raise ValueError(
                    f"{name} missing features "
                    f"in {dataset_name}: {missing}"
                )

        print(
            f"{name}: "
            f"{len(features)} features - PASSED"
        )


# ============================================================
# SAVE SPLIT
# ============================================================

def save_split(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):

    split_package = {

        "X_train": X_train,

        "X_val": X_val,

        "X_test": X_test,

        "y_train": y_train,

        "y_val": y_val,

        "y_test": y_test,

        "random_state": RANDOM_STATE,

        "train_ratio": TRAIN_RATIO,

        "validation_ratio": VALIDATION_RATIO,

        "test_ratio": TEST_RATIO,
    }

    output_path = (
        MODEL_DIR
        / "data_splits.joblib"
    )

    joblib.dump(
        split_package,
        output_path,
    )

    print(
        "\nSaved split package:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SNEHA ANTENATAL DETECTION")
    print("DATA SPLITTING PIPELINE")
    print("=" * 70)

    # Load + clean
    df = load_and_clean_data()

    # Create split
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = create_data_split(df)

    # Information
    print_split_information(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    # Overlap
    verify_no_overlap(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    # Features
    verify_experiment_features(
        X_train,
        X_val,
        X_test,
    )

    # Save
    save_split(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )

    print("\n" + "=" * 70)
    print("DATA SPLITTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()