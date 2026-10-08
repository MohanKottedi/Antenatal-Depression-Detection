# ============================================================
# Sneha Antenatal Detection
# Experiment B Preprocessing
# Risk Factors Only
# ============================================================

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    MODEL_DIR,
    FEATURES_EXPERIMENT_A,
    FEATURES_EXPERIMENT_B,
)


# ============================================================
# PATHS
# ============================================================

SPLIT_PATH = MODEL_DIR / "data_splits.joblib"

PREPROCESSOR_PATH = (
    MODEL_DIR / "preprocessor_experiment_B.joblib"
)

PROCESSED_PATH = (
    MODEL_DIR / "processed_experiment_B.joblib"
)


# ============================================================
# LOAD SPLIT
# ============================================================

def load_splits():

    if not SPLIT_PATH.exists():

        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_PATH}"
        )

    splits = joblib.load(SPLIT_PATH)

    required_keys = [
        "X_train",
        "X_val",
        "X_test",
        "y_train",
        "y_val",
        "y_test",
    ]

    for key in required_keys:

        if key not in splits:

            raise KeyError(
                f"Missing '{key}' from split package."
            )

    return splits


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

def convert_to_dataframe(
    X,
    feature_columns,
):

    if isinstance(X, pd.DataFrame):

        df = X.copy()

        if len(df.columns) != len(feature_columns):

            raise ValueError(
                "Number of columns does not match "
                "Experiment A feature definition."
            )

        df.columns = feature_columns

        return df

    X = np.asarray(X)

    if X.ndim != 2:

        raise ValueError(
            f"Expected 2D matrix. Got {X.shape}"
        )

    if X.shape[1] != len(feature_columns):

        raise ValueError(
            f"Expected {len(feature_columns)} columns, "
            f"got {X.shape[1]}"
        )

    return pd.DataFrame(
        X,
        columns=feature_columns,
    )


# ============================================================
# SELECT EXPERIMENT B FEATURES
# ============================================================

def select_risk_features(df):

    missing = [
        column
        for column in FEATURES_EXPERIMENT_B
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing Experiment B features:\n{missing}"
        )

    return df[
        FEATURES_EXPERIMENT_B
    ].copy()


# ============================================================
# DETECT FEATURE TYPES
# ============================================================

def detect_feature_types(df):

    numeric_columns = []
    categorical_columns = []

    for column in FEATURES_EXPERIMENT_B:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            numeric_columns.append(column)

        else:

            categorical_columns.append(column)

    return (
        numeric_columns,
        categorical_columns,
    )


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(
    numeric_columns,
    categorical_columns,
):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("SNEHA ANTENATAL DETECTION")
    print("EXPERIMENT B PREPROCESSING")
    print("RISK FACTORS ONLY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load existing split
    # --------------------------------------------------------

    splits = load_splits()

    X_train_A = convert_to_dataframe(
        splits["X_train"],
        FEATURES_EXPERIMENT_A,
    )

    X_val_A = convert_to_dataframe(
        splits["X_val"],
        FEATURES_EXPERIMENT_A,
    )

    X_test_A = convert_to_dataframe(
        splits["X_test"],
        FEATURES_EXPERIMENT_A,
    )

    y_train = np.asarray(
        splits["y_train"],
        dtype=np.int64,
    )

    y_val = np.asarray(
        splits["y_val"],
        dtype=np.int64,
    )

    y_test = np.asarray(
        splits["y_test"],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Select ONLY risk factors
    # --------------------------------------------------------

    X_train = select_risk_features(
        X_train_A
    )

    X_val = select_risk_features(
        X_val_A
    )

    X_test = select_risk_features(
        X_test_A
    )

    # --------------------------------------------------------
    # Verify exactly 16 features
    # --------------------------------------------------------

    if X_train.shape[1] != 16:

        raise ValueError(
            f"Experiment B should contain 16 features. "
            f"Found {X_train.shape[1]}."
        )

    print("\nOriginal feature count:")
    print("Experiment A: 25")
    print("Experiment B: 16")

    print("\nExperiment B features:")

    for index, column in enumerate(
        FEATURES_EXPERIMENT_B,
        start=1,
    ):

        print(
            f"{index:2d}. {column}"
        )

    # --------------------------------------------------------
    # Detect types
    # --------------------------------------------------------

    (
        numeric_columns,
        categorical_columns,
    ) = detect_feature_types(
        X_train
    )

    print("\nNumeric features:")

    for column in numeric_columns:
        print(f"  - {column}")

    print("\nCategorical features:")

    for column in categorical_columns:
        print(f"  - {column}")

    # --------------------------------------------------------
    # Build preprocessor
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    # --------------------------------------------------------
    # FIT ONLY ON TRAIN
    # --------------------------------------------------------

    print(
        "\nFitting Experiment B preprocessor "
        "on TRAINING data only..."
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    # --------------------------------------------------------
    # Transform validation/test
    # --------------------------------------------------------

    X_val_processed = (
        preprocessor.transform(
            X_val
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32,
    )

    X_val_processed = np.asarray(
        X_val_processed,
        dtype=np.float32,
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validate finite values
    # --------------------------------------------------------

    for name, array in [
        ("Train", X_train_processed),
        ("Validation", X_val_processed),
        ("Test", X_test_processed),
    ]:

        if not np.isfinite(array).all():

            raise ValueError(
                f"{name} contains NaN or infinity "
                "after preprocessing."
            )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    # --------------------------------------------------------
    # Print shapes
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PROCESSED SHAPES")
    print("=" * 70)

    print(
        f"Train      : {X_train_processed.shape}"
    )

    print(
        f"Validation : {X_val_processed.shape}"
    )

    print(
        f"Test       : {X_test_processed.shape}"
    )

    print(
        f"Final processed features: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Save preprocessor
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    # --------------------------------------------------------
    # Save processed data
    # --------------------------------------------------------

    package = {

        "experiment": "Experiment B",

        "description": "Risk factors only",

        "X_train": X_train_processed,

        "X_val": X_val_processed,

        "X_test": X_test_processed,

        "y_train": y_train,

        "y_val": y_val,

        "y_test": y_test,

        "feature_names": feature_names,

        "original_feature_columns": (
            FEATURES_EXPERIMENT_B
        ),

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns,

        "preprocessor_path": str(
            PREPROCESSOR_PATH
        ),
    }

    joblib.dump(
        package,
        PROCESSED_PATH,
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXPERIMENT B PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Original features : "
        f"{len(FEATURES_EXPERIMENT_B)}"
    )

    print(
        f"Processed features: "
        f"{len(feature_names)}"
    )

    print(
        f"\nPreprocessor saved:\n"
        f"{PREPROCESSOR_PATH}"
    )

    print(
        f"\nProcessed data saved:\n"
        f"{PROCESSED_PATH}"
    )

    print(
        "\nSUCCESS: Experiment B preprocessing completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()