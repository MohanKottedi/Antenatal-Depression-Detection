# ============================================================
# Sneha Antenatal Detection
# Data Cleaning and Validation
# ============================================================

from pathlib import Path

import pandas as pd

from config import (
    DATASET_PATH,
    TARGET_COLUMN,
    TARGET_MAPPING,
    SYMPTOM_COLUMNS,
    RISK_FACTOR_COLUMNS,
    FEATURES_EXPERIMENT_A,
    FEATURES_EXPERIMENT_B,
    EXCLUDED_COLUMNS,
    CATEGORICAL_CLEANUP_MAPS,
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset():
    """
    Load the raw antenatal dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("=" * 70)
    print("DATASET LOADING")
    print("=" * 70)

    print(f"Dataset path : {DATASET_PATH}")
    print(f"Raw rows     : {len(df)}")
    print(f"Raw columns  : {len(df.columns)}")

    return df


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):
    """
    Remove unnecessary whitespace from column names.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# 3. CLEAN STRING VALUES
# ============================================================

def clean_string_values(df):
    """
    Clean whitespace and convert empty string-like values
    into missing values.
    """

    df = df.copy()

    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

            # Convert empty strings to NaN
            df[column] = df[column].replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "None": pd.NA,
                    "none": pd.NA,
                    "NULL": pd.NA,
                    "null": pd.NA,
                }
            )

    return df


# ============================================================
# 4. STANDARDIZE CATEGORICAL VALUES
# ============================================================

def standardize_categorical_values(df):
    """
    Apply known categorical corrections.

    Example:
        YY -> Yes

    This is based on the categorical cleanup rule identified
    during dataset preparation.
    """

    df = df.copy()

    for column, mapping in CATEGORICAL_CLEANUP_MAPS.items():

        if column in df.columns:

            df[column] = df[column].replace(mapping)

    return df


# ============================================================
# 5. VALIDATE REQUIRED COLUMNS
# ============================================================

def validate_required_columns(df):
    """
    Verify that all columns required by the project exist.
    """

    required_columns = (
        FEATURES_EXPERIMENT_A
        + [
            "Scalling",
            TARGET_COLUMN,
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("\nMissing required columns:")

        for column in missing_columns:
            print(f"  - {column}")

        raise ValueError(
            "Dataset is missing required project columns."
        )

    print("\nRequired columns: PASSED")


# ============================================================
# 6. CLEAN TARGET COLUMN
# ============================================================

def clean_target_column(df):
    """
    Clean the Labelling target column.

    Dataset labels:
        Depressed -> 1
        Not       -> 0
    """

    df = df.copy()

    print("\n" + "=" * 70)
    print("TARGET CLEANING")
    print("=" * 70)

    # Convert target to string and remove whitespace
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype("string")
        .str.strip()
    )

    print("\nTarget values BEFORE mapping:")
    print(
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .to_string()
    )

    # Map the actual dataset labels
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .map(TARGET_MAPPING)
    )

    # Check invalid/missing values
    invalid_count = df[TARGET_COLUMN].isna().sum()

    if invalid_count > 0:

        raise ValueError(
            f"Found {invalid_count} invalid/missing target values "
            f"in '{TARGET_COLUMN}' after mapping."
        )

    # Convert to integer
    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(int)
    )

    print("\nTarget values AFTER mapping:")

    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTarget mapping:")
    print("  Depressed -> 1")
    print("  Not       -> 0")

    return df

# ============================================================
# 7. REMOVE EXACT DUPLICATES
# ============================================================

def remove_duplicates(df):
    """
    Remove exact duplicate rows.
    """

    df = df.copy()

    before = len(df)

    df = df.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)

    after = len(df)

    removed = before - after

    print("\nDuplicate removal:")
    print(f"  Rows before : {before}")
    print(f"  Rows after  : {after}")
    print(f"  Removed     : {removed}")

    return df


# ============================================================
# 8. VALIDATE SCALLING
# ============================================================

def validate_symptom_score(df):
    """
    Verify that Scalling equals the sum of the nine symptom
    columns.

    The project dataset uses the symptom score to generate
    the Labelling target.
    """

    print("\n" + "=" * 70)
    print("SYMPTOM SCORE VALIDATION")
    print("=" * 70)

    symptom_sum = (
        df[SYMPTOM_COLUMNS]
        .sum(axis=1)
    )

    scaling = pd.to_numeric(
        df["Scalling"],
        errors="coerce"
    )

    # Check missing values
    if scaling.isna().any():

        raise ValueError(
            "Scalling contains missing or non-numeric values."
        )

    # Compare values
    score_matches = (
        symptom_sum == scaling
    )

    all_match = score_matches.all()

    print(
        f"Scalling equals symptom sum: {all_match}"
    )

    if not all_match:

        mismatch_count = (
            ~score_matches
        ).sum()

        print(
            f"Number of mismatches: {mismatch_count}"
        )

        mismatch_df = pd.DataFrame(
            {
                "Calculated_Symptom_Sum":
                    symptom_sum[~score_matches],

                "Scalling":
                    scaling[~score_matches],
            }
        )

        print("\nExample mismatches:")
        print(
            mismatch_df.head(10)
        )

        raise ValueError(
            "Scalling does not match the sum of symptom columns."
        )

    print(
        f"Symptom score minimum: {symptom_sum.min()}"
    )

    print(
        f"Symptom score maximum: {symptom_sum.max()}"
    )

    return df


# ============================================================
# 9. VALIDATE LABEL GENERATION RULE
# ============================================================

def validate_label_generation(df):
    """
    Validate the relationship between symptom score and
    the actual Labelling target.

    This function DOES NOT overwrite the target.

    It only reports the relationship found in the dataset.
    """

    symptom_sum = (
        df[SYMPTOM_COLUMNS]
        .sum(axis=1)
    )

    actual_labels = df[TARGET_COLUMN]

    print("\n" + "=" * 70)
    print("SYMPTOM SCORE / LABEL ANALYSIS")
    print("=" * 70)

    print("\nMean symptom score by label:")

    print(
        df.assign(
            SymptomScore=symptom_sum
        )
        .groupby(TARGET_COLUMN)["SymptomScore"]
        .agg(
            ["count", "min", "max", "mean", "median"]
        )
        .to_string()
    )

    # Cross-tabulation
    print("\nLabel distribution by symptom-score group:")

    score_groups = pd.cut(
        symptom_sum,
        bins=[-float("inf"), 9, float("inf")],
        labels=[
            "Score < 10",
            "Score >= 10"
        ]
    )

    cross_tab = pd.crosstab(
        score_groups,
        actual_labels
    )

    print(
        cross_tab.to_string()
    )

    print(
        "\nNOTE: The dataset's Labelling column is preserved "
        "as the ground-truth target."
    )

    return df

# ============================================================
# 10. CHECK MISSING VALUES
# ============================================================

def report_missing_values(df):
    """
    Report missing values without deleting them.

    Missing values will later be handled by the preprocessing
    pipeline using imputation.
    """

    missing = (
        df.isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    if len(missing) == 0:

        print("No missing values found.")

    else:

        print(missing.to_string())

    return missing


# ============================================================
# 11. REPORT DATA TYPES
# ============================================================

def report_data_types(df):
    """
    Display column data types.
    """

    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)

    dtype_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
        }
    )

    print(
        dtype_df.to_string(index=False)
    )

    return dtype_df


# ============================================================
# 12. VALIDATE EXPERIMENT FEATURES
# ============================================================

def validate_experiment_features(df):
    """
    Validate the two planned experiments.

    Experiment A:
        Risk factors + symptoms
        25 features

    Experiment B:
        Risk factors only
        16 features
    """

    print("\n" + "=" * 70)
    print("EXPERIMENT FEATURE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Experiment A
    # --------------------------------------------------------

    missing_a = [
        column
        for column in FEATURES_EXPERIMENT_A
        if column not in df.columns
    ]

    if missing_a:

        raise ValueError(
            f"Experiment A missing features: {missing_a}"
        )

    print(
        f"Experiment A: {len(FEATURES_EXPERIMENT_A)} features PASSED"
    )

    # --------------------------------------------------------
    # Experiment B
    # --------------------------------------------------------

    missing_b = [
        column
        for column in FEATURES_EXPERIMENT_B
        if column not in df.columns
    ]

    if missing_b:

        raise ValueError(
            f"Experiment B missing features: {missing_b}"
        )

    print(
        f"Experiment B: {len(FEATURES_EXPERIMENT_B)} features PASSED"
    )

    # --------------------------------------------------------
    # Check Scalling leakage
    # --------------------------------------------------------

    for feature_list_name, feature_list in [
        ("Experiment A", FEATURES_EXPERIMENT_A),
        ("Experiment B", FEATURES_EXPERIMENT_B),
    ]:

        if "Scalling" in feature_list:

            raise ValueError(
                f"Scalling must not be used as a model feature "
                f"in {feature_list_name}."
            )

    print(
        "Scalling excluded from model features: PASSED"
    )

    # --------------------------------------------------------
    # Check target leakage
    # --------------------------------------------------------

    for feature_list_name, feature_list in [
        ("Experiment A", FEATURES_EXPERIMENT_A),
        ("Experiment B", FEATURES_EXPERIMENT_B),
    ]:

        if TARGET_COLUMN in feature_list:

            raise ValueError(
                f"Target column '{TARGET_COLUMN}' found in "
                f"{feature_list_name}."
            )

    print(
        "Target column excluded from model features: PASSED"
    )


# ============================================================
# 13. FINAL DATA VALIDATION
# ============================================================

def final_validation(df):
    """
    Perform final sanity checks before saving/using the dataset.
    """

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    # Check index
    if not df.index.is_unique:

        raise ValueError(
            "Dataset index is not unique."
        )

    # Check target
    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            "Target contains missing values."
        )

    # Check target values
    if not set(
        df[TARGET_COLUMN].unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Target contains values other than 0 and 1."
        )

    # Check duplicates
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        raise ValueError(
            f"Dataset still contains {duplicate_count} duplicates."
        )

    print("Index uniqueness          : PASSED")
    print("Target missing values     : PASSED")
    print("Target values {0,1}       : PASSED")
    print("Duplicate check           : PASSED")

    print("\nFinal dataset:")
    print(
        f"  Rows    : {len(df)}"
    )
    print(
        f"  Columns : {len(df.columns)}"
    )

    print("\nFinal target distribution:")

    print(
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    return df


# ============================================================
# 14. COMPLETE CLEANING PIPELINE
# ============================================================

def clean_dataset(df):
    """
    Complete dataset cleaning and validation pipeline.

    Returns:
        pandas.DataFrame
    """

    print("\n")
    print("=" * 70)
    print("SNEHA ANTENATAL DETECTION")
    print("DATA CLEANING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df = clean_column_names(df)

    # --------------------------------------------------------
    # Clean string values
    # --------------------------------------------------------

    df = clean_string_values(df)

    # --------------------------------------------------------
    # Standardize categorical values
    # --------------------------------------------------------

    df = standardize_categorical_values(df)

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    validate_required_columns(df)

    # --------------------------------------------------------
    # Clean target
    # --------------------------------------------------------

    df = clean_target_column(df)

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicates(df)

    # --------------------------------------------------------
    # Reset index after duplicate removal
    # --------------------------------------------------------

    df = df.reset_index(drop=True)

    # --------------------------------------------------------
    # Validate symptom score
    # --------------------------------------------------------

    df = validate_symptom_score(df)

    # --------------------------------------------------------
    # Validate label generation
    # --------------------------------------------------------

    df = validate_label_generation(df)

    # --------------------------------------------------------
    # Report missing values
    # --------------------------------------------------------

    report_missing_values(df)

    # --------------------------------------------------------
    # Report data types
    # --------------------------------------------------------

    report_data_types(df)

    # --------------------------------------------------------
    # Validate experiments
    # --------------------------------------------------------

    validate_experiment_features(df)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    df = final_validation(df)

    print("\n" + "=" * 70)
    print("DATA CLEANING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return df


# ============================================================
# 15. RUN DIRECTLY
# ============================================================

# if __name__ == "__main__":

    # cleaned_df = clean_dataset()

    # print("\nCleaning pipeline finished.")
    # print(
    #     f"Final shape: {cleaned_df.shape}"
    # )