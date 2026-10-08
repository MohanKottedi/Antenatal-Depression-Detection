from pathlib import Path
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "tabular_transformer_experiment_A.pt"


# ============================================================
# MODEL DEFINITION
# Must exactly match train_tabular_transformer_a.py
# ============================================================

class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_numeric_features,
        num_symptom_features,
        category_cardinalities,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        ffn_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # CLS token
        # ----------------------------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ----------------------------------------------------
        # Numeric feature embeddings
        # Each numeric feature becomes one token
        # ----------------------------------------------------
        self.numeric_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for _ in range(num_numeric_features)
        ])

        # ----------------------------------------------------
        # Categorical embeddings
        # ----------------------------------------------------
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim)
            for cardinality in category_cardinalities
        ])

        # ----------------------------------------------------
        # Symptom embeddings
        # Each symptom becomes one numerical token
        # ----------------------------------------------------
        self.symptom_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for _ in range(num_symptom_features)
        ])

        # ----------------------------------------------------
        # Total tokens:
        # 1 CLS + numeric + categorical + symptoms
        # ----------------------------------------------------
        total_tokens = (
            1
            + num_numeric_features
            + len(category_cardinalities)
            + num_symptom_features
        )

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, total_tokens, embed_dim)
        )

        # ----------------------------------------------------
        # Transformer encoder
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(
        self,
        numeric_features,
        categorical_features,
        symptom_features,
    ):

        batch_size = numeric_features.size(0)

        tokens = []

        # ----------------------------------------------------
        # CLS
        # ----------------------------------------------------
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens.append(cls)

        # ----------------------------------------------------
        # Numeric tokens
        # ----------------------------------------------------
        for i, embedding in enumerate(self.numeric_embeddings):

            x = numeric_features[:, i:i + 1]

            x = embedding(x)

            x = x.unsqueeze(1)

            tokens.append(x)

        # ----------------------------------------------------
        # Categorical tokens
        # ----------------------------------------------------
        for i, embedding in enumerate(self.categorical_embeddings):

            x = categorical_features[:, i]

            x = embedding(x)

            x = x.unsqueeze(1)

            tokens.append(x)

        # ----------------------------------------------------
        # Symptom tokens
        # ----------------------------------------------------
        for i, embedding in enumerate(self.symptom_embeddings):

            x = symptom_features[:, i:i + 1]

            x = embedding(x)

            x = x.unsqueeze(1)

            tokens.append(x)

        # ----------------------------------------------------
        # Combine tokens
        # ----------------------------------------------------
        x = torch.cat(tokens, dim=1)

        # ----------------------------------------------------
        # Positional embeddings
        # ----------------------------------------------------
        x = x + self.position_embeddings[:, :x.size(1), :]

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------
        x = self.transformer(x)

        # ----------------------------------------------------
        # CLS representation
        # ----------------------------------------------------
        cls_output = x[:, 0]

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------
        logits = self.classifier(cls_output)

        return logits.squeeze(-1)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)
DATA_PATH = BASE_DIR / "models" / "transformer_data_experiment_A.joblib"

import joblib

transformer_data = joblib.load(DATA_PATH)

numeric_imputer = transformer_data["numeric_imputer"]
numeric_scaler = transformer_data["numeric_scaler"]

symptom_imputer = transformer_data["symptom_imputer"]
symptom_scaler = transformer_data["symptom_scaler"]

MODEL_CONFIG = checkpoint["model_config"]

NUMERIC_COLUMNS = checkpoint["numeric_columns"]
CATEGORICAL_COLUMNS = checkpoint["categorical_columns"]
SYMPTOM_COLUMNS = checkpoint["symptom_columns"]
CATEGORY_MAPPINGS = checkpoint["category_mappings"]


# ============================================================
# CREATE MODEL
# ============================================================

model = TabularTransformer(
    num_numeric_features=MODEL_CONFIG["num_numeric_features"],
    num_symptom_features=MODEL_CONFIG["num_symptom_features"],
    category_cardinalities=MODEL_CONFIG["category_cardinalities"],
    embed_dim=MODEL_CONFIG["embed_dim"],
    num_heads=MODEL_CONFIG["num_heads"],
    num_layers=MODEL_CONFIG["num_layers"],
    ffn_dim=MODEL_CONFIG["ffn_dim"],
    dropout=MODEL_CONFIG["dropout"],
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _get_numeric_value(answers, column):
    """
    Get numeric feature from answers.
    Raises a clear error if the feature is missing.
    """

    if column not in answers:
        raise ValueError(
            f"Missing required numeric feature: '{column}'"
        )

    value = answers[column]

    if value is None or value == "":
        raise ValueError(
            f"Numeric feature '{column}' cannot be empty."
        )

    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid numeric value for '{column}': {value}"
        )


def _get_categorical_value(answers, column):
    """
    Convert categorical answer to the integer ID
    used during Transformer training.
    """

    if column not in answers:
        raise ValueError(
            f"Missing required categorical feature: '{column}'"
        )

    value = answers[column]

    if value is None or value == "":
        raise ValueError(
            f"Categorical feature '{column}' cannot be empty."
        )

    value = str(value).strip()

    mapping = CATEGORY_MAPPINGS[column]

    # Exact match
    if value in mapping:
        return mapping[value]

    # Case-insensitive match
    normalized = value.lower()

    for category, index in mapping.items():

        if category.lower() == normalized:
            return index

    raise ValueError(
        f"Unknown value '{value}' for '{column}'. "
        f"Expected one of: {list(mapping.keys())}"
    )


def _get_symptom_value(answers, column):
    """
    Convert symptom answer into numeric value.

    The training pipeline treats each symptom as an individual
    numerical feature. Scalling is NOT used.
    """

    if column not in answers:
        raise ValueError(
            f"Missing required symptom feature: '{column}'"
        )

    value = answers[column]

    if value is None or value == "":
        raise ValueError(
            f"Symptom feature '{column}' cannot be empty."
        )

    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid symptom value for '{column}': {value}"
        )

    return value


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

@torch.no_grad()
def predict_antenatal_depression(answers):
    """
    Predict antenatal depression using Transformer A.

    Parameters
    ----------
    answers : dict
        Dictionary containing the 25 Experiment-A features.

    Returns
    -------
    dict
        Prediction, probability and model information.
    """

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    numeric_values = [
        _get_numeric_value(answers, column)
        for column in NUMERIC_COLUMNS
    ]

    numeric_df = pd.DataFrame(
        [numeric_values],
        columns=NUMERIC_COLUMNS
    )

    numeric_array = numeric_imputer.transform(
        numeric_df
    )

    numeric_array = numeric_scaler.transform(
        numeric_array
    )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_values = [
        _get_categorical_value(answers, column)
        for column in CATEGORICAL_COLUMNS
    ]

    # --------------------------------------------------------
    # Symptom features
    # --------------------------------------------------------

    symptom_values = [
        _get_symptom_value(answers, column)
        for column in SYMPTOM_COLUMNS
    ]

    symptom_df = pd.DataFrame(
        [symptom_values],
        columns=SYMPTOM_COLUMNS
    )

    symptom_array = symptom_imputer.transform(
        symptom_df
    )

    symptom_array = symptom_scaler.transform(
        symptom_array
    )

    # --------------------------------------------------------
    # Create tensors
    # --------------------------------------------------------

    numeric_tensor = torch.tensor(
        numeric_array,
        dtype=torch.float32,
        device=DEVICE,
    )

    categorical_tensor = torch.tensor(
        [categorical_values],
        dtype=torch.long,
        device=DEVICE,
    )

    symptom_tensor = torch.tensor(
        symptom_array,
        dtype=torch.float32,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    logits = model(
        numeric_tensor,
        categorical_tensor,
        symptom_tensor,
    )

    probability = torch.sigmoid(logits).item()

    prediction = int(probability >= 0.5)

    label = (
        "Depressed"
        if prediction == 1
        else "Not Depressed"
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "prediction": prediction,
        "label": label,
        "probability": round(probability, 6),
        "device": str(DEVICE),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info():
    """
    Return information about the loaded Transformer.
    """

    return {
        "model": "Tabular Transformer",
        "experiment": "A",
        "device": str(DEVICE),
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "symptom_features": SYMPTOM_COLUMNS,
        "num_features": (
            len(NUMERIC_COLUMNS)
            + len(CATEGORICAL_COLUMNS)
            + len(SYMPTOM_COLUMNS)
        ),
        "best_epoch": checkpoint.get("best_epoch"),
        "validation_auc": checkpoint.get("best_validation_auc"),
        "test_metrics": checkpoint.get("test_metrics"),
    }


# ============================================================
# DIRECT TEST
# ============================================================
# if __name__ == "__main__":

#     print("=" * 70)
#     print("SNEHA ANTENATAL DETECTION")
#     print("TRANSFORMER A PREDICTION TEST")
#     print("=" * 70)

#     # ========================================================
#     # SAMPLE 1: LOW / NO SYMPTOMS
#     # ========================================================

#     not_depressed_sample = {
#         "Age": 25,
#         "Gestational Age": 28,
#         "Number of sons": 1,
#         "Number of daughters": 0,
#         "Total Number of Children": 1,

#         "Gravida": "Primigravida",
#         "Female Education": "Graduation",
#         "Husband Education": "Graduation",
#         "Working Status": "Housewife",
#         "Physical Health": "Healthy",
#         "Previous Miscarriage": "No",
#         "Sufficient Money for Basic Needs": "Yes",
#         "Current Appereance Acceptance": "Yes",
#         "Family System": "Nuclear",
#         "Male Gender Preference": "No",
#         "Relationship with Mother in-law": "Good",
#     }

#     for column in SYMPTOM_COLUMNS:
#         not_depressed_sample[column] = 0


#     # ========================================================
#     # SAMPLE 2: HIGH SYMPTOMS
#     # ========================================================

#     depressed_sample = {
#         "Age": 25,
#         "Gestational Age": 28,
#         "Number of sons": 1,
#         "Number of daughters": 0,
#         "Total Number of Children": 1,

#         "Gravida": "Primigravida",
#         "Female Education": "Graduation",
#         "Husband Education": "Graduation",
#         "Working Status": "Housewife",
#         "Physical Health": "Healthy",
#         "Previous Miscarriage": "No",
#         "Sufficient Money for Basic Needs": "Yes",
#         "Current Appereance Acceptance": "Yes",
#         "Family System": "Nuclear",
#         "Male Gender Preference": "No",
#         "Relationship with Mother in-law": "Good",
#     }

#     for column in SYMPTOM_COLUMNS:
#         depressed_sample[column] = 3


#     # ========================================================
#     # INITIALIZE RESULTS
#     # This prevents Pylance possibly-unbound warnings.
#     # ========================================================

#     result_not = None
#     result_dep = None


#     # ========================================================
#     # PREDICT SAMPLE 1
#     # ========================================================

#     print("\n" + "=" * 70)
#     print("SAMPLE 1 - LOW / NO SYMPTOMS")
#     print("=" * 70)

#     try:

#         result_not = predict_antenatal_depression(
#             not_depressed_sample
#         )

#         print(f"Prediction : {result_not['prediction']}")
#         print(f"Label      : {result_not['label']}")
#         print(f"Probability: {result_not['probability']}")
#         print(f"Device     : {result_not['device']}")

#     except Exception as e:

#         print("Prediction failed for Sample 1.")
#         print(f"Error: {e}")


#     # ========================================================
#     # PREDICT SAMPLE 2
#     # ========================================================

#     print("\n" + "=" * 70)
#     print("SAMPLE 2 - HIGH SYMPTOMS")
#     print("=" * 70)

#     try:

#         result_dep = predict_antenatal_depression(
#             depressed_sample
#         )

#         print(f"Prediction : {result_dep['prediction']}")
#         print(f"Label      : {result_dep['label']}")
#         print(f"Probability: {result_dep['probability']}")
#         print(f"Device     : {result_dep['device']}")

#     except Exception as e:

#         print("Prediction failed for Sample 2.")
#         print(f"Error: {e}")


#     # ========================================================
#     # COMPARISON
#     # ========================================================

#     if result_not is not None and result_dep is not None:

#         print("\n" + "=" * 70)
#         print("SAMPLE COMPARISON")
#         print("=" * 70)

#         print(
#             f"Low symptoms  → "
#             f"{result_not['label']} "
#             f"({result_not['probability']:.6f})"
#         )

#         print(
#             f"High symptoms → "
#             f"{result_dep['label']} "
#             f"({result_dep['probability']:.6f})"
#         )

#         print("=" * 70)
if __name__ == "__main__":

    print("=" * 80)
    print("SNEHA ANTENATAL DETECTION")
    print("TRANSFORMER A - SYMPTOM SCORE BOUNDARY TEST")
    print("=" * 80)

    base_sample = {
        "Age": 25,
        "Gestational Age": 28,
        "Number of sons": 1,
        "Number of daughters": 0,
        "Total Number of Children": 1,
        "Gravida": "Primigravida",
        "Female Education": "Graduation",
        "Husband Education": "Graduation",
        "Working Status": "Housewife",
        "Physical Health": "Healthy",
        "Previous Miscarriage": "No",
        "Sufficient Money for Basic Needs": "Yes",
        "Current Appereance Acceptance": "Yes",
        "Family System": "Nuclear",
        "Male Gender Preference": "No",
        "Relationship with Mother in-law": "Good",
    }

    # Test different total symptom scores
    test_scores = [
        5,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        15,
        18,
        21,
        24,
    ]

    print("\n" + "-" * 80)
    print(f"{'Score':<10}{'Prediction':<15}{'Label':<20}{'Probability':<15}")
    print("-" * 80)

    for score in test_scores:

        sample = base_sample.copy()

        # Start every symptom at 1
        for column in SYMPTOM_COLUMNS:
            sample[column] = 1

        # We need to distribute the additional score
        # across the 9 symptoms.
        remaining = score - 9

        for column in SYMPTOM_COLUMNS:
            if remaining <= 0:
                break

            addition = min(3, remaining)
            sample[column] += addition
            remaining -= addition

        result = predict_antenatal_depression(sample)

        print(
            f"{score:<10}"
            f"{result['prediction']:<15}"
            f"{result['label']:<20}"
            f"{result['probability']:<15.6f}"
        )

    print("-" * 80)

    print("\nTest complete.")