from src.config import (
    RISK_FACTOR_COLUMNS,
    SYMPTOM_COLUMNS,
    FEATURES_EXPERIMENT_A,
)


# ============================================================
# FEATURE GROUPS
# ============================================================

NUMERIC_FEATURES = [
    "Age",
    "Gestational Age",
    "Number of sons",
    "Number of daughters",
    "Total Number of Children",
]


CATEGORICAL_FEATURES = [
    "Gravida",
    "Female Education",
    "Husband Education",
    "Working Status",
    "Physical Health",
    "Previous Miscarriage",
    "Sufficient Money for Basic Needs",
    "Current Appereance Acceptance",
    "Family System",
    "Male Gender Preference",
    "Relationship with Mother in-law",
]


SYMPTOM_FEATURES = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying sleep or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling badabout yourself that you are failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have Noticed.",
    "Thoughts that you would be better off dead, or of hurting yourself",
]


# ============================================================
# SYMPTOM SCALE
# ============================================================

SYMPTOM_SCALE = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day",
}


# ============================================================
# FEATURE SCHEMA
# ============================================================

FEATURE_SCHEMA = {}


for feature in NUMERIC_FEATURES:

    FEATURE_SCHEMA[feature] = {
        "group": "numeric",
        "type": "number",
        "required": True,
        "value": None,
        "confidence": None,
        "evidence": None,
    }


for feature in CATEGORICAL_FEATURES:

    FEATURE_SCHEMA[feature] = {
        "group": "categorical",
        "type": "category",
        "required": True,
        "value": None,
        "confidence": None,
        "evidence": None,
    }


for feature in SYMPTOM_FEATURES:

    FEATURE_SCHEMA[feature] = {
        "group": "symptom",
        "type": "ordinal",
        "required": True,
        "allowed_values": [0, 1, 2, 3],
        "scale": SYMPTOM_SCALE,
        "value": None,
        "confidence": None,
        "evidence": None,
    }


# ============================================================
# VALIDATION
# ============================================================

assert len(NUMERIC_FEATURES) == 5
assert len(CATEGORICAL_FEATURES) == 11
assert len(SYMPTOM_FEATURES) == 9

assert len(FEATURE_SCHEMA) == 25

assert set(FEATURE_SCHEMA.keys()) == set(
    FEATURES_EXPERIMENT_A
)