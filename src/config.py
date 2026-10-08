from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

DATASET_PATH = DATA_DIR / "pakdataset.csv"


# Create model directory automatically
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "Labelling"

TARGET_MAPPING = {
    "Depressed": 1,
    "Not": 0,
}


# ============================================================
# CLINICAL SYMPTOM FEATURES
# ============================================================

SYMPTOM_COLUMNS = [
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
# ANTENATAL / DEMOGRAPHIC / SOCIAL RISK FEATURES
# ============================================================

RISK_FACTOR_COLUMNS = [
    "Age",

    "Gestational Age",

    "Number of sons",

    "Number of daughters",

    "Total Number of Children",

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


# ============================================================
# FEATURES USED BY THE TWO EXPERIMENTS
# ============================================================

# Experiment A:
# Risk factors + symptom features
FEATURES_EXPERIMENT_A = (
    RISK_FACTOR_COLUMNS + SYMPTOM_COLUMNS
)


# Experiment B:
# Risk factors only
FEATURES_EXPERIMENT_B = (
    RISK_FACTOR_COLUMNS
)


# ============================================================
# COLUMNS THAT MUST NEVER BE MODEL INPUTS
# ============================================================

EXCLUDED_COLUMNS = [
    "Scalling",
    TARGET_COLUMN,
]


# ============================================================
# KNOWN DATA CLEANING RULES
# ============================================================

CATEGORICAL_CLEANUP_MAPS = {
    "Current Appereance Acceptance": {
        "YY": "Yes",
    }
}