"""
Smoke test for the Random Forest predictor.

Run from the ml folder:
    python -m src.test_random_forest_predictor

The sample answers are for software integration testing only, not a real patient.
Replace categories with values that exist in your training data if needed.
"""
from src.predict_random_forest import get_model_info, predict_antenatal_depression

sample_answers = {
    "Age": 26,
    "Gestational Age": 24,
    "Number of sons": 1,
    "Number of daughters": 1,
    "Total Number of Children": 2,
    "Gravida": "Multigravida",
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
    "Little interest or pleasure in doing things": 0,
    "Feeling down, depressed, or hopeless": 0,
    "Trouble falling or staying sleep or sleeping too much": 0,
    "Feeling tired or having little energy": 0,
    "Poor appetite or overeating": 0,
    "Feeling badabout yourself that you are failure or have let yourself or your family down": 0,
    "Trouble concentrating on things, such as reading the newspaper or watching television": 0,
    "Moving or speaking so slowly that other people could have Noticed.": 0,
    "Thoughts that you would be better off dead, or of hurting yourself": 0,
}

print("MODEL INFO")
print(get_model_info())
print("\nSAMPLE PREDICTION (software smoke test only)")
print(predict_antenatal_depression(sample_answers))
