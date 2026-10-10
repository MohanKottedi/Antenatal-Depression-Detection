import json
from typing import Any, Dict, List

from .feature_schema import (
    FEATURE_SCHEMA,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    SYMPTOM_FEATURES,
)


# ============================================================
# CATEGORY MAPPINGS
# ============================================================
#
# These are the exact categories learned during Transformer
# preprocessing.
#
# The LLM may output category names.
# Python converts them to the model IDs later.
# ============================================================

CATEGORY_VALUES = {
    "Gravida": [
        "Multigravida",
        "Multigravida+D1191",
        "Primigravida",
    ],

    "Female Education": [
        "Graduation",
        "Intermediate",
        "Matric",
        "Middle",
        "Primary",
        "Uneducated",
    ],

    "Husband Education": [
        "Graduation",
        "Intermediate",
        "Matric",
        "Middle",
        "Primary",
        "Uneducated",
    ],

    "Working Status": [
        "Housewife",
        "Working Lady",
    ],

    "Physical Health": [
        "Disable",
        "Healthy",
    ],

    "Previous Miscarriage": [
        "No",
        "Yes",
    ],

    "Sufficient Money for Basic Needs": [
        "No",
        "Yes",
        "__MISSING__",
    ],

    "Current Appereance Acceptance": [
        "No",
        "Yes",
    ],

    "Family System": [
        "Joint",
        "Nuclear",
    ],

    "Male Gender Preference": [
        "No",
        "Yes",
        "__MISSING__",
    ],

    "Relationship with Mother in-law": [
        "Good",
        "Moderate",
        "Poor",
    ],
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """
You are Sneha, a warm and supportive antenatal conversation
assistant.

Your job is NOT to diagnose the patient.

Your job is to understand the patient's natural language and
extract information that is explicitly supported by what the
patient said.

IMPORTANT RULES:

1. Never invent information.

2. Never guess a feature value.

3. If the patient has not provided enough information, do not
   create an update for that feature.

4. Preserve the patient's meaning while converting it to the
   allowed feature value.

5. For categorical features, use ONLY the allowed values
   provided in the feature schema.

6. For symptom features, use only:
      0 = Not at all
      1 = Several days
      2 = More than half the days
      3 = Nearly every day

7. Confidence must represent how strongly the patient's actual
   words support the extracted value.

8. Evidence must be a short explanation based only on the
   patient's message.

9. Do not diagnose depression.

10. Do not calculate the final depression prediction.

11. If information is ambiguous, leave the feature unchanged.

12. If the patient's response contains information about
    multiple features, extract all supported features.

13. The next question should sound natural and conversational,
    not like a questionnaire.

14. Never infer symptom frequency from symptom intensity.
    Words such as "very tired", "extremely sad", or
    "little energy" do not establish a frequency score.

15. For symptom scores, obtain evidence about frequency over
    the relevant assessment period before assigning a score.
    If frequency is unclear, omit that symptom update.

16. Do not use confidence as a substitute for evidence.
    A confident guess is still a guess.

17. If a symptom is mentioned but its frequency is unclear,
    ask a gentle follow-up question about how often it occurs.

18. A symptom update is valid only when the patient's message
    supports both the symptom and its frequency. If frequency
    is unclear, omit the update and ask about frequency.

Return ONLY valid JSON.
"""


# ============================================================
# BUILD FEATURE INFORMATION FOR THE LLM
# ============================================================

def build_feature_instructions() -> Dict[str, Any]:

    schema = {}

    for feature in NUMERIC_FEATURES:

        schema[feature] = {
            "type": "number"
        }

    for feature in CATEGORICAL_FEATURES:

        schema[feature] = {
            "type": "category",
            "allowed_values": CATEGORY_VALUES[feature]
        }

    for feature in SYMPTOM_FEATURES:

        schema[feature] = {
            "type": "symptom",
            "allowed_values": {
                "0": "Not at all",
                "1": "Several days",
                "2": "More than half the days",
                "3": "Nearly every day"
            }
        }

    return schema


# ============================================================
# BUILD USER PROMPT
# ============================================================

def build_extraction_prompt(
    patient_message: str,
    current_features: Dict[str, Any],
) -> str:

    feature_instructions = build_feature_instructions()

    return f"""
FEATURE SCHEMA:

{json.dumps(feature_instructions, indent=2)}


CURRENTLY KNOWN FEATURES:

{json.dumps(current_features, indent=2)}


PATIENT MESSAGE:

{patient_message}


TASK:

Extract only information supported by the patient's message.

Return exactly this structure:

{{
    "updates": [
        {{
            "feature": "feature name",
            "value": "value",
            "confidence": 0.0,
            "evidence": "short evidence"
        }}
    ],

    "next_question": "natural conversational question"
}}

If no feature can be extracted, return:

{{
    "updates": [],
    "next_question": "natural follow-up question"
}}
"""




def build_targeted_extraction_prompt(
    patient_message,
    current_features,
    pending_feature,
    pending_question,
    recent_history=None,
):
    schema = build_feature_instructions()

    history_text = "\n".join(recent_history or [])

    current_json = json.dumps(
        current_features,
        ensure_ascii=False,
        default=str,
    )

    return f"""
REQUIRED FEATURE SCHEMA
{schema}

CURRENT COLLECTED FEATURES
{current_json}

RECENT CONVERSATION
{history_text}

CURRENT TARGET FEATURE
{pending_feature}

QUESTION SNEHA ASKED
{pending_question}

LATEST PATIENT MESSAGE
{patient_message}

EXTRACTION RULES
1. Extract only an answer supported by the patient's message
   and the conversation context.
2. Only update the CURRENT TARGET FEATURE in this version.
3. Never assign a short answer to an unrelated feature.
4. Use the exact feature names and allowed categories in the schema.
5. Never invent a numeric value, category, or symptom score.
6. A symptom score requires a clear answer using the configured
   symptom scale and its verified assessment period.
7. If the answer is unclear, return an empty updates list.
8. Do not treat confidence as proof that an answer is correct.
9. Keep next_question focused on the current feature if clarification
   is needed.
10. Do not diagnose or make clinical risk decisions.

Return a JSON object with this structure:
{{
  "updates": [
    {{
      "feature": "exact feature name",
      "value": "validated candidate value",
      "confidence": 0.0,
      "evidence": "patient's supporting words"
    }}
  ],
  "next_question": "short clarification if required"
}}

Return JSON only.
"""



# ============================================================
# BASIC VALIDATION
# ============================================================

import math


def validate_update(update: Dict[str, Any]) -> bool:
    if not isinstance(update, dict):
        return False

    feature = update.get("feature")

    if feature not in FEATURE_SCHEMA:
        return False

    value = update.get("value")

    # Numeric features
    if feature in NUMERIC_FEATURES:
        if value is None or isinstance(value, bool):
            return False

        try:
            number = float(value)
        except (TypeError, ValueError):
            return False

        if not math.isfinite(number):
            return False

        # Basic input sanity checks, not clinical thresholds.
        if feature == "Age" and not 10 <= number <= 60:
            return False

        if feature == "Gestational Age" and not 0 <= number <= 45:
            return False

        if feature in {
            "Number of sons",
            "Number of daughters",
            "Total Number of Children",
        }:
            if number < 0 or not number.is_integer():
                return False

        return True

    # Categorical features
    if feature in CATEGORICAL_FEATURES:
        return (
            isinstance(value, str)
            and value in CATEGORY_VALUES[feature]
        )

    # Symptom scores
    if feature in SYMPTOM_FEATURES:
        if value is None or isinstance(value, bool):
            return False

        try:
            number = float(value)
        except (TypeError, ValueError):
            return False

        return (
            math.isfinite(number)
            and number.is_integer()
            and int(number) in [0, 1, 2, 3]
        )

    return False



import re


def has_explicit_symptom_frequency(
    evidence: str,
    patient_message: str,
) -> bool:
    """
    Conservative guard: accept a symptom score only when
    the patient's own message states a frequency category.
    This is a safeguard, not a clinical interpretation engine.
    """
    text = patient_message.lower().strip()

    frequency_patterns = [
        r"\bnot at all\b",
        r"\bseveral days\b",
        r"\bmore than half the days\b",
        r"\bnearly every day\b",
        r"\balmost every day\b",
        r"\bevery day\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in frequency_patterns
    )

# ============================================================
# VALIDATE LLM RESPONSE
# ============================================================
def validate_extraction(
    extraction: Dict[str, Any],
    patient_message: str = "",
) -> Dict[str, Any]:

    if not isinstance(extraction, dict):
        extraction = {}

    raw_updates = extraction.get("updates", [])

    if not isinstance(raw_updates, list):
        raw_updates = []

    valid_updates: List[Dict[str, Any]] = []

    for update in raw_updates:
        if not validate_update(update):
            continue

        feature = update["feature"]
        value = update["value"]
        # Do not accept symptom scores unless the patient's
        # original message explicitly states the frequency.
        if feature in SYMPTOM_FEATURES:
            evidence = update.get("evidence", "")

            if not has_explicit_symptom_frequency(
                evidence,
                patient_message,
            ):
                continue

        # Normalize numeric values.
        if feature in NUMERIC_FEATURES:
            value = float(value)
            if value.is_integer():
                value = int(value)

        # Normalize symptom scores to integers.
        elif feature in SYMPTOM_FEATURES:
            value = int(float(value))

        try:
            confidence = float(update.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if not math.isfinite(confidence):
            confidence = 0.0

        confidence = max(0.0, min(1.0, confidence))

        evidence = update.get("evidence", "")
        if not isinstance(evidence, str):
            evidence = ""

        valid_updates.append({
            "feature": feature,
            "value": value,
            "confidence": confidence,
            "evidence": evidence,
        })

    question = extraction.get("next_question", "")
    if not isinstance(question, str):
        question = ""

    return {
        "updates": valid_updates,
        "next_question": question,
    }
