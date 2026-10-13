import json
from typing import Any, Dict, List, Optional

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

8. Evidence must be an exact, short quote from the patient's
   latest message that supports the extracted value.

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

19. If there is no directly supported update but exactly one
    allowed value is strongly supported by the latest message,
    return it as suggested_update with confidence from 0.0 to 1.0.
    It will be confirmed by the patient and must not be accepted yet.
20. Never randomly choose or guess an option based on population
    likelihood, prior answers, or confidence alone. A suggestion
    requires an exact evidence quote from the latest message.
    Never suggest an unstated symptom frequency.

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
2. Update only the CURRENT TARGET FEATURE. Ignore every other
   feature, even if it is mentioned in the message.
3. Never assign a short answer to an unrelated feature.
4. Use the exact feature names and allowed categories in the schema.
5. Never invent a numeric value, category, or symptom score.
   Evidence must be copied as an exact, short quote from the latest
   patient message, not paraphrased or inferred.
6. A symptom score requires a clear answer using the configured
   symptom scale and its verified assessment period.
7. If the answer is unclear, return an empty updates list.
8. Do not treat confidence as proof that an answer is correct.
9. Keep next_question focused on the current feature if clarification
   is needed.
10. Do not diagnose or make clinical risk decisions.
11. If exactly one allowed value is strongly supported by the latest
    message but is not clear enough for an update, return it as
    suggested_update. Sneha will ask the patient to confirm it.
12. Never guess or randomly select a value. Suggestions require exact
    evidence from the latest message and must meet the same feature
    validation rules as updates. Do not suggest unstated symptom
    frequencies.
13. For Gestational Age, distinguish weeks from days. When the patient
    explicitly gives days, convert days to weeks by dividing by 7;
    when they give both, add weeks and days/7. Never treat a number
    followed by "days" as that many weeks.

Return a JSON object with this structure:
{{
  "updates": [
    {{
      "feature": "exact feature name",
      "value": "validated candidate value",
      "confidence": 0.0,
      "evidence": "exact short quote from the latest patient message"
    }}
  ],
  "suggested_update": {{
    "feature": "exact feature name",
    "value": "one allowed candidate value",
    "confidence": 0.0,
    "evidence": "exact short quote from the latest patient message"
  }},
  "next_question": "short clarification if required"
}}

Use null for suggested_update if there is no single evidence-supported
candidate.
Return JSON only.
"""



# ============================================================
# BASIC VALIDATION
# ============================================================

import math
import re


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



def has_grounded_evidence(evidence: str, patient_message: str) -> bool:
    """Require extraction evidence to be a quote from the latest answer."""
    if not isinstance(evidence, str) or not evidence.strip():
        return False

    def normalize(text: str) -> str:
        return re.sub(r"\W+", " ", text.casefold()).strip()

    normalized_evidence = normalize(evidence)
    normalized_message = normalize(patient_message)
    return bool(normalized_evidence) and normalized_evidence in normalized_message


def parse_gestational_age_weeks(patient_message: str) -> Optional[float]:
    """Convert an explicit weeks/days duration to decimal weeks."""
    number = r"(?P<value>\d+(?:\.\d+)?)"
    week_matches = list(re.finditer(
        rf"{number}\s*(?:weeks?|wks?|wk)\b",
        patient_message,
        flags=re.IGNORECASE,
    ))
    day_matches = list(re.finditer(
        rf"{number}\s*(?:days?|d)\b",
        patient_message,
        flags=re.IGNORECASE,
    ))

    if not week_matches and not day_matches:
        return None

    if len(week_matches) > 1 or len(day_matches) > 1:
        return None

    weeks = float(week_matches[0].group("value")) if week_matches else 0.0
    days = float(day_matches[0].group("value")) if day_matches else 0.0
    return weeks + days / 7.0


def has_numeric_value_evidence(value: Any, patient_message: str) -> bool:
    """Require numeric extractions to be present in the patient's answer."""
    number = float(value)
    numeric_forms = [str(int(number)) if number.is_integer() else str(number)]

    small_numbers = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
        14: "fourteen", 15: "fifteen", 16: "sixteen",
        17: "seventeen", 18: "eighteen", 19: "nineteen",
    }
    tens = {
        20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
        60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
    }
    integer = int(number)
    if number.is_integer() and integer in small_numbers:
        numeric_forms.append(small_numbers[integer])
    elif number.is_integer() and integer in tens:
        numeric_forms.append(tens[integer])
    elif number.is_integer() and 20 < integer < 100:
        tens_word = tens[(integer // 10) * 10]
        ones_word = small_numbers[integer % 10]
        numeric_forms.extend((
            f"{tens_word}-{ones_word}",
            f"{tens_word} {ones_word}",
        ))

    text = patient_message.casefold()
    return any(
        re.search(
            rf"(?<![\w.]){re.escape(form)}(?![\w.])",
            text,
        )
        for form in numeric_forms
    )


def has_explicit_symptom_frequency(
    evidence: str,
    patient_message: str,
    score: int,
) -> bool:
    """
    Conservative guard: accept a symptom score only when
    the patient's own message states a frequency category.
    This is a safeguard, not a clinical interpretation engine.
    """
    text = patient_message.lower().strip()

    frequency_patterns = {
        0: [r"\bnot at all\b"],
        1: [r"\bseveral days\b"],
        2: [r"\bmore than half the days\b"],
        3: [
            r"\bnearly every day\b",
            r"\balmost every day\b",
            r"\bevery day\b",
        ],
    }
    observed_scores = {
        candidate
        for candidate, patterns in frequency_patterns.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }
    return len(observed_scores) == 1 and score in observed_scores

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
        evidence = update.get("evidence", "")
        if not has_grounded_evidence(evidence, patient_message):
            continue
        converted_gestational_age = (
            parse_gestational_age_weeks(patient_message)
            if feature == "Gestational Age"
            else None
        )
        if converted_gestational_age is not None:
            value = converted_gestational_age
        if (
            feature in NUMERIC_FEATURES
            and converted_gestational_age is None
            and not has_numeric_value_evidence(value, patient_message)
        ):
            continue

        # Do not accept symptom scores unless the patient's
        # original message explicitly states the frequency.
        if feature in SYMPTOM_FEATURES:
            if not has_explicit_symptom_frequency(
                evidence,
                patient_message,
                int(value),
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

        valid_updates.append({
            "feature": feature,
            "value": value,
            "confidence": confidence,
            "evidence": evidence,
        })

    suggestion = extraction.get("suggested_update")
    validated_suggestion = None
    if isinstance(suggestion, dict):
        suggestion_updates = validate_extraction(
            {"updates": [suggestion]},
            patient_message=patient_message,
        )["updates"]
        if len(suggestion_updates) == 1:
            validated_suggestion = suggestion_updates[0]

    question = extraction.get("next_question", "")
    if not isinstance(question, str):
        question = ""

    return {
        "updates": valid_updates,
        "suggested_update": validated_suggestion,
        "next_question": question,
    }
