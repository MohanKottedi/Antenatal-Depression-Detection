"""Pure helpers and reviewed general content for patient wellbeing features."""

from datetime import date, timedelta
import re
from typing import Any


MOOD_OPTIONS = (
    "happy",
    "calm",
    "neutral",
    "sad",
    "anxious",
    "stressed",
    "irritated",
    "overwhelmed",
)

DIETARY_PREFERENCES = (
    "no_preference",
    "vegetarian",
    "vegan",
    "non_vegetarian",
)

TRIMESTER_GUIDANCE = {
    1: (
        "Early development is underway. Symptoms and energy levels vary; discuss "
        "new or worrying symptoms with your maternity care team."
    ),
    2: (
        "Growth and movement continue to change during pregnancy. Your maternity "
        "team can explain which routine appointments and checks are recommended "
        "for you."
    ),
    3: (
        "The baby continues to grow as you prepare for birth. Ask your maternity "
        "team about your individual birth and urgent-care plan."
    ),
}

PREGNANCY_REFERENCES = [
    {
        "label": "NHS: pregnancy week by week",
        "url": "https://www.nhs.uk/pregnancy/week-by-week/",
    },
    {
        "label": "ACOG: methods for estimating the due date",
        "url": (
            "https://www.acog.org/clinical/clinical-guidance/committee-opinion/"
            "articles/2017/05/methods-for-estimating-the-due-date"
        ),
    },
]

NUTRITION_REFERENCES = [
    {
        "label": "NHS: healthy diet in pregnancy",
        "url": "https://www.nhs.uk/pregnancy/keeping-well/have-a-healthy-diet/",
    },
    {
        "label": "NHS: foods to avoid in pregnancy",
        "url": "https://www.nhs.uk/pregnancy/keeping-well/foods-to-avoid/",
    },
]

NUTRITION_GUIDANCE = (
    "Aim for varied meals from foods that fit your needs and local guidance. "
    "Wash produce, keep raw and ready-to-eat foods separate, and follow safe "
    "cooking, refrigeration, and food-handling advice. Food recommendations "
    "cannot account for every allergy or medical condition; check ingredients "
    "and discuss individual needs with your maternity clinician or a registered "
    "dietitian. Do not start or stop supplements based on this page."
)

MEAL_IDEAS = {
    "no_preference": {
        "breakfast": "Porridge with fruit and pasteurized milk or fortified alternative.",
        "lunch": "Lentil and vegetable soup with whole-grain bread.",
        "dinner": "Thoroughly cooked beans or fish, vegetables, and rice or potatoes.",
        "snack": "Washed fruit with pasteurized yogurt, if suitable for you.",
    },
    "vegetarian": {
        "breakfast": "Porridge with fruit and pasteurized milk or fortified alternative.",
        "lunch": "Chickpea and vegetable bowl with whole grains.",
        "dinner": "Lentil curry with thoroughly cooked vegetables and rice.",
        "snack": "Washed fruit with pasteurized yogurt, if suitable for you.",
    },
    "vegan": {
        "breakfast": "Oats with fruit and a calcium-fortified plant drink.",
        "lunch": "Chickpea and vegetable bowl with whole grains.",
        "dinner": "Bean and vegetable stew with rice or potatoes.",
        "snack": "Washed fruit with nuts or seeds, if suitable for you.",
    },
    "non_vegetarian": {
        "breakfast": "Porridge with fruit and pasteurized milk or fortified alternative.",
        "lunch": "Thoroughly cooked chicken or beans with vegetables and whole grains.",
        "dinner": "Thoroughly cooked fish or lentils with vegetables and potatoes.",
        "snack": "Washed fruit with pasteurized yogurt, if suitable for you.",
    },
}

MEDITATION_ACTIVITIES = {
    "gentle-breathing": {
        "title": "Gentle breathing",
        "description": (
            "Sit comfortably and breathe normally, noticing each gentle breath. "
            "Do not hold your breath or force a deep breath."
        ),
        "durations": (120, 300),
    },
    "grounding": {
        "title": "Five-senses grounding",
        "description": (
            "At a comfortable pace, notice things you can see, feel, hear, "
            "smell, and taste. Skip anything that feels uncomfortable."
        ),
        "durations": (120, 300),
    },
    "body-relaxation": {
        "title": "Seated relaxation",
        "description": (
            "Sit supported and gently notice areas that feel comfortable. "
            "Release tension only if it feels easy; avoid positions or movements "
            "that cause pain, dizziness, or breathlessness."
        ),
        "durations": (300, 600),
    },
}

URGENT_SYMPTOM_PATTERNS = (
    r"\bheavy bleeding\b",
    r"\bheavy vaginal bleeding\b",
    r"\bsevere (?:headache|stomach pain|abdominal pain|belly pain)\b",
    r"\bchest pain\b",
    r"\b(?:difficulty|trouble) breathing\b",
    r"\bshortness of breath\b",
    r"\bseizure\b",
    r"\bfainted\b|\bfainting\b|\blost consciousness\b",
    r"\bblurred vision\b|\bvision changes\b",
    r"\bwaters? broke\b|\bfluid leaking\b",
    r"\b(?:baby|baby's) movements? (?:stopped|reduced|slowed)\b",
)

URGENT_ASSISTANT_RESPONSE = (
    "I can't assess symptoms or decide how urgent they are. Some symptoms in "
    "pregnancy need prompt professional assessment. Please contact your local "
    "maternity unit, midwife, or clinician now for advice. If you feel in "
    "immediate danger, have severe symptoms, or cannot reach them, call your "
    "local emergency services or go to the nearest emergency department. "
    "Do not wait for an AI response."
)


def pregnancy_snapshot(
    lmp_date: date | None,
    confirmed_due_date: date | None,
    today: date | None = None,
) -> dict[str, Any]:
    today = today or date.today()
    calculated_due_date = lmp_date + timedelta(days=280) if lmp_date else None
    reference_due_date = confirmed_due_date or calculated_due_date
    gestational_start = lmp_date
    dating_source = "last_menstrual_period" if lmp_date else None
    if gestational_start is None and confirmed_due_date is not None:
        gestational_start = confirmed_due_date - timedelta(days=280)
        dating_source = "clinically_confirmed_due_date"

    elapsed_days = (today - gestational_start).days if gestational_start else None
    if elapsed_days is not None and elapsed_days >= 0:
        gestational_week, gestational_day = divmod(elapsed_days, 7)
        trimester = (
            1 if gestational_week < 14 else
            2 if gestational_week < 28 else
            3
        )
        progress = min(100.0, round(elapsed_days / 280 * 100, 1))
    else:
        gestational_week = None
        gestational_day = None
        trimester = None
        progress = None

    return {
        "lmp_date": lmp_date.isoformat() if lmp_date else None,
        "estimated_due_date": (
            calculated_due_date.isoformat() if calculated_due_date else None
        ),
        "confirmed_due_date": (
            confirmed_due_date.isoformat() if confirmed_due_date else None
        ),
        "due_date": reference_due_date.isoformat() if reference_due_date else None,
        "gestational_week": gestational_week,
        "gestational_day": gestational_day,
        "trimester": trimester,
        "progress_percentage": progress,
        "dating_source": dating_source,
        "milestone": TRIMESTER_GUIDANCE.get(trimester) if trimester else None,
        "upcoming_milestones": [
            "Follow the appointment schedule given by your maternity team.",
            "Ask your clinician what changes or symptoms should prompt a call.",
            "Review your local birth and urgent-care plan with your care team.",
        ] if trimester else [],
        "references": PREGNANCY_REFERENCES,
        "notice": (
            "Pregnancy dates are estimates. A clinician may adjust dating based "
            "on your history and ultrasound."
        ),
    }


def urgent_symptoms(message: str) -> bool:
    lowered = message.lower()
    return any(re.search(pattern, lowered) for pattern in URGENT_SYMPTOM_PATTERNS)
