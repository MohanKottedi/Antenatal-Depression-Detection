# ml/src/sneha/conversation_manager.py

from typing import Any, Dict, List, Optional
from copy import deepcopy

from .feature_schema import (
    FEATURE_SCHEMA,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    SYMPTOM_FEATURES,
)

from .feature_extractor import validate_extraction

# ============================================================
# CONVERSATION STAGES
# ============================================================

STAGES = [
    "introduction",
    "background",
    "social",
    "emotional",
    "symptoms",
    "safety",
    "completed",
]


QUESTION_TEMPLATES = {
    "Age": "May I ask how old you are?",
    "Gestational Age": "How many weeks pregnant are you?",
    "Number of sons": "How many sons do you have?",
    "Number of daughters": "How many daughters do you have?",
    "Total Number of Children": "How many children do you have in total?",
    "Gravida": "Is this your first pregnancy, or have you been pregnant before?",
    "Female Education": "What is the highest level of education you have completed?",
    "Husband Education": "What is your husband's highest level of education?",
    "Working Status": "Are you currently working, or are you mainly managing things at home?",
    "Physical Health": "How would you describe your physical health?",
    "Previous Miscarriage": "Have you experienced a miscarriage in the past?",
    "Sufficient Money for Basic Needs": "Do you generally have enough money for your basic needs?",
    "Current Appereance Acceptance": "How do you feel about your appearance these days?",
    "Family System": "Do you live in a joint family or a nuclear family?",
    "Male Gender Preference": "Is there any pressure in your family to have a boy?",
    "Relationship with Mother in-law": "How would you describe your relationship with your mother-in-law?",
    "Little interest or pleasure in doing things": "How often have you had little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless": "How often have you felt down, depressed, or hopeless?",
    "Trouble falling or staying sleep or sleeping too much": "How often have you had trouble sleeping or slept more than usual?",
    "Feeling tired or having little energy": "How often have you felt tired or had little energy?",
    "Poor appetite or overeating": "How often have you had a poor appetite or eaten more than usual?",
    "Feeling badabout yourself that you are failure or have let yourself or your family down": "How often have you felt bad about yourself or felt you had let yourself or your family down?",
    "Trouble concentrating on things, such as reading the newspaper or watching television": "How often have you had trouble concentrating on things?",
    "Moving or speaking so slowly that other people could have Noticed.": "How often have you moved or spoken so slowly that other people might have noticed?",
    "Thoughts that you would be better off dead, or of hurting yourself": "Sometimes people have thoughts about not wanting to be alive or about hurting themselves. Have you had any such thoughts?",
}


# ============================================================
# CONVERSATION MANAGER
# ============================================================

class ConversationManager:

    def __init__(self):

        self.features: Dict[str, Any] = {}

        self.feature_metadata: Dict[str, Dict[str, Any]] = {}

        self.conversation_history: List[Dict[str, str]] = []

        self.stage = "introduction"

        self.started = False

        self.completed = False
        self.pending_feature: Optional[str] = None
        self.pending_question: Optional[str] = None

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.features = {}

        self.feature_metadata = {}

        self.conversation_history = []

        self.stage = "introduction"

        self.started = False

        self.completed = False
        self.pending_feature = None
        self.pending_question = None

    # ========================================================
    # HISTORY
    # ========================================================

    def add_message(
        self,
        role: str,
        content: str,
    ):

        self.conversation_history.append({
            "role": role,
            "content": content,
        })

    # ========================================================
    # UPDATE FEATURES
    # ========================================================

        
    def update_features(self, extraction, patient_message=""):
        validated = validate_extraction(
            extraction,
            patient_message=patient_message,
        )

        accepted = []

        child_features = {
            "Number of sons",
            "Number of daughters",
            "Total Number of Children",
        }

        for item in validated["updates"]:
            feature = item["feature"]
            value = item["value"]

            # Check consistency whenever all three counts are known.
            candidate = dict(self.features)
            candidate[feature] = value

            if child_features.issubset(candidate):
                sons = candidate["Number of sons"]
                daughters = candidate["Number of daughters"]
                total = candidate["Total Number of Children"]

                if sons + daughters != total:
                    if feature in child_features:
                        continue

            existing = self.feature_metadata.get(feature, {})
            confidence = item["confidence"]

            if (
                existing
                and existing.get("confidence", 0) > confidence
            ):
                continue

            self.features[feature] = value
            self.feature_metadata[feature] = {
                "confidence": confidence,
                "evidence": item["evidence"],
            }

            accepted.append(item)

        validated["updates"] = accepted
        return validated

    # ========================================================
    # MISSING FEATURES
    # ========================================================

    def get_missing_features(self) -> List[str]:

        return [
            feature
            for feature in FEATURE_SCHEMA
            if feature not in self.features
        ]

    # ========================================================
    # COLLECTED FEATURES
    # ========================================================

    def get_collected_features(self) -> Dict[str, Any]:

        return deepcopy(
            self.features
        )

    # ========================================================
    # FEATURE COMPLETION
    # ========================================================

    def is_complete(self) -> bool:

        return len(
            self.get_missing_features()
        ) == 0

    # ========================================================
    # PROGRESS
    # ========================================================

    def get_progress(self) -> Dict[str, Any]:

        total = len(FEATURE_SCHEMA)

        collected = len(self.features)

        return {
            "collected": collected,
            "total": total,
            "remaining": total - collected,
            "percentage": round(
                (collected / total) * 100,
                2
            ) if total else 100.0,
        }

    # ========================================================
    # STAGE MANAGEMENT
    # ========================================================

    def determine_stage(self):

        missing = set(
            self.get_missing_features()
        )

        # ----------------------------------------------------
        # Safety has priority.
        # ----------------------------------------------------

        suicidal_feature = (
            "Thoughts that you would be better off dead, "
            "or of hurting yourself"
        )

    

        # ----------------------------------------------------
        # Background
        # ----------------------------------------------------

        background_features = {
            "Age",
            "Gestational Age",
            "Number of sons",
            "Number of daughters",
            "Total Number of Children",
            "Gravida",
        }

        if missing.intersection(
            background_features
        ):

            self.stage = "background"

            return self.stage

        # ----------------------------------------------------
        # Social
        # ----------------------------------------------------

        social_features = {
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
        }

        if missing.intersection(
            social_features
        ):

            self.stage = "social"

            return self.stage

        # ----------------------------------------------------
        # Symptoms
        # ----------------------------------------------------

        if missing.intersection(
            set(SYMPTOM_FEATURES)
        ):

            self.stage = "symptoms"

            return self.stage

        # ----------------------------------------------------
        # Completed
        # ----------------------------------------------------

        self.stage = "completed"

        self.completed = True

        return self.stage

    # ========================================================
    # NEXT FEATURE
    # ========================================================

    
    def prepare_next_question(self):
        feature = self.get_next_feature()

        if feature is None:
            self.completed = True
            self.clear_pending_feature()
            return None

        question = QUESTION_TEMPLATES.get(feature)

        if question is None:
            raise ValueError(
                f"No question template defined for feature: {feature}"
            )

        self.set_pending_feature(feature, question)
        return question


    def get_next_feature(
        self,
    ) -> Optional[str]:

        self.determine_stage()

        missing = self.get_missing_features()

        if not missing:

            return None

        # ----------------------------------------------------
        # Prefer feature relevant to current stage.
        # ----------------------------------------------------

        stage_groups = {

            "background": [
                "Age",
                "Gestational Age",
                "Number of sons",
                "Number of daughters",
                "Total Number of Children",
                "Gravida",
            ],

            "social": [
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
            ],

            "symptoms": SYMPTOM_FEATURES,

            "safety": [
                "Thoughts that you would be better off dead, "
                "or of hurting yourself"
            ],
        }

        preferred = stage_groups.get(
            self.stage,
            []
        )

        for feature in preferred:

            if feature in missing:

                return feature

        # Fallback

        return missing[0]

    # ========================================================
    # PROCESS PATIENT MESSAGE
    # ========================================================

        
    def process_extraction(
        self,
        patient_message: str,
        extraction: Dict[str, Any],
    ):
        self.add_message("user", patient_message)

        # Only accept answers for the feature currently being asked.
        if self.pending_feature is not None:
            extraction = dict(extraction)

            extraction["updates"] = [
                item
                for item in extraction.get("updates", [])
                if isinstance(item, dict)
                and item.get("feature") == self.pending_feature
            ]

        # Validate and store the accepted updates.
        validated = self.update_features(
            extraction,
            patient_message=patient_message,
        )

        # Clear the pending feature once it has a validated answer.
        if (
            self.pending_feature is not None
            and self.pending_feature in self.features
        ):
            self.clear_pending_feature()

        self.determine_stage()
        next_question = self.prepare_next_question()
        next_feature = self.pending_feature

        return {
            "updates": validated["updates"],
            "features": self.get_collected_features(),
            "missing_features": self.get_missing_features(),
            "next_feature": next_feature,
            "progress": self.get_progress(),
            "stage": self.stage,
            "completed": self.completed,
            "next_question": next_question,
        }


    # ========================================================
    # STATE
    # ========================================================

    def get_state(self):

        return {
            "features":
                self.get_collected_features(),

            "feature_metadata":
                deepcopy(
                    self.feature_metadata
                ),

            "missing_features":
                self.get_missing_features(),

            "progress":
                self.get_progress(),

            "stage":
                self.stage,

            "completed":
                self.completed,

            "conversation_history":
                deepcopy(
                    self.conversation_history
                ),
        }
    
    def set_pending_feature(self, feature: str, question: str):
        """Remember which feature the current question is about."""
        if feature not in FEATURE_SCHEMA:
            raise ValueError(f"Unknown feature: {feature}")

        self.pending_feature = feature
        self.pending_question = question

    def clear_pending_feature(self):
        self.pending_feature = None
        self.pending_question = None

    def get_pending_feature(self) -> Optional[str]:
        return self.pending_feature
