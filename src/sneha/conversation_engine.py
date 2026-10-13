
from typing import Any, Dict, Optional
from copy import deepcopy
from .safety import may_indicate_self_harm
from .conversation_manager import (
    ConversationManager,
    QUESTION_TEMPLATES,
)
from .feature_schema import SYMPTOM_FEATURES, SYMPTOM_SCALE
from .feature_extractor import (
    CATEGORY_VALUES,
    EXTRACTION_SYSTEM_PROMPT,
    build_targeted_extraction_prompt,
    parse_gestational_age_weeks,
    validate_extraction,
)
from .llm_client import SnehaLLMClient


class ConversationEngine:
    """Coordinate Sneha's conversation, Qwen, and feature validation."""

    SUGGESTION_CONFIRMATION_THRESHOLD = 0.8
    CLARIFICATION_LIMIT = 3
    NUMERIC_ANSWER_HINTS = {
        "Age": "Please type your age in completed years (for example, 26).",
        "Gestational Age": (
            "Please type a duration with its unit, such as 15 days or 12 weeks."
        ),
        "Number of sons": "Please type a whole number, such as 0, 1, or 2.",
        "Number of daughters": "Please type a whole number, such as 0, 1, or 2.",
        "Total Number of Children": (
            "Please type the total number of children as a whole number."
        ),
    }

    def __init__(
        self,
        llm_client: Optional[SnehaLLMClient] = None,
        manager: Optional[ConversationManager] = None,
    ):
        self.safety_check_pending = False
        self.screening_stopped = False
        self.pending_confirmation: Optional[Dict[str, Any]] = None
        self.clarification_attempts: Dict[str, int] = {}
        self._prediction_attempted = False
        self._prediction_result: Optional[Dict[str, Any]] = None
        self.llm = llm_client or SnehaLLMClient()
        self.manager = manager or ConversationManager()

    def _state_fields(self) -> Dict[str, Any]:
        stage = self.manager.determine_stage()
        return {
            "progress": self.manager.get_progress(),
            "stage": stage,
            "completed": self.manager.completed,
            "screening_stopped": self.screening_stopped,
        }

    def _generate_question(self, feature: str) -> str:
        """Return the reviewed question template without an LLM round-trip."""

        fallback = QUESTION_TEMPLATES.get(feature)

        if not fallback:
            raise ValueError(
                f"No question template for feature: {feature}"
            )

        return fallback

    @staticmethod
    def _parse_confirmation(message: str) -> Optional[bool]:
        normalized = " ".join(
            "".join(
                character.lower() if character.isalnum() else " "
                for character in message
            ).split()
        )
        if normalized in {
            "yes",
            "yeah",
            "yep",
            "correct",
            "right",
            "exactly",
            "that is right",
        }:
            return True
        if normalized in {
            "no",
            "nope",
            "incorrect",
            "not quite",
            "that is wrong",
            "that is not right",
        }:
            return False
        return None

    @staticmethod
    def _confirmation_value(update: Dict[str, Any]) -> str:
        feature = update["feature"]
        value = update["value"]
        if feature in SYMPTOM_FEATURES:
            return SYMPTOM_SCALE[int(value)]
        return str(value)

    def _confirmation_question(self) -> str:
        if self.pending_confirmation is None:
            raise RuntimeError("No answer is awaiting confirmation.")
        value = self._confirmation_value(self.pending_confirmation)
        return f"I understood {value}. Is that correct? Please answer yes or no."

    @staticmethod
    def _answer_options(feature: str) -> list[str]:
        if feature in SYMPTOM_FEATURES:
            return list(SYMPTOM_SCALE.values())
        if feature in CATEGORY_VALUES:
            return [
                option
                for option in CATEGORY_VALUES[feature]
                if option != "__MISSING__"
            ]
        return []

    def _failed_answer_response(
        self,
        result: Dict[str, Any],
        pending_feature: str,
        pending_question: str,
    ) -> Dict[str, Any]:
        self.clarification_attempts[pending_feature] = (
            self.clarification_attempts.get(pending_feature, 0) + 1
        )
        message = f"I didn't quite catch that. {pending_question}"
        response: Dict[str, Any] = {
            **result,
            "assistant_message": message,
            "next_question": message,
            "pending_feature": pending_feature,
            "screening_stopped": self.screening_stopped,
            "clarification_attempts": self.clarification_attempts[
                pending_feature
            ],
        }

        if self.clarification_attempts[pending_feature] >= self.CLARIFICATION_LIMIT:
            options = self._answer_options(pending_feature)
            if options:
                response["assistant_message"] = (
                    f"{message}\nPlease choose one of these options:"
                )
                response["next_question"] = response["assistant_message"]
                response["answer_options"] = options
            else:
                response["assistant_message"] = (
                    f"{message}\n"
                    f"{self.NUMERIC_ANSWER_HINTS[pending_feature]}"
                )
                response["next_question"] = response["assistant_message"]
                response["answer_hint"] = self.NUMERIC_ANSWER_HINTS[
                    pending_feature
                ]

        self.manager.add_message("assistant", response["assistant_message"])
        return response

    def start(self) -> Dict[str, Any]:
        """Start the interview and ask the first question."""
        if self.screening_stopped:
            return {
                "assistant_message": (
                    "The screening has stopped so we can focus on your safety."
                ),
                "pending_feature": None,
                **self._state_fields(),
            }

        if self.manager.started:
            feature = self.manager.get_pending_feature()

            if feature is None:
                question = self.manager.prepare_next_question()
                feature = self.manager.get_pending_feature()
            else:
                question = self.manager.pending_question
        else:
            self.manager.started = True
            question = self.manager.prepare_next_question()
            feature = self.manager.get_pending_feature()

        if feature is not None:
            question = self._generate_question(feature)
            self.manager.set_pending_feature(feature, question)
            self.manager.add_message("assistant", question)

        return {
            "assistant_message": question,
            "pending_feature": feature,
            **self._state_fields(),
        }

        
    def respond(self, patient_message: str) -> Dict[str, Any]:
        """Process a patient answer with safety checks and validation."""

        patient_message = patient_message.strip()

        if not patient_message:
            message = (
                "The screening has stopped so we can focus on your safety."
                if self.screening_stopped and not self.safety_check_pending
                else (
                    "Whenever you're ready, please answer the immediate-safety "
                    "question."
                    if self.safety_check_pending
                    else "Take your time. Whenever you're ready, please share your answer."
                )
            )
            return {
                "assistant_message": message,
                "pending_feature": (
                    None
                    if self.screening_stopped
                    else self.manager.get_pending_feature()
                ),
                **self._state_fields(),
            }

        if not self.manager.started:
            raise RuntimeError(
                "Start the interview before processing an answer."
            )

        # SAFETY: handle an ongoing safety follow-up first.
        if self.safety_check_pending:
            self.manager.add_message("user", patient_message)
            self.safety_check_pending = False

            safety_message = (
                "Thank you for answering. I will stop the screening here. If you "
                "are in immediate danger or may act on these thoughts, contact "
                "local emergency services now and reach out to someone trusted "
                "who can stay with you."
            )

            self.manager.add_message("assistant", safety_message)

            return {
                "assistant_message": safety_message,
                "pending_feature": None,
                **self._state_fields(),
                "safety_follow_up": False,
            }

        if self.screening_stopped:
            return {
                "assistant_message": (
                    "The screening has stopped and will not produce a model "
                    "result for this session. If you may be in immediate danger, "
                    "contact local emergency services and someone you trust."
                ),
                "pending_feature": None,
                **self._state_fields(),
                "safety_follow_up": False,
            }

        # SAFETY: interrupt the interview if self-harm may be indicated.
        if may_indicate_self_harm(patient_message):
            self.manager.add_message("user", patient_message)
            self.safety_check_pending = True
            self.screening_stopped = True
            self.pending_confirmation = None

            safety_message = (
                "Thank you for sharing that with me. I want to check your "
                "immediate safety: are you in danger right now, or do you "
                "feel you might act on these thoughts? If you may act on "
                "them or cannot stay safe, contact local emergency services "
                "now and reach out to someone trusted who can stay with you."
            )

            self.manager.add_message("assistant", safety_message)

            return {
                "assistant_message": safety_message,
                "pending_feature": None,
                **self._state_fields(),
                "safety_follow_up": True,
            }

        pending_feature = self.manager.get_pending_feature()
        pending_question = self.manager.pending_question

        if pending_feature is None or pending_question is None:
            return {
                "assistant_message": (
                    "Thank you for sharing. Let's check where we are "
                    "in the interview before continuing."
                ),
                "pending_feature": None,
                **self._state_fields(),
            }

        answer_options = self._answer_options(pending_feature)
        normalized_message = " ".join(patient_message.casefold().split())
        selected_option = next(
            (
                option
                for option in answer_options
                if normalized_message == option.casefold()
            ),
            None,
        )

        if (
            self.pending_confirmation is None
            and self.clarification_attempts.get(pending_feature, 0)
            >= self.CLARIFICATION_LIMIT
            and selected_option is not None
        ):
            if (
                pending_feature == SYMPTOM_FEATURES[-1]
                and list(SYMPTOM_SCALE.values()).index(selected_option) > 0
            ):
                self.manager.add_message("user", patient_message)
                self.screening_stopped = True
                self.safety_check_pending = True
                self.pending_confirmation = None
                safety_message = (
                    "Thank you for sharing that with me. I want to check your "
                    "immediate safety: are you in danger right now, or do you "
                    "feel you might act on these thoughts? If you may act on "
                    "them or cannot stay safe, contact local emergency services "
                    "now and reach out to someone trusted who can stay with you."
                )
                self.manager.add_message("assistant", safety_message)
                return {
                    "assistant_message": safety_message,
                    "pending_feature": None,
                    **self._state_fields(),
                    "safety_follow_up": True,
                }

            self.pending_confirmation = None
            result = self.manager.process_extraction(
                patient_message=patient_message,
                extraction={
                    "updates": [{
                        "feature": pending_feature,
                        "value": (
                            next(
                                score
                                for score, label in SYMPTOM_SCALE.items()
                                if label == selected_option
                            )
                            if pending_feature in SYMPTOM_FEATURES
                            else selected_option
                        ),
                        "confidence": 1.0,
                        "evidence": patient_message,
                    }],
                },
            )
            self.clarification_attempts.pop(pending_feature, None)
        elif self.pending_confirmation is not None:
            confirmation = self._parse_confirmation(patient_message)
            if confirmation is None:
                self.manager.add_message("user", patient_message)
                clarification = self._confirmation_question()
                self.manager.add_message("assistant", clarification)
                return {
                    "assistant_message": clarification,
                    "pending_feature": pending_feature,
                    **self._state_fields(),
                    "confirmation_pending": True,
                    "updates": [],
                }
            if not confirmation:
                self.manager.add_message("user", patient_message)
                self.pending_confirmation = None
                clarification = (
                    "Thanks for correcting me. "
                    f"{pending_question}"
                )
                self.manager.add_message("assistant", clarification)
                return {
                    "assistant_message": clarification,
                    "pending_feature": pending_feature,
                    **self._state_fields(),
                    "confirmation_pending": False,
                    "updates": [],
                }

            suggestion = self.pending_confirmation
            self.pending_confirmation = None
            result = self.manager.process_extraction(
                patient_message=patient_message,
                extraction={"updates": [suggestion]},
                evidence_message=suggestion["evidence"],
            )
            self.clarification_attempts.pop(pending_feature, None)
        else:
            gestational_age_weeks = (
                parse_gestational_age_weeks(patient_message)
                if pending_feature == "Gestational Age"
                else None
            )
            if gestational_age_weeks is not None:
                result = self.manager.process_extraction(
                    patient_message=patient_message,
                    extraction={
                        "updates": [{
                            "feature": "Gestational Age",
                            "value": gestational_age_weeks,
                            "confidence": 1.0,
                            "evidence": patient_message,
                        }],
                    },
                )
            else:
                # Keep recent conversation context for short answers.
                recent_history = []

                for message in self.manager.conversation_history[-8:]:
                    if isinstance(message, dict):
                        role = message.get("role", "")
                        content = message.get("content", "")
                        recent_history.append(f"{role}: {content}")

                prompt = build_targeted_extraction_prompt(
                    patient_message=patient_message,
                    current_features=self.manager.get_collected_features(),
                    pending_feature=pending_feature,
                    pending_question=pending_question,
                    recent_history=recent_history,
                )

                extraction = self.llm.extract_json(
                    system_prompt=EXTRACTION_SYSTEM_PROMPT,
                    user_prompt=prompt,
                )
                validated_extraction = validate_extraction(
                    extraction,
                    patient_message=patient_message,
                )
                result = self.manager.process_extraction(
                    patient_message=patient_message,
                    extraction=extraction,
                )

                suggestion = validated_extraction["suggested_update"]
                if (
                    not result["updates"]
                    and suggestion is not None
                    and suggestion["feature"] == pending_feature
                    and suggestion["confidence"]
                    >= self.SUGGESTION_CONFIRMATION_THRESHOLD
                ):
                    self.pending_confirmation = suggestion
                    confirmation_question = self._confirmation_question()
                    self.manager.add_message(
                        "assistant",
                        confirmation_question,
                    )
                    result["assistant_message"] = confirmation_question
                    result["next_question"] = confirmation_question
                    result["pending_feature"] = pending_feature
                    result["confirmation_pending"] = True
                    result["screening_stopped"] = self.screening_stopped
                    return result

        if not result["updates"]:
            return self._failed_answer_response(
                result,
                pending_feature,
                pending_question,
            )

        self.clarification_attempts.pop(pending_feature, None)

        next_feature = result.get("next_feature")

        if next_feature is not None:
            next_question = self._generate_question(next_feature)

            self.manager.set_pending_feature(
                next_feature,
                next_question,
            )
            self.manager.add_message("assistant", next_question)
            result["assistant_message"] = next_question
            result["next_question"] = next_question
            result["pending_feature"] = self.manager.get_pending_feature()
        else:
            result["assistant_message"] = None
            result["next_question"] = None
            result["pending_feature"] = None

            if self.manager.completed:
                screening_result = self.predict_result()
                result["screening_result"] = screening_result

                if screening_result.get("status") == "success":
                    final_message = (
                        "Thank you for answering all the questions. The Transformer "
                        "screening estimate is ready. This is not a diagnosis; please "
                        "discuss any concerns with a qualified healthcare professional."
                    )
                else:
                    final_message = (
                        "Thank you for completing the questions. I couldn't calculate "
                        "the screening result from the collected answers yet. "
                        + screening_result.get(
                            "message",
                            "Please check the model inputs.",
                        )
                    )

                self.manager.add_message("assistant", final_message)
                result["assistant_message"] = final_message

        result["screening_stopped"] = self.screening_stopped

        return result

    def predict_result(self) -> Dict[str, Any]:
        """Run the saved Tabular Transformer A after valid completion."""
        progress = self.manager.get_progress()
        if not self.manager.completed or not self.manager.is_complete():
            return {
                "status": "incomplete",
                "prediction": None,
                "class_probabilities": None,
                "missing_features": self.manager.get_missing_features(),
                "progress": progress,
                "completed": False,
            }

        if self.screening_stopped:
            return {
                "status": "safety_interrupted",
                "prediction": None,
                "class_probabilities": None,
                "progress": progress,
                "completed": False,
            }

        if self.screening_stopped:
            return {
                "status": "safety_interrupted",
                "prediction": None,
                "class_probabilities": None,
                "progress": progress,
                "completed": False,
            }

        if self._prediction_attempted:
            return deepcopy(self._prediction_result)

        self._prediction_attempted = True
        try:
            from src.config import TARGET_MAPPING
            from src.predict_transformer import (
                get_model_info,
                predict_antenatal_depression,
            )

            prediction = predict_antenatal_depression(
                self.manager.get_collected_features()
            )
            positive_probability = float(prediction["probability"])
            class_probabilities = {
                "Not Depressed": (
                    positive_probability
                    if TARGET_MAPPING["Not"] == 1
                    else 1.0 - positive_probability
                ),
                "Depressed": (
                    positive_probability
                    if TARGET_MAPPING["Depressed"] == 1
                    else 1.0 - positive_probability
                ),
            }
            self._prediction_result = {
                "status": "success",
                "model_info": get_model_info(),
                "prediction": prediction,
                "class_probabilities": class_probabilities,
                "progress": progress,
                "completed": True,
            }
        except (FileNotFoundError, ImportError, OSError) as exc:
            self._prediction_result = {
                "status": "unavailable",
                "message": str(exc),
                "prediction": None,
                "class_probabilities": None,
                "progress": progress,
                "completed": True,
            }
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            self._prediction_result = {
                "status": "invalid_features",
                "message": str(exc),
                "prediction": None,
                "class_probabilities": None,
                "progress": progress,
                "completed": True,
            }
        return deepcopy(self._prediction_result)
