from typing import Any, Dict, Optional

from .safety import may_indicate_self_harm
from .conversation_manager import ConversationManager, QUESTION_TEMPLATES
from .feature_extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    build_targeted_extraction_prompt,
)
from .llm_client import SnehaLLMClient


class ConversationEngine:
    """Coordinate Sneha's conversation, Qwen, feature validation, and screening."""

    def __init__(
        self,
        llm_client: Optional[SnehaLLMClient] = None,
        manager: Optional[ConversationManager] = None,
    ):
        self.safety_check_pending = False
        self.llm = llm_client or SnehaLLMClient()
        self.manager = manager or ConversationManager()

    def _generate_question(self, feature: str) -> str:
        """Generate a friendly question for one specific feature."""
        fallback = QUESTION_TEMPLATES.get(feature)

        if not fallback:
            raise ValueError(f"No question template for feature: {feature}")

        # Keep the self-harm screening question direct and unambiguous.
        if feature in (
            "Thoughts that you would be better off dead, or of hurting yourself",
            "Thoughts that you would be better off dead,or of hurting yourself",
        ):
            return fallback

        try:
            question = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Sneha, a warm and respectful antenatal "
                            "screening assistant. Ask exactly one short, clear "
                            "question at a time. Use friendly, natural language. "
                            "Do not diagnose, assume answers, or ask about "
                            "unrelated topics. Return only the question."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Required feature: {feature}\n"
                            f"Reference question: {fallback}\n\n"
                            "Ask a natural question that collects this specific "
                            "feature. Preserve its meaning. Do not introduce assumptions."
                        ),
                    },
                ],
                temperature=0.2,
            ).strip()

            if not question or "\n" in question:
                return fallback
            return question
        except (RuntimeError, ValueError):
            return fallback

    def start(self) -> Dict[str, Any]:
        """Start the interview and ask the first question."""
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
            "progress": self.manager.get_progress(),
            "completed": self.manager.completed,
        }

    def respond(self, patient_message: str) -> Dict[str, Any]:
        """Process an answer, then ask the next question or run Random Forest."""
        patient_message = patient_message.strip()

        if not patient_message:
            return {
                "assistant_message": (
                    "Take your time. Whenever you're ready, please share your answer."
                ),
                "pending_feature": self.manager.get_pending_feature(),
                "completed": self.manager.completed,
            }

        if not self.manager.started:
            raise RuntimeError("Start the interview before processing an answer.")

        # SAFETY: handle an ongoing safety follow-up first.
        if self.safety_check_pending:
            self.manager.add_message("user", patient_message)
            safety_message = (
                "Thank you for telling me. Are you in immediate danger right now, "
                "or do you feel you might act on these thoughts? If you may act on "
                "them or cannot stay safe, contact local emergency services now "
                "and reach out to someone trusted who can stay with you."
            )
            self.manager.add_message("assistant", safety_message)
            return {
                "assistant_message": safety_message,
                "pending_feature": self.manager.get_pending_feature(),
                "completed": False,
                "safety_follow_up": True,
            }

        # SAFETY: interrupt the interview if self-harm may be indicated.
        if may_indicate_self_harm(patient_message):
            self.manager.add_message("user", patient_message)
            self.safety_check_pending = True
            safety_message = (
                "Thank you for sharing that with me. I want to check your immediate "
                "safety: are you in danger right now, or do you feel you might act "
                "on these thoughts? If you may act on them or cannot stay safe, "
                "contact local emergency services now and reach out to someone "
                "trusted who can stay with you."
            )
            self.manager.add_message("assistant", safety_message)
            return {
                "assistant_message": safety_message,
                "pending_feature": self.manager.get_pending_feature(),
                "completed": False,
                "safety_follow_up": True,
            }

        pending_feature = self.manager.get_pending_feature()
        pending_question = self.manager.pending_question

        if pending_feature is None or pending_question is None:
            return {
                "assistant_message": (
                    "Thank you for sharing. Let's check where we are in the interview "
                    "before continuing."
                ),
                "pending_feature": None,
                "completed": self.manager.completed,
            }

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

        result = self.manager.process_extraction(
            patient_message=patient_message,
            extraction=extraction,
        )

        next_feature = result.get("next_feature")

        if next_feature is not None:
            next_question = self._generate_question(next_feature)
            self.manager.set_pending_feature(next_feature, next_question)
            self.manager.add_message("assistant", next_question)
            result["assistant_message"] = next_question
            result["next_question"] = next_question
            result["pending_feature"] = self.manager.get_pending_feature()
        else:
            # Run Random Forest only when the manager confirms the interview is complete.
            if self.manager.completed:
                screening_result = self.predict_random_forest_result()
                result["screening_result"] = screening_result

                if screening_result.get("status") == "success":
                    final_message = (
                        "Thank you for answering all the questions. I've completed "
                        "the screening calculation. This result is only a screening "
                        "estimate, not a diagnosis. A qualified healthcare professional "
                        "can help interpret it and discuss any concerns."
                    )
                else:
                    final_message = (
                        "Thank you for completing the questions. I couldn't calculate "
                        "the screening result from the collected answers yet. "
                        + str(screening_result.get("message", "Please check the model inputs."))
                    )
                self.manager.add_message("assistant", final_message)
                result["assistant_message"] = final_message
                result["next_question"] = None
                result["pending_feature"] = None
            else:
                result["assistant_message"] = None
                result["next_question"] = None
                result["pending_feature"] = self.manager.get_pending_feature()

        return result

    def predict_random_forest_result(self) -> Dict[str, Any]:
        """Run Random Forest Experiment A using collected raw questionnaire answers."""
        from src.predict_random_forest import (
            predict_antenatal_depression,
            get_model_info,
        )

        features = self.manager.get_collected_features()
        try:
            prediction = predict_antenatal_depression(features)
            return {
                "status": "success",
                "model_info": get_model_info(),
                "prediction": prediction,
            }
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            return {
                "status": "invalid_features",
                "message": str(exc),
            }

    def predict_result(self) -> Dict[str, Any]:
        """Run the existing Transformer A predictor; kept separate from Random Forest."""
        from src.predict_transformer import (
            predict_antenatal_depression,
            get_model_info,
        )

        model_info = get_model_info()
        required_columns = (
            model_info["numeric_features"]
            + model_info["categorical_features"]
            + model_info["symptom_features"]
        )
        features = self.manager.get_collected_features()
        missing = [
            column for column in required_columns
            if column not in features or features[column] is None
        ]

        if missing:
            return {
                "status": "incomplete",
                "message": (
                    "Please complete the required questions before running the model."
                ),
                "missing_features": missing,
            }

        try:
            prediction = predict_antenatal_depression(features)
        except (ValueError, KeyError) as exc:
            return {"status": "invalid_features", "message": str(exc)}

        return {
            "status": "success",
            "model": model_info["model"],
            "prediction": prediction,
        }
