
from src.sneha.llm_client import SnehaLLMClient
from src.sneha.feature_extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extraction_prompt,
    build_targeted_extraction_prompt,
    validate_extraction,
)

def main():
    client = SnehaLLMClient()

    patient_message = input(
        "Patient: "
    ).strip()

    if not patient_message:
        print("Please enter a patient message.")
        return

    # No features are known in this first test.
    current_features = {}

    
    pending_feature = "Feeling tired or having little energy"
    pending_question = "How often have you been feeling tired or low on energy?"

    recent_history = [
        "Sneha: How often have you been feeling tired or low on energy?",
    ]

    prompt = build_targeted_extraction_prompt(
        patient_message=patient_message,
        current_features={},
        pending_feature=pending_feature,
        pending_question=pending_question,
        recent_history=recent_history,
    )


    print("\nSneha is thinking...\n")

    result = client.extract_json(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=prompt,
    )

    
    # Deterministic guard: only accept the feature currently being asked.
    if isinstance(result, dict):
        result["updates"] = [
            item
            for item in result.get("updates", [])
            if isinstance(item, dict)
            and item.get("feature") == pending_feature
        ]

    validated = validate_extraction(
        result,
        patient_message=patient_message,
    )

    
    if patient_message.strip().lower() in {
    "nearly every day",
    "almost every day",
    "every day",
    "several days",
    "not at all",
    "more than half the days",
}:
        print(
            "\nSafety check: this is a frequency-only answer. "
            "It cannot identify a symptom by itself."
        )
        print("Discarding extracted updates for this isolated test.")
        validated["updates"] = []
    print("Validated extraction:")
    print(validated)

    tiredness_feature = "Feeling tired or having little energy"

    # If tiredness was mentioned but no valid score was extracted,
    # ask a clarification instead of moving to another symptom.
    mentioned_tiredness = any(
        word in patient_message.lower()
        for word in ["tired", "fatigue", "little energy", "exhausted"]
    )

    tiredness_recorded = any(
        update["feature"] == tiredness_feature
        for update in validated["updates"]
    )

    if mentioned_tiredness and not tiredness_recorded:
        print(
            "\nSneha: I'm sorry you've been feeling so drained. "
            "How often have you been feeling this way?"
        )
    else:
        print("\nSneha:", validated["next_question"])


if __name__ == "__main__":
    main()
