
from src.sneha.conversation_engine import ConversationEngine


def main():
    engine = ConversationEngine()

    result = engine.start()

    print("\nSneha:", result["assistant_message"])
    print("Pending feature:", result["pending_feature"])

    while not engine.manager.completed:
        patient_message = input("\nPatient: ").strip()

        if patient_message.lower() in {"quit", "exit"}:
            print("Interview stopped.")
            break

        try:
            result = engine.respond(patient_message)

            print("\nAccepted updates:", result.get("updates", []))
            print("Progress:", result.get("progress"))
            print("Stage:", result.get("stage"))
            print("Pending feature:", result.get("pending_feature"))
            print("Sneha:", result.get("assistant_message"))

        except (RuntimeError, ValueError) as exc:
            print("Interview error:", exc)
            break


if __name__ == "__main__":
    main()

