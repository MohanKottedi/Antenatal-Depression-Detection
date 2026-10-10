
import json
import os
from typing import Any

import requests


class SnehaLLMClient:
    """Local Qwen client using the Ollama chat API."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            content = data.get("message", {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("Ollama returned an empty response.")

            return content.strip()

        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Check that Ollama is running "
                "and that the model is available."
            ) from exc

    def extract_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        raw = self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            json_mode=True,
        )

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Qwen returned invalid JSON: {raw}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object from Qwen.")

        return result


if __name__ == "__main__":
    client = SnehaLLMClient()

    reply = client.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are Sneha, a warm, respectful antenatal "
                    "screening assistant. Ask one question at a time. "
                    "Do not diagnose or make assumptions about the patient."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Start a conversation with a pregnant patient "
                    "in a gentle, nonjudgmental way."
                ),
            },
        ]
    )

    print("\nSneha:", reply)
