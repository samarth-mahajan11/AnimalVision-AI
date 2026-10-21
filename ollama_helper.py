
import os
import ollama


def ask_animal_ai(animal, confidence, question):
    if not animal:
        return "Please upload an animal image first."

    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        return "AI assistant configuration is missing. Please contact the app administrator."

    prompt = f"""
You are AnimalVision AI, a professional wildlife expert.

Detected Animal: {animal}
Prediction Confidence: {confidence * 100:.2f}%

User Question: {question}

Rules:
1. Answer only the user's question.
2. Keep answers concise (2-5 sentences unless asked for more).
3. Use the detected animal as context.
4. Answer accurately in simple English.
5. If confidence is below 80%, mention that the prediction may not be fully accurate.
"""

    try:
        client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=90,
        )

        response = client.chat(
            model="gemma4:31b",
            messages=[{"role": "user", "content": prompt}],
        )

        return response["message"]["content"]

    except Exception as exc:
        print(f"Ollama Cloud request failed: {type(exc).__name__}: {exc}")
        return (
            "The AI assistant is temporarily unavailable. "
            "Please try again later. Animal classification remains available."
        )
