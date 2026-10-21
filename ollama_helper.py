
# ==========================================================
# AnimalVision AI - Ollama AI Assistant
# ==========================================================

import os
import ollama


def ask_animal_ai(animal, confidence, question):

    if not animal:
        return "Please upload an animal image first."

    prompt = f"""
You are AnimalVision AI, a professional wildlife expert.

Detected Animal: {animal}
Prediction Confidence: {confidence * 100:.2f}%

User Question: {question}

Rules:
1. Answer only the user's question.
2. Keep answers concise (2–5 sentences unless asked for more).
3. Use the detected animal as context.
4. Answer accurately and naturally in simple English.
5. If confidence is below 80%, mention that the prediction
   may not be fully accurate.
"""

    try:
        # Configure OLLAMA_HOST in Streamlit Secrets when
        # using a remotely hosted Ollama service.
        host = os.getenv("OLLAMA_HOST")

        client = ollama.Client(host=host) if host else ollama.Client()

        response = client.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]

    except Exception:
        return (
            "The AI assistant is temporarily unavailable. "
            "Animal classification and animal information "
            "are still available. Please try again later."
        )
