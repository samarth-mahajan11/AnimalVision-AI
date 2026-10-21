
# ==========================================================
# AnimalVision AI - Gemini AI Assistant
# ==========================================================

import streamlit as st
from google import genai


def ask_animal_ai(animal, confidence, question):

    if not animal:
        return "Please upload an animal image first."

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        if response.text:
            return response.text

        return "I couldn't generate an answer. Please try again."

    except Exception:
        return (
            "The AI assistant is temporarily unavailable. "
            "Please check the API key and try again later. "
            "Animal classification is still available."
        )
