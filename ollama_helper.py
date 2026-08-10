# ==========================================================
# AnimalVision AI
# Ollama AI Assistant
# ==========================================================

import ollama


# ==========================================================
# AI FUNCTION
# ==========================================================

def ask_animal_ai(animal, confidence, question):

    prompt = f"""
You are AnimalVision AI, a professional wildlife expert.

Detected Animal:
{animal}

Prediction Confidence:
{confidence*100:.2f}%

User Question:
{question}

Rules:

1. Answer ONLY the user's question.
2. Do not generate a complete animal report unless the user explicitly asks for:
   - "Generate report"
   - "Complete report"
   - "Full report"
3. Keep answers concise (2–5 sentences) unless more detail is requested.
4. Use the detected animal as the context.
5. If the user asks something impossible (example: "Can sheep fly?"), answer it correctly and briefly.
6. If the confidence is low (below 80%), mention that the prediction may not be fully accurate.
7. Never ignore the user's question.
8. Do not start with phrases like "Based on my extensive knowledge..." or generate unrelated information.

Answer naturally in simple English.
"""

    try:

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return f"❌ Ollama AI Error\n\n{str(e)}"