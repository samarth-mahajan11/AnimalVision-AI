
# ==========================================================
# AnimalVision AI
# Professional AI Wildlife Recognition System
# ==========================================================

import os
import json
import html

import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

from animal_info import ANIMAL_INFO

# Optional AI assistant
try:
    from ollama_helper import ask_animal_ai
except ImportError:
    ask_animal_ai = None

# Optional voice input
try:
    from streamlit_mic_recorder import speech_to_text
except ImportError:
    speech_to_text = None

# Optional text-to-speech
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AnimalVision AI",
    page_icon="🐾",
    layout="wide"
)


# ==========================================================
# TEXT TO SPEECH
# ==========================================================

def speak_text(text):
    """Display browser-based speech controls for the AI response."""

    import json

    speech_text = json.dumps(str(text))

    components_html = f"""
    <button id="speak" style="
        background:#2563eb;color:white;border:0;
        border-radius:8px;padding:10px 16px;cursor:pointer;">
        🔊 Speak AI Response
    </button>
    <button id="stop" style="
        background:#475569;color:white;border:0;
        border-radius:8px;padding:10px 16px;cursor:pointer;">
        ⏹ Stop
    </button>

    <script>
    const speechText = {speech_text};

    document.getElementById("speak").onclick = () => {{
        if (!window.speechSynthesis) {{
            alert("Speech playback is not supported by this browser.");
            return;
        }}

        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(speechText);
        utterance.lang = "en-US";
        utterance.rate = 1;
        window.speechSynthesis.speak(utterance);
    }};

    document.getElementById("stop").onclick = () => {{
        window.speechSynthesis.cancel();
    }};
    </script>
    """

    st.components.v1.html(components_html, height=60)



# ==========================================================
# SESSION STORAGE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0


# ==========================================================
# LOAD CSS
# ==========================================================

def load_css():
    css_path = "css/style.css"

    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as css_file:
            st.markdown(
                f"<style>{css_file.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "models/AnimalSpeciesAI_Final.keras"
CONFIG_PATH = "project_config.json"


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ==========================================================
# LOAD CONFIG
# ==========================================================

@st.cache_data
def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


try:
    model = load_model()
    config = load_config()

    CLASS_NAMES = config["class_names"]
    IMAGE_SIZE = tuple(config["image_size"])

except Exception as error:
    st.error(
        "AnimalVision AI could not load its model or configuration. "
        "Check that the model and project_config.json are committed "
        "to your GitHub repository."
    )
    st.exception(error)
    st.stop()


# ==========================================================
# ANIMAL ICONS
# ==========================================================

animal_icons = {
    "Dog": "🐶",
    "Cat": "🐱",
    "Horse": "🐴",
    "Butterfly": "🦋",
    "Chicken": "🐔",
    "Cow": "🐄",
    "Elephant": "🐘",
    "Sheep": "🐑",
    "Spider": "🕷️",
    "Squirrel": "🐿️"
}


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.title("🐾 AnimalVision AI")
    st.markdown("---")

    st.subheader("🤖 AI Model")

    st.write(
        """
        🧠 Architecture: EfficientNetB0

        ⚡ Framework: TensorFlow

        🦾 AI Assistant: Ollama Llama 3.2

        🎨 Frontend: Streamlit
        """
    )

    st.markdown("---")
    st.subheader("📜 Prediction History")

    if not st.session_state.history:
        st.write("No predictions yet.")
    else:
        for item in st.session_state.history[-5:][::-1]:
            st.write(
                f"🐾 {item['animal']}\n\n"
                f"Confidence: {item['confidence']:.2f}%"
            )


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="title">🐾 AnimalVision AI</div>
    <div class="subtitle">
        Deep Learning Powered Wildlife Recognition
    </div>
    """,
    unsafe_allow_html=True
)

st.success("🤖 AnimalVision AI model loaded successfully")

st.markdown("---")


# ==========================================================
# IMAGE INPUT
# ==========================================================

st.markdown(
    """
    <h3 style="text-align:center;color:#38BDF8;">
        🎯 AI Confidence
    </h3>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Animal Image",
    type=["jpg", "jpeg", "png"]
)


# ==========================================================
# PREDICTION
# ==========================================================

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        # Preprocess image
        img = image.resize(IMAGE_SIZE)
        img_array = np.asarray(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        with st.spinner(
            "🧠 AnimalVision AI is analyzing the image..."
        ):
            prediction = model.predict(img_array, verbose=0)

        predicted_index = int(np.argmax(prediction[0]))
        predicted_animal = CLASS_NAMES[predicted_index]
        confidence = float(np.max(prediction[0]))

        # Store latest prediction
        st.session_state.prediction = predicted_animal
        st.session_state.confidence = confidence

        # Save history only once for this uploaded image
        image_key = f"{uploaded_file.name}:{uploaded_file.size}"

        if st.session_state.get("last_image_key") != image_key:
            st.session_state.history.append({
                "animal": predicted_animal,
                "confidence": confidence * 100
            })
            st.session_state.last_image_key = image_key

        icon = animal_icons.get(predicted_animal, "🐾")
        image_path = f"assets/{predicted_animal.lower()}.png"

        with col2:
            with st.container(border=True):
                if os.path.exists(image_path):
                    st.image(image_path, width=340)
                else:
                    st.markdown(
                        f"<h1 style='text-align:center'>{icon}</h1>",
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f"""
                    <h2 style="text-align:center;color:white;
                               margin-top:5px;font-size:38px;
                               font-weight:800;">
                        {html.escape(str(predicted_animal))}
                    </h2>
                    <p style="text-align:center;color:#CBD5E1;
                              font-size:20px;">
                        AI Prediction
                    </p>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### 🎯 Confidence Score")
                st.progress(max(0.0, min(confidence, 1.0)))

                st.markdown(
                    f"""
                    <h2 style="text-align:center;color:#22C55E;
                               font-size:40px;font-weight:900;">
                        {confidence * 100:.2f}%
                    </h2>
                    """,
                    unsafe_allow_html=True
                )

                if confidence >= 0.95:
                    st.success("🟢 Very High Confidence")
                elif confidence >= 0.80:
                    st.info("🔵 Good Confidence")
                else:
                    st.warning(
                        "🟡 Lower confidence — try a clearer image."
                    )

    except Exception as error:
        st.error(
            "Unable to process this image. Please upload a valid "
            "JPG or PNG image and try again."
        )
        st.exception(error)


# ==========================================================
# CURRENT PREDICTION
# ==========================================================

predicted_animal = st.session_state.prediction
confidence = st.session_state.confidence


# ==========================================================
# ANIMAL INFORMATION
# ==========================================================

if predicted_animal and predicted_animal in ANIMAL_INFO:
    info = ANIMAL_INFO[predicted_animal]

    st.markdown("---")
    st.markdown(
        """
        <h2 style="text-align:center;color:white">
            📖 Animal Information
        </h2>
        """,
        unsafe_allow_html=True
    )

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        with st.container(border=True):
            st.subheader("🌍 Habitat")
            st.write(info["habitat"])

        with st.container(border=True):
            st.subheader("🍽️ Diet")
            st.write(info["diet"])

    with col_info2:
        with st.container(border=True):
            st.subheader("🧬 Scientific Name")
            st.write(info["scientific"])

        with st.container(border=True):
            st.subheader("⏳ Lifespan")
            st.write(info["lifespan"])

    with st.container(border=True):
        st.subheader("💡 Interesting Fact")
        st.write(info["fact"])

else:
    st.info("Upload an animal image to view its information.")


# ==========================================================
# AI ASSISTANT
# ==========================================================

st.markdown("---")
st.subheader("🧠 AnimalVision AI Assistant")

mode = st.radio(
    "Choose Assistant Mode",
    ["📝 Text Assistant", "🎤 Voice Assistant"],
    horizontal=True
)


# ==========================================================
# ASSISTANT HELPER
# ==========================================================

def get_ai_answer(question):
    """Call the configured assistant without crashing the app."""

    if not predicted_animal:
        return "Please upload an animal image first."

    if ask_animal_ai is None:
        return (
            "The AI assistant is not configured correctly yet. "
            "Animal classification and animal information remain available."
        )

    try:
        return ask_animal_ai(
            predicted_animal,
            confidence,
            question
        )
    except Exception:
        return (
            "The AI assistant is temporarily unavailable. "
            "Please try again later."
        )


# ==========================================================
# TEXT ASSISTANT
# ==========================================================

if mode == "📝 Text Assistant":
    question = st.text_input(
        "Ask anything about this animal...",
        key="text_ai_question"
    )

    if st.button("🚀 Ask AI", key="text_ai_button"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not predicted_animal:
            st.warning("Please upload an animal image first.")
        else:
            with st.spinner("🤖 AnimalVision AI is thinking..."):
                answer = get_ai_answer(question.strip())

            st.markdown("### 🤖 AnimalVision AI")
            st.write(answer)


# ==========================================================
# VOICE ASSISTANT
# ==========================================================

else:
    st.info(
        "🎤 Use the microphone to ask your question. "
        "Voice input requires microphone permission in your browser."
    )

    if speech_to_text is None:
        st.warning(
            "Voice input is unavailable because "
            "`streamlit-mic-recorder` could not be imported. "
            "Please use Text Assistant."
        )
    else:
        try:
            voice_text = speech_to_text(
                language="en",
                use_container_width=True,
                just_once=True,
                key="voice_ai"
            )

            if voice_text:
                st.success(f"🎤 You said: {voice_text}")

                if not predicted_animal:
                    st.warning("Please upload an animal image first.")
                else:
                    with st.spinner(
                        "🤖 AnimalVision AI is thinking..."
                    ):
                        answer = get_ai_answer(voice_text)

                    st.markdown("### 🤖 AnimalVision AI")
                    st.write(answer)

                    speak_text(answer)

        except Exception:
            st.warning(
                "Voice input is unavailable right now. "
                "Please use Text Assistant instead."
            )
