# ==========================================================
# AnimalVision AI
# Professional AI Wildlife Recognition System
# ==========================================================

import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

from animal_info import ANIMAL_INFO
from ollama_helper import ask_animal_ai
from streamlit_mic_recorder import speech_to_text
import pyttsx3


# ==========================================================
# TEXT TO SPEECH
# ==========================================================

def speak_text(text):

    engine = pyttsx3.init()

    engine.setProperty("rate", 170)

    engine.setProperty("volume", 1.0)

    engine.say(text)

    engine.runAndWait()

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AnimalVision AI",
    page_icon="🐾",
    layout="wide"
)


# ==========================================================
# SESSION STORAGE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================================
# LOAD CSS
# ==========================================================

def load_css():

    css_path = "css/style.css"

    if os.path.exists(css_path):

        with open(css_path) as f:

            st.markdown(
                f"<style>{f.read()}</style>",
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

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()



# ==========================================================
# LOAD CONFIG
# ==========================================================

with open(CONFIG_PATH,"r") as f:

    config = json.load(f)


CLASS_NAMES = config["class_names"]

IMAGE_SIZE = tuple(
    config["image_size"]
)
# Global prediction variables

predicted_animal = None
confidence = 0.0



# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:


    st.title("🐾 AnimalVision AI")


    st.markdown("---")


    st.subheader("🤖 AI Model")


    st.write(
        """
        🧠 Architecture:
        EfficientNet

        ⚡ Framework:
        TensorFlow

        🦾 AI Assistant:
        Ollama Llama 3.2

        🎨 Frontend:
        Streamlit
        """
    )


    st.markdown("---")


    st.subheader("📜 Prediction History")


    if len(st.session_state.history) == 0:

        st.write(
            "No predictions yet."
        )

    else:

        for item in st.session_state.history[-5:]:

            st.write(
                f"""
                🐾 {item['animal']}

                Confidence:
                {item['confidence']:.2f}%
                """
            )



# ==========================================================
# HEADER
# ==========================================================


st.markdown(
"""
<div class="title">

🐾 AnimalVision AI

</div>


<div class="subtitle">

Deep Learning Powered Wildlife Recognition

</div>

""",
unsafe_allow_html=True
)



st.success(
    "🤖 AI Model Loaded Successfully"
)


st.markdown("---")



# ==========================================================
# IMAGE INPUT
# ==========================================================

st.markdown(
    """
    <h3 style='text-align:center;color:#38BDF8;'>
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
# PREDICTION
# ==========================================================


if uploaded_file is not None:


        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])

        with col1:
          st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
          )



    



        # ==================================================
        # PREPROCESS IMAGE
        # ==================================================


        img = image.convert("RGB")


        img = img.resize(
            IMAGE_SIZE
        )


        img_array = np.array(
            img
        )


        img_array = np.expand_dims(

            img_array,

            axis=0

        )



        # ==================================================
        # MODEL PREDICTION
        # ==================================================


        with st.spinner(
            "🧠 AnimalVision AI is analyzing the uploaded image..."
        ):


            prediction = model.predict(

                img_array,

                verbose=0

            )



        predicted_index = np.argmax(
            prediction[0]
        )


        predicted_animal = CLASS_NAMES[
            predicted_index
        ]



        confidence = float(
            np.max(prediction[0])
        )



        # ==================================================
        # SAVE HISTORY
        # ==================================================


        st.session_state.history.append(

            {

                "animal": predicted_animal,

                "confidence": confidence * 100

            }

        )



        icon = animal_icons.get(

            predicted_animal,

            "🐾"

        )



        # ==================================================
        # PREDICTION DISPLAY
        # ==================================================
        with col2:

            image_path = f"assets/{predicted_animal.lower()}.png"

            with st.container(border=True):

                # Animal Image
                if os.path.exists(image_path):
                    st.image(
                        image_path,
                        width=340
                    )
                else:
                    st.markdown(
                        f"<h1 style='text-align:center'>{icon}</h1>",
                        unsafe_allow_html=True
                    )

                # Animal Name
                st.markdown(
                    f"""
                    <h2 style='text-align:center;
                            color:white;
                            margin-top:5px;
                            font-size:38px;
                            font-weight:800;'>

                    {predicted_animal}

                    </h2>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    """
                    <p style='text-align:center;
                            color:#CBD5E1;
                            font-size:20px;'>

                    AI Prediction

                    </p>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### 🎯 Confidence Score")

                st.progress(confidence)

                st.markdown(
                    f"""
                    <h2 style='text-align:center;
                            color:#22C55E;
                            font-size:40px;
                            font-weight:900;'>

                    {confidence*100:.2f}%

                    </h2>
                    """,
                    unsafe_allow_html=True
                )

                if confidence >= 0.95:
                    st.success("🟢 Very High Confidence")

                elif confidence >= 0.80:
                    st.info("🔵 Good Confidence")

                else:
                    st.warning("🟡 Low Confidence - Try clearer image")

        # ==========================================================
# ANIMAL INFORMATION SECTION
# ==========================================================
# ANIMAL INFORMATION SECTION
# ==========================================================


if predicted_animal is not None and predicted_animal in ANIMAL_INFO:


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


        st.markdown(

        f"""

        <div class="info-card">


        <h3>🌍 Habitat</h3>

        <h2>{info['habitat']}</h2>


        </div>


        """,

        unsafe_allow_html=True

        )



        st.markdown("<br>", unsafe_allow_html=True)



        st.markdown(

        f"""

        <div class="info-card">


        <h3>🍽 Diet</h3>

        <h2>{info['diet']}</h2>


        </div>


        """,

        unsafe_allow_html=True

        )





    with col_info2:


        st.markdown(

        f"""

        <div class="info-card">


        <h3>🧬 Scientific Name</h3>

        <h4>{info['scientific']}</h4>


        </div>


        """,

        unsafe_allow_html=True

        )



        st.markdown("<br>", unsafe_allow_html=True)



        st.markdown(

        f"""

        <div class="info-card">


        <h3>⏳ Lifespan</h3>

        <h2>{info['lifespan']}</h2>


        </div>


        """,

        unsafe_allow_html=True

        )





    st.markdown("<br>", unsafe_allow_html=True)



    st.markdown(

    f"""

    <div class="fact-card">


    <h2>💡 Interesting Fact</h2>


    <p style="font-size:18px">

    {info['fact']}

    </p>


    </div>


    """,

    unsafe_allow_html=True

    )



else:


    st.warning(
        "No static information available. Ask AI for details."
    )

# ==========================================================
# AI ASSISTANT
# ==========================================================

st.markdown("---")

st.subheader("🧠 AnimalVision AI Assistant")

# -------------------------------
# Mode Selection
# -------------------------------

mode = st.radio(
    "Choose Assistant Mode",
    ["📝 Text Assistant", "🎤 Voice Assistant"],
    horizontal=True
)

# -------------------------------
# TEXT MODE
# -------------------------------

if mode == "📝 Text Assistant":

    question = st.text_input(
        "Ask anything about this animal..."
    )

    if st.button("🚀 Ask AI"):

        if question.strip():

            with st.spinner("🤖 AnimalVision AI is thinking..."):

                answer = ask_animal_ai(
                    predicted_animal,
                    confidence,
                    question
                )

            st.markdown(
                f"""
                <div class="ai-response">

                <h3>🤖 AnimalVision AI</h3>

                <p>

                {answer}

                </p>

                </div>
                """,
                unsafe_allow_html=True
                )

           

        else:

            st.warning("Please enter a question.")

# -------------------------------
# VOICE MODE
# -------------------------------

else:

    st.info(
        "🎤 Press the microphone and ask your question."
    )

    voice_text = speech_to_text(
        language="en",
        use_container_width=True,
        just_once=True,
        key="voice_ai"
    )

    if voice_text:

        st.success(f"🎤 You said : {voice_text}")

        with st.spinner("🤖 AnimalVision AI is thinking..."):

            answer = ask_animal_ai(
                predicted_animal,
                confidence,
                voice_text
            )

        st.info("🔊 AnimalVision AI is speaking...")

        speak_text(answer)