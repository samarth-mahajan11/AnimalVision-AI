import os
from dotenv import load_dotenv
from google import genai


# Load .env file
load_dotenv()


# Get API key
API_KEY = os.getenv("GEMINI_API_KEY")


# Create Gemini client
client = genai.Client(
    api_key=API_KEY
)


# ==========================================
# Generate animal information
# ==========================================

def generate_animal_info(animal_name):

    prompt = f"""

You are an expert wildlife assistant.

The AI detected this animal:

{animal_name}


Provide information in this format:

🐾 Animal:
{animal_name}

🧬 Scientific Name:

🌍 Habitat:

🍽 Diet:

⏳ Lifespan:

🛡 Conservation Status:

💡 Interesting Fact:


Use simple English.
Keep it professional and concise.

"""

    response = client.models.generate_content(
       model="gemini-2.0-flash-lite",
        contents=prompt
    )

    return response.text



# ==========================================
# Ask questions about animal
# ==========================================

def ask_about_animal(animal_name, question):


    prompt = f"""

You are a wildlife expert.

Animal:
{animal_name}


User Question:

{question}


Answer clearly in simple English.

"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )


    return response.text