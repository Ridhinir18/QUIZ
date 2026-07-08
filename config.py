import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
Ai_MODEL = os.getenv("AI_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "lmstestseries0987654321")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)