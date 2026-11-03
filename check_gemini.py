"""
check_gemini.py  –  Quick connectivity test for the Gemini API.

Run from the project root after setting GEMINI_API_KEY in your .env:

    python check_gemini.py
"""
from dotenv import load_dotenv

load_dotenv()

import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")

model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

print(f"Checking Gemini API connectivity (model: {model_name}) ...")
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model=model_name,
    contents="Reply with exactly: GEMINI_READY",
)

print("API response:", response.text.strip())
print("\nGemini API is reachable and ready for medical report extraction.")
