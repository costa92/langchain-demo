from google import genai
import os 

api_key = os.getenv("GOOGLE_AI_API_KEY")
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="上海在什么地方",
)

print(response.text)