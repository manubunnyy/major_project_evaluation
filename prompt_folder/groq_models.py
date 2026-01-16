import requests
import os

api_key = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

data = response.json()
for model in data.get('data', []):
    print(model.get('id'))