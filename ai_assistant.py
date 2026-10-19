from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    # Not needed as we are hitting local LLM, adding dummy key since OpenAI has made it non null
    api_key=os.getenv("API_KEY"),
)

while True:
    user_input = input("\nYou:")
    if user_input.lower() in ["quit", "bye", "exit"]:
        break
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=[{"role": "user", "content": user_input}],
    )
    print("Assistant: ", response.choices[0].message.content)
