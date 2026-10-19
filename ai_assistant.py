from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    # Not needed as we are hitting local LLM, adding dummy key since OpenAI has made it non null
    api_key=os.getenv("API_KEY"),
)

# Introducing memory in our assistant
messages = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "bye", "exit"]:
        break
    messages.append({"content": user_input, "role": "user"})
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
    )
    response_content = response.choices[0].message.content
    messages.append({"content": response_content, "role": "assistant"})
    print("Assistant: ", response.choices[0].message.content)
