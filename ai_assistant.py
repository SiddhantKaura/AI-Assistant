from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from tools import TOOLS, FUNCTION_NAME_TO_FUNCTION_MAP

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    # Not needed as we are hitting local LLM, adding dummy key since OpenAI has made it non null
    api_key=os.getenv("API_KEY"),
)

# Introducing memory in our assistant
messages = []
messages.append(
    {
        "role": "system",
        "content": "Act as my assistant and reply in less than 20 words.",
    }
)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "bye", "exit"]:
        break
    messages.append({"content": user_input, "role": "user"})
    while True:
        response = client.chat.completions.create(
            model=os.getenv("MODEL"), messages=messages, tools=TOOLS
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            print("Assistant: ", msg.content)
            break

        for tool_call in msg.tool_calls:
            print(tool_call.function)
            tool_call_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(tool_call_name, tool_args)
            # ** -> Dictionary Unpacking operator converts dict to attributes
            tool_response = FUNCTION_NAME_TO_FUNCTION_MAP.get(tool_call_name)(
                **tool_args
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_response),
                }
            )
