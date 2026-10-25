# Standard library imports
import os

# Third party imports
from openai import OpenAI
from dotenv import load_dotenv

# Local modules
from tools import TOOLS, run_tool
from utils.utils import log  # From folder.file

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    # Not needed as we are hitting local LLM, adding dummy key since OpenAI has made it non null
    api_key=os.getenv("API_KEY"),
)

MAX_STEPS = 10

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

    # Bounded chained Tool Calling
    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=os.getenv("MODEL"), messages=messages, tools=TOOLS
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            print("Assistant: ", msg.content or "")
            break
        log(f"Step {step + 1}: model requested {len(msg.tool_calls)} tool calls")

        for tool_call in msg.tool_calls:
            tool_response = run_tool(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_response),
                }
            )
    else:
        # In python, this else will run when the above bounded tool loop ends due to steps exhaustion.
        messages.append(
            {
                "role": "assistant",
                "content": "Sorry, I could not finish this within the steps limit.",
            }
        )
