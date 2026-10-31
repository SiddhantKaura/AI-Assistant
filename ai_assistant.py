# Standard library imports
import os

# Third party imports
from openai import OpenAI
from dotenv import load_dotenv

# Local modules
from tools import TOOLS, run_tool
from utils.utils import log  # From folder.file
from types import SimpleNamespace

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
        "content": "Act as my assistant and reply in less than 40 words.",
    }
)


def stream_response(messages):
    stream = client.chat.completions.create(
        model=os.getenv("MODEL"), messages=messages, tools=TOOLS, stream=True
    )
    isFirstChunk = True
    final_response = ""
    tool_calls_dict = {}
    tool_calls = []
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            delta_content = delta.content
            delta_tool_calls = delta.tool_calls
            if delta_content:
                if isFirstChunk:
                    print("Assistant: ", end="")
                    isFirstChunk = False
                print(delta_content, end="", flush=True)
                final_response += delta_content
            for tc in delta_tool_calls or []:
                tool_call_obj = tool_calls_dict.setdefault(
                    tc.index,
                    SimpleNamespace(
                        id=None, function=SimpleNamespace(name="", arguments="")
                    ),
                )
                if tc.id:
                    tool_call_obj.id = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_call_obj.function.name += tc.function.name
                    if tc.function.arguments:
                        tool_call_obj.function.arguments += tc.function.arguments
    except KeyboardInterrupt:
        print("\n{Stopped by You}")
        tool_calls_dict = {}

    for key in sorted(tool_calls_dict or {}):
        tool_calls.append(tool_calls_dict[key])
    return final_response, tool_calls


def build_tool_calls_payload(tool_calls):
    tool_calls_payload = []
    for tool_call in tool_calls:
        obj = {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }
        tool_calls_payload.append(obj)
    return tool_calls_payload


while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["quit", "bye", "exit"]:
        break
    messages.append({"content": user_input, "role": "user"})

    # Bounded chained Tool Calling
    for step in range(MAX_STEPS):
        final_response, tool_calls = stream_response(messages=messages)
        messages.append(
            {
                "role": "assistant",
                "content": final_response,
                "tool_calls": build_tool_calls_payload(tool_calls=tool_calls),
            }
        )
        if not tool_calls:
            break
        log(f"Step {step + 1}: model requested {len(tool_calls)} tool calls")

        for tool_call in tool_calls:
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
