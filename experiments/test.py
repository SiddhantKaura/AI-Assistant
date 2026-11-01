import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from tools import TOOLS
from types import SimpleNamespace

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    # Not needed as we are hitting local LLM, adding dummy key since OpenAI has made it non null
    api_key=os.getenv("API_KEY"),
)

messages = [
    {
        "role": "system",
        "content": "Act as a straightforward assistant who replies in 40 words.",
    },
    {"role": "user", "content": "What is Weather in Delhi and Pune ?"},
]


stream = client.chat.completions.create(
    model=os.getenv("MODEL"), messages=messages, tools=TOOLS, stream=True
)

tool_calls_dict = {}
# SimpleNamespace(id="", function=SimpleNamespace(name="", arguments=""))

for chunk in stream:
    delta = chunk.choices[0].delta
    content = delta.content
    if content:
        print(content, end="", flush=True)
        time.sleep(0.2)
    for tc in delta.tool_calls or []:
        obj = tool_calls_dict.setdefault(
            tc.index,
            SimpleNamespace(id="", function=SimpleNamespace(name="", arguments="")),
        )
        if tc.id:
            obj.id += tc.id
        if tc.function:
            if tc.function.name:
                obj.function.name += tc.function.name
            if tc.function.arguments:
                obj.function.arguments += tc.function.arguments

tool_calls = []
for key in sorted(tool_calls_dict):
    tool_calls.append(tool_calls_dict[key])

print(tool_calls)
