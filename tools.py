from datetime import datetime
from utils.utils import log
import requests
import json


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=4"
    response = requests.get(url, timeout=10)
    # Raise error if other than 200 status code.
    response.raise_for_status()
    return response.text


# TOOLS = [{}, {}, {}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns current date and time as a formatted string.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "Returns weather at a particular location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "name of the city to get weather for.",
                    }
                },
                "required": ["city"],
            },
        },
    },
]

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
}


def run_tool(tool_call):
    tool_call_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments or {})
    try:
        log(f"⚙️ {tool_call_name}({tool_args})")
        # ** -> Dictionary Unpacking operator converts dict to attributes
        tool_response = FUNCTION_NAME_TO_FUNCTION_MAP.get(tool_call_name)(**tool_args)
    except Exception as e:
        tool_response = f"Error while running {tool_call_name}:{e}"
    log(f"  ⎆  {tool_response}")

    return tool_response
