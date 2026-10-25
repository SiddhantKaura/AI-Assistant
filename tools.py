from datetime import datetime
import requests


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_city_weather(city):
    url = f"https://wttr.in/{city}?format=4"
    response = requests.get(url, timeout=10)
    print("Response is ", response.text)
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
                    "city": {"type": "string", "description": "name of the city"}
                },
            },
        },
    },
]

FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
}
