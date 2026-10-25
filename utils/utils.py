import os

START_DEBUG_SEQUENCE = "\033[90m"
END_DEBUG_SEQUENCE = "\033[0m"


def log(text):
    if os.getenv("DEBUG") == "True":
        print(f" {START_DEBUG_SEQUENCE}{text}{END_DEBUG_SEQUENCE}")
