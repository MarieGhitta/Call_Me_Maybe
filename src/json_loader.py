import json
from typing import Any


def load_json(file: str) -> Any:
    try:
        with open(file, "r") as json_file:
            json_content = json_file.read()
            if not json_content:
                raise ValueError("ERROR: empty file.")
            try:
                json_data = json.loads(json_content)
            except json.JSONDecodeError:
                raise ValueError("ERROR: invalid JSON.")
            if not json_data:
                raise ValueError("ERROR: empty JSON.")
    except FileNotFoundError:
        raise ValueError("ERROR: File does not exist.")
    return json_data
