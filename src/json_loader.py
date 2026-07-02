"""Utility functions for loading JSON files."""

import json
from typing import Any


def load_json(file: str) -> Any:
    """Load and parse a JSON file.

    Args:
        file (str): Path to the JSON file.

    Raises:
        ValueError: If the file does not exist.
        ValueError: If the file is empty.
        ValueError: If the JSON content is invalid.

    Returns:
        Any: Parsed JSON content.
    """
    try:
        with open(file, "r") as json_file:
            json_content = json_file.read()
            if not json_content:
                raise ValueError("ERROR: empty file.")
            try:
                json_data = json.loads(json_content)
            except json.JSONDecodeError:
                raise ValueError("ERROR: invalid JSON.")
    except FileNotFoundError:
        raise ValueError("ERROR: File does not exist.")
    return json_data
