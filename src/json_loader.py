import json


def load_json(file: str) -> list[dict]:
    with open(file, "r") as json_file:
        json_content = json_file.read()
        if not json_content:
            raise ValueError("ERROR: empty file.")
        json_data = json.loads(json_content)
        if not json_data:
            raise ValueError("ERROR: empty json.")
    return json_data


    