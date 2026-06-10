from .json_loader import load_json


def main():
    try:
        prompt = load_json("data/input/function_calling_tests.json")
        print(prompt)
        print()
        prompt = load_json("data/input/functions_definition.json")
        print(prompt)
        print()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()