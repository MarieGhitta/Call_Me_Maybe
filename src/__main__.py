from .json_loader import load_json


def main():
    try:
        prompt = load_json("data/input/function_calling_tests.json")
        print(prompt)
    except Exception:
        print("ERROR")


if __name__ == "__main__":
    main()