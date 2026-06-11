from .models_loader import load_prompts, load_functions


def main():
    try:
        loaded_prompt = load_prompts("data/input/function_calling_tests.json")
        print("\n---------prompts--------")
        print(loaded_prompt)
        print()
        loaded_functions = load_functions("data/input/functions_definition.json")
        print("\n---------functions--------")
        print(loaded_functions)
        print()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()