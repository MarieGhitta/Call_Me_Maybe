from .json_loader import load_json
from .models import Prompt, FunctionDefinition


def main():
    try:
        prompt_data = load_json("data/input/function_calling_tests.json")
        data_obj = Prompt(**prompt_data[0])
        print(data_obj)
        print()
        prompt_fn = load_json("data/input/functions_definition.json")
        fn_obj = FunctionDefinition(**prompt_fn[0])
        print(fn_obj)
        print()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()