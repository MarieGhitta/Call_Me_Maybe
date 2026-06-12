from .models_loader import load_prompts, load_functions
from llm_sdk.llm_sdk import Small_LLM_Model


def main():
    try:
        loaded_prompt = load_prompts("data/input/function_calling_tests.json")
        loaded_functions = load_functions("data/input/functions_definition.json")
        model = Small_LLM_Model()
        for prompt in loaded_prompt:
            tokens = model.encode(prompt.prompt)
            print(tokens)
            fn_name = model.decode(tokens)
            print(fn_name)
        for fn in loaded_functions:
            tokens = model.encode(fn.name)
            print(tokens)
            fn_name = model.decode(tokens)
            print(fn_name)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()