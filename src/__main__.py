from .models_loader import load_prompts, load_functions
import re
from llm_sdk.llm_sdk import Small_LLM_Model
from .generator import Constrainator


def main():
    try:
        loaded_prompt = load_prompts("data/input/function_calling_tests.json")
        loaded_functions = load_functions("data/input/functions_definition.json")
        model = Small_LLM_Model()
        # tokenised_functions = function_tokenisation(loaded_functions, model)
        constrainator = Constrainator(model, loaded_functions)
        for p in loaded_prompt:
            result = constrainator.ask(p)
            print(result)
        
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()