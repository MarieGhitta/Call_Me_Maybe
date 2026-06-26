from .models_loader import load_prompts, load_functions
import json
from llm_sdk.llm_sdk import Small_LLM_Model
from .generator import Constrainator


def main():
    try:
        loaded_prompt = load_prompts("data/input/function_calling_tests.json")
        loaded_functions = load_functions("data/input/functions_definition.json")
        model = Small_LLM_Model()
        constrainator = Constrainator(model, loaded_functions)
        results = []
        for p in loaded_prompt:
            results.append(constrainator.generate_function_call(p.prompt))
        with open("data/output/function_calling_results.json", "w") as file:
            json.dump(results, file, indent=4)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()