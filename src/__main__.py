from .models_loader import load_prompts, load_functions
import json
from llm_sdk.llm_sdk import Small_LLM_Model
from .generator import Constrainator
import argparse
from pathlib import Path


def parse_args() -> set:
    """Parse input and output files.

    Returns:
        _type_: arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json")
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json")
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json")
    return parser.parse_args()


def main() -> None:
    """Main function to run the programm.
    """
    try:
        args = parse_args()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        loaded_functions = load_functions(args.functions_definition)
        loaded_prompt = load_prompts(args.input)
        model = Small_LLM_Model()
        constrainator = Constrainator(model, loaded_functions)
        results = []
        for p in loaded_prompt:
            results.append(constrainator.generate_function_call(p.prompt))
        with open(output_path, "w") as file:
            json.dump(results, file, indent=4)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
