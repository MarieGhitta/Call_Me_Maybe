"""Entry point for the Call Me Maybe project."""


from .models_loader import load_prompts, load_functions
import json
from llm_sdk.llm_sdk import Small_LLM_Model
from .generator import Constrainator
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for
            function definitions, input prompts, and output file.
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
    """Run the function-calling pipeline.

    Loads input files, initializes the LLM and constrained decoder,
    generates function calls for each prompt, and writes the results
    to the output JSON file.

    Raises:
        Exception: Prints any unexpected runtime error.
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
