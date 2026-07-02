"""Functions for loading prompts and function definitions."""

from .models import Prompt, FunctionDefinition
from .json_loader import load_json


def load_prompts(file: str) -> list[Prompt]:
    """Load prompts from a JSON file.

    Invalid or empty prompts are skipped.

    Args:
        file (str): Path to the prompts file.

    Returns:
        list[Prompt]: Valid prompt objects.
    """
    prompt_data = load_json(file)
    prompts = []
    for prompt in prompt_data:
        try:
            prompts.append(Prompt(**prompt))
        except Exception:
            print("Skipping empty prompt.")
    return prompts


def load_functions(file: str) -> list[FunctionDefinition]:
    """Load function definitions from a JSON file.

    Invalid function definitions are skipped.

    Args:
        file (str): Path to the function definitions file.

    Returns:
        list[FunctionDefinition]: Valid function definitions.
    """
    fn_def = load_json(file)
    functions = []
    for fn in fn_def:
        try:
            fn_obj = FunctionDefinition(**fn)
            functions.append(fn_obj)
        except Exception:
            print("Incorrect format function.")
    return functions
