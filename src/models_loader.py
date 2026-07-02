from .models import Prompt, FunctionDefinition
from .json_loader import load_json


def load_prompts(file: str) -> list[Prompt]:
    """loading prompts in list format.

    Args:
        file (str): path of the file.

    Returns:
        list[Prompt]: liste with all the prompts.
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
    """loading functions in list format.

    Args:
        file (str): path of the file.

    Returns:
        list[FunctionDefinition]: liste with all the functions.
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
