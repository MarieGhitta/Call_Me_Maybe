from .models import Prompt, FunctionDefinition
from .json_loader import load_json


def load_prompts(file: str) -> list[Prompt]:
    prompt_data = load_json(file)
    prompts = []
    for prompt in prompt_data:
        try:
            prompts.append(Prompt(**prompt))
        except Exception:
            print("Skipping empty prompt.")
    return prompts


def load_functions(file: str) -> list[FunctionDefinition]:
    fn_def = load_json(file)
    functions = []
    for fn in fn_def:
        fn_obj = FunctionDefinition(**fn)
        functions.append(fn_obj)
    return functions
