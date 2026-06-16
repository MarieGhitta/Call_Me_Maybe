from .models_loader import load_prompts, load_functions
# import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model


def function_tokenisation(loaded_functions: list, model: Small_LLM_Model):
    tokenized_fn = {}
    for fn in loaded_functions:
        tokenized_fn[fn.name] = model.encode(fn.name).tolist()[0]
    return tokenized_fn


def compatible_functions(tokens_generated: list, fonctions_tokenised: dict):
    comp_fn = []
    for name, value in fonctions_tokenised.items():
        if tokens_generated == value[:len(tokens_generated)]:
            comp_fn.append(name)
    return comp_fn


def allowed_next_tokens(tokens_generated: list, fonctions_tokenised: dict):
    comp_fn = compatible_functions(tokens_generated, fonctions_tokenised)
    allowed_tok = set()
    for cf in comp_fn:
        allowed_tok.add(fonctions_tokenised[cf][len(tokens_generated)])
    return allowed_tok


def main():
    try:
        loaded_prompt = load_prompts("data/input/function_calling_tests.json")
        loaded_functions = load_functions("data/input/functions_definition.json")
        model = Small_LLM_Model()
        tokenised_functions = function_tokenisation(loaded_functions, model)
        print(compatible_functions([8822, 2891], tokenised_functions))
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()