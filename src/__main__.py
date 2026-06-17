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


def allowed_next_tokens(tokens_generated: list, functions_tokenised: dict):
    comp_fn = compatible_functions(tokens_generated, functions_tokenised)
    allowed_tok = set()
    for cf in comp_fn:
        allowed_tok.add(functions_tokenised[cf][len(tokens_generated)])
    return allowed_tok


def select_fn(model: Small_LLM_Model, 
              user_prompt: str, 
              loaded_functions: list, 
              tokenized_functions):
    prompt = "Choose the most appropriate function for the user request.\n"
    for fn in loaded_functions:
        prompt += f"\nFunction: {fn.name}"
        prompt += f"\nDescription: {fn.description}\n"
    prompt += f"\nUser Request: {user_prompt}\n"
    prompt += "\nSelected function:"
    tokenised_prompt = model.encode(prompt).tolist()[0]
    logits_fn = model.get_logits_from_input_ids(tokenised_prompt)
    generate_tokens = []
    allowed = allowed_next_tokens(generate_tokens, tokenized_functions)
    for token in allowed:
        print(token, logits_fn[token])
    return allowed

def main():
    try:
        loaded_prompt = load_prompts("data/input/function_calling_tests.json")
        loaded_functions = load_functions("data/input/functions_definition.json")
        model = Small_LLM_Model()
        tokenised_functions = function_tokenisation(loaded_functions, model)
        compatible_fn = compatible_functions([8822, 2891], tokenised_functions)
        print(select_fn(model, "What is the sum of 2 and 3?", loaded_functions, tokenised_functions))

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()