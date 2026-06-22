from .models_loader import load_prompts, load_functions
# import numpy as np
from llm_sdk.llm_sdk import Small_LLM_Model


def function_tokenisation(loaded_functions: list, model: Small_LLM_Model) -> dict:
    tokenized_fn = {}
    for fn in loaded_functions:
        tokenized_fn[fn.name] = model.encode(fn.name).tolist()[0]
    return tokenized_fn


def parameters_tokenisation(found_fn, model: Small_LLM_Model) -> dict:
    tokenized_args = {}
    for p in found_fn.parameters:
        tokenized_args[p] = model.encode(p).tolist()[0]
    return tokenized_args


def compatible_tokens(tokens_generated: list, functions_tokenised: dict) -> list:
    comp_fn = []
    for name, value in functions_tokenised.items():
        if tokens_generated == value[:len(tokens_generated)]:
            comp_fn.append(name)
    return comp_fn


def allowed_next_fn_tokens(tokens_generated: list, functions_tokenised: dict) -> set:
    comp_fn = compatible_tokens(tokens_generated, functions_tokenised)
    allowed_tok = set()
    for cf in comp_fn:
        allowed_tok.add(functions_tokenised[cf][len(tokens_generated)])
    return allowed_tok


def select_fn(model: Small_LLM_Model, 
              user_prompt: str, 
              loaded_functions: list, 
              tokenized_functions):
    generate_tokens = []
    entry_model = []
    prompt = "Choose the most appropriate function for the user request.\n"
    for fn in loaded_functions:
        prompt += f"\nFunction: {fn.name}"
        prompt += f"\nDescription: {fn.description}\n"
    prompt += f"\nUser Request: {user_prompt}\n"
    prompt += "\nSelected function:"
    tokenised_prompt = model.encode(prompt).tolist()[0]
    comp_fn = compatible_tokens(generate_tokens, tokenized_functions)
    while True:
        entry_model = tokenised_prompt + generate_tokens
        logits_fn = model.get_logits_from_input_ids(entry_model)
        allowed = allowed_next_fn_tokens(generate_tokens, tokenized_functions)
        best_tok = best_token_authorized(allowed, logits_fn)
        generate_tokens.append(best_tok)
        comp_fn = compatible_tokens(generate_tokens, tokenized_functions)
        if len(generate_tokens) == len(tokenized_functions[comp_fn[0]]) and len(comp_fn) == 1:
            break
    return comp_fn[0]


def selected_fn(found_fn: str, loaded_functions: list):
    for fn in loaded_functions:
        if fn.name == found_fn:
            return fn


def allowed_next_keys_tokens(model: Small_LLM_Model,
                             tokens_generated: list,
                             fn, allowed: set):
    open_brace = model.encode('{').tolist()[0][0]
    quote_tok = model.encode('"').tolist()[0][0]
    colon_tok = model.encode(':').tolist()[0][0]
    if not tokens_generated:
        allowed.add(open_brace)
        return
    elif tokens_generated == [open_brace]:
        allowed.add(quote_tok)
        return
    nb_quotes = tokens_generated.count(quote_tok)
    if nb_quotes == 2:
        allowed.add(colon_tok)
        return
    else:
        param_tokens = tokens_generated[2:]
        param_tokenized = (parameters_tokenisation(fn, model))
        comp_args = compatible_tokens(param_tokens, param_tokenized)
        for ca in comp_args:
            if param_tokens == param_tokenized[ca]:
                allowed.add(quote_tok)
            else:
                allowed.add(param_tokenized[ca][len(param_tokens)])


def allowed_next_value_tokens(model: Small_LLM_Model,
                             tokens_generated: list,
                             fn, allowed: set):
    

def allowed_next_args_tokens(model: Small_LLM_Model,
                             tokens_generated: list,
                             fn) -> set:
    allowed = set()
    allowed_next_keys_tokens(model,
                             tokens_generated,
                             fn, allowed)
    return allowed


def select_args(model: Small_LLM_Model,
                user_prompt: str,
                loaded_functions: list,
                found_fn: str):
    generate_tokens = []
    entry_model = []
    prompt = "Extract the arguments for the selected function.\n"
    fn = selected_fn(found_fn, loaded_functions)
    for p in fn.parameters:
        prompt += f"- {p}: {fn.parameters[p]['type']}\n"
    prompt += f"\nUser Request: {user_prompt}\n"
    prompt += "\nArguments:"
    tokenised_prompt = model.encode(prompt).tolist()[0]
    while True:
        entry_model = tokenised_prompt + generate_tokens
        logits_args = model.get_logits_from_input_ids(entry_model)


def best_token_authorized(allowed_tokens: set, logits: list):
    max_logits = float("-inf")
    max_tok = 0
    for tok in allowed_tokens:
        if logits[tok] > max_logits:
            max_logits = logits[tok]
            max_tok = tok
    return max_tok


def main():
    try:
        loaded_prompt = load_prompts("data/input/function_calling_tests.json")
        loaded_functions = load_functions("data/input/functions_definition.json")
        model = Small_LLM_Model()
        tokenised_functions = function_tokenisation(loaded_functions, model)
        # print(select_fn(model, "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'", loaded_functions, tokenised_functions))
        print(model.encode("{").tolist())
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()