from .models_loader import load_prompts, load_functions
import numpy as np
# from llm_sdk.llm_sdk import Small_LLM_Model


def main():
    try:
        loaded_prompt = load_prompts("data/input/function_calling_tests.json")
        loaded_functions = load_functions("data/input/functions_definition.json")
        # model = Small_LLM_Model()
        
        # for prompt in loaded_prompt:
        #     tokens = model.encode(prompt.prompt)
        #     print(tokens)
        #     fn_name = model.decode(tokens)
        #     print(fn_name)
        # get_logits_from_input_ids()
        # for fn in loaded_functions:
        #     tokens = model.encode(fn.name)
        #     print(tokens)
        #     fn_name = model.decode(tokens)
        #     print(fn_name)
        
        #i = 0
        #test = "Choose the most fitting function with this instruction"
        #tokens = model.encode(test)
        #while i < 20:
        #    tokens = model.encode(test)
        #    logits = model.get_logits_from_input_ids(tokens.tolist()[0])
        #    valeur_max = max(logits)
        #    indice = logits.index(valeur_max)
        #    premslogits = model.decode([indice])
        #    # print(f"{test} {premslogits}")
        #    test = test + premslogits
        #    i += 1
        #print(test)


    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()