    for instruction in loaded_prompt:
            instruction = instruction.prompt # "What is the sum of 2 and 3?"
            functions = "\n".join(f"- {func.name}: {func.description}" for func in loaded_functions)
            print(prompt.prompt)
            print(functions)
            break
            # 1) Prompt: "Choose the most fitting function of the list for the given instruction."
            input_ids = model.encode(prefix + functions + instruction)
            # 'Choose the most fitting function of the list for the given instruction.
            # - fn_add_numbers: Add two numbers together and return their sum.
            # - fn_greet: Generate a greeting message for a person by name.
            # - fn_reverse_string: Reverse a string and return the reversed result.
            # - fn_get_square_root: Calculate the square root of a number.
            # - fn_substitute_string_with_regex: Replace all occurrences matching a regex pattern in a string.
            # Instruction: What is the sum of 2 and 3?
            # {"name": "'

            # ... {"name": "fn_add_numbers", "parameters": {"a": "'

            # 2) On fait une liste de candidats potentiels parmis les fonctions
            candidates = functions.copy()
            curr_ids = input_ids.copy()
            i = 0

            # 2bis) Si ce sont des nombres -> nbr_tokens = ...

            # 3) On genere le prochain token
            # [1005, 658 ...]
            for _ in range(100):
                logits = np.array(model.get_logits_from_input_ids(input_ids))
                # {1005: .89, 6674685: .65, 156: -.23, ...}
                mask = np.full(len(logits), -np.inf)
                # {1005: -inf, 6674685: -inf, 156: -inf, ... "add": .64, "greet": .89, "reverse": .3}

                valid_tokens = [f.name[i] for f in functions]

                mask[valid_tokens] = logits[valid_tokens]

                next_token = np.argmax(mask)
                input_ids += next_token

                # "Choose the most fitting function of the list for the given instruction.
                # - fn_add_numbers: Add two numbers together and return their sum.
                # - fn_greet: Generate a greeting message for a person by name.
                # - fn_reverse_string: Reverse a string and return the reversed result.
                # - fn_get_square_root: Calculate the square root of a number.
                # - fn_substitute_string_with_regex: Replace all occurrences matching a regex pattern in a string.
                # Instruction: What is the sum of 2 and 3?"
                # {"name": "fn_add_numbers"'

                candidates = [c for c in candidates if c[i] == next_token]

                if len(candidates) == 1:
                    break
                i += 1
            else:
                raise GenerationError("Max token limit exceeded")
        