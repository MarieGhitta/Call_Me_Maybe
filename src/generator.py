import numpy as np
from math import inf
from .json_loader import load_json


class OutputJSON:
    def __init__(self):
        self.put = []
        self.buf = []

    def append(self, data):
        self.put.extend(data)

    def add(self, data):
        self.buf.extend(data)

    def commit(self):
        self.put.extend(self.buf)
        self.buf = []


class Constrainator:
    def __init__(self, model, loaded_functions):
        self.model = model
        self.loaded_functions = loaded_functions
        self.tokenized_functions = self._function_tokenisation()
        self.comma_token = self.model.encode(",").tolist()[0][0]
        self.bracket_token = self.model.encode("}").tolist()[0][0]
        self.vocab = load_json(self.model.get_path_to_vocab_file())
        self.tokens = self.TokensCollection(self.vocab)
        self.true_token = self.model.encode("true").tolist()[0][0]
        self.false_token = self.model.encode("false").tolist()[0][0]
    

    def _function_tokenisation(self) -> dict:
        tokenized_fn = {}
        for fn in self.loaded_functions:
            tokenized_fn[fn.name] = self.model.encode(fn.name).tolist()[0]
        return tokenized_fn

    def _compatible_tokens(self, tokens_generated) -> list[str]:
        comp_fn = []
        for name, value in self.tokenized_functions.items():
            if tokens_generated == value[:len(tokens_generated)]:
                comp_fn.append(name)
        return comp_fn

    def _allowed_next_tokens(self, generated_tokens) -> set:
        comp_fn = self._compatible_tokens(generated_tokens)
        allowed_tok = set()
        for cf in comp_fn:
            token_list = self.tokenized_functions[cf]
            if len(token_list) > len(generated_tokens):
                allowed_tok.add(
                    token_list[len(generated_tokens)])
        return allowed_tok

    def _select_function(self, out, user_prompt):
        generated_tokens = []
        prompt = """
                You are a function selection system.
                Your task is to choose exactly one function that best matches
                the user request.
                Rules:
                - Choose only from the provided functions
                - Use function descriptions to understand semantic meaning
                - Do not invent functions
                - Select the single best matching function
                - Output only the function name
                Available functions:
                """
        for fn in self.loaded_functions:
            prompt += f"{fn.name}: {fn.description}\n"
        prompt += f"\nUser request: {user_prompt}\n"
        prompt += "Selected function:"
        tokenised_prompt = self.model.encode(prompt).tolist()[0]
        while True:
            entry_model = tokenised_prompt + out.put + generated_tokens
            logits = np.array(
                self.model.get_logits_from_input_ids(entry_model))
            allowed = self._allowed_next_tokens(generated_tokens)
            mask = np.ones(len(logits), dtype=bool)
            mask[list(allowed)] = False
            logits[mask] = -inf
            best_token = int(np.argmax(logits))
            generated_tokens.append(best_token)
            comp_fn = self._compatible_tokens(generated_tokens)
            if len(comp_fn) == 1:
                remaining_fn = comp_fn[0]
                target_length = len(self.tokenized_functions[remaining_fn])
                if len(generated_tokens) == target_length:
                    break
        return comp_fn[0]

    class TokensCollection:
        def __init__(self, vocab):
            self.vocab = vocab
            self.len = len(vocab)
            self.numeric_chars = '.-0123456789'
            self.commas = self._tokensif(lambda token: "," in token)
            self.close_bracket = self._tokensif(
                lambda token: token.count("}") == 1)
            self.numbers = self._tokensif(
                lambda token: all(
                    char in self.numeric_chars for char in token))
            self.line_breaks = self._tokensif(lambda token: "\n" in token)
            self.quotes = self._tokensif(lambda token: '"' in token)

        def _tokensif(self, cond):
            result = []
            for token, token_id in self.vocab.items():
                if cond(token):
                    result.append(token_id)
            return result

    def _get_number(self, out, context_tokens, is_last):
        for _ in range(30):
            model_input = context_tokens + out.put + out.buf
            logits = np.array(
                self.model.get_logits_from_input_ids(model_input))
            mask = np.ones(len(logits), dtype=bool)
            mask[list(self.tokens.numbers)] = False
            if is_last:
                mask[self.tokens.close_bracket] = False
            else:
                mask[self.tokens.commas] = False
            logits[mask] = -inf
            best_token = int(np.argmax(logits))
            if (best_token in self.tokens.commas or
                    best_token in self.tokens.close_bracket):
                break
            out.add([best_token])
        else:
            raise ValueError("Number generation exceeded max tokens.")
        if not out.buf:
            raise ValueError("Could not generate a valid number.")
        generated_number = self.model.decode(out.buf)
        if "." in generated_number:
            value = float(generated_number)
        else:
            value = int(generated_number)
        out.commit()
        return value

    def _normalize_regex(self, regex):
        if regex.startswith("/") and regex.count("/") >= 2:
            last_slash = regex.rfind("/")
            flags = regex[last_slash + 1:]
            if all(char in "gim" for char in flags):
                regex = regex[1:last_slash]
        return regex

    def _get_string(self, out, context_tokens, is_last):
        for _ in range(50):
            model_input = context_tokens + out.put + out.buf
            logits = np.array(
                self.model.get_logits_from_input_ids(model_input))
            mask = np.zeros(len(logits), dtype=bool)
            mask[list(self.tokens.line_breaks)] = True
            mask[self.tokens.commas] = True
            mask[self.tokens.close_bracket] = True
            logits[mask] = -inf
            best_token = int(np.argmax(logits))
            best_token = int(np.argmax(logits))
            candidate = out.buf + [best_token]
            decoded = self.model.decode(candidate)
            if len(out.buf) > 0 and '"' in decoded:
                out.add([best_token])
                break
            out.add([best_token])
        else:
            raise ValueError("String generation exceeded max tokens.")
        if not out.buf:
            raise ValueError("Could not generate a valid string.")
        decoded = self.model.decode(out.buf)
        generated_string = decoded.split('"')[0]
        out.commit()
        return generated_string
    
    def _get_bool(self, out, context_tokens):
        generated_tokens = []
        model_input = context_tokens + out.put + out.buf
        logits = np.array(
            self.model.get_logits_from_input_ids(model_input))
        allowed = {self.true_token, self.false_token}
        mask = np.full(len(logits), -inf)
        for token in allowed:
            mask[token] = logits[token]
        best_token = int(np.argmax(mask))
        out.add([best_token])
        generated_tokens = self.model.decode(out.buf)
        out.commit()
        if "true" in generated_tokens:
            return True
        if "false" in generated_tokens:
            return False
        raise ValueError("Could not generate valid boolean")

    def _get_args(self, function_definition, out, context_tokens):
        generated_args = {}
        param_names = list(function_definition.parameters.keys())
        for i, param in enumerate(param_names):
            is_last = (i == len(param_names) - 1)
            param_definition = function_definition.parameters[param]
            param_type = param_definition["type"]
            if param_type == "number":
                arg_prefix = f'"{param}":'
                arg_tokens = self.model.encode(arg_prefix).tolist()[0]
                out.append(arg_tokens)
                generated_args[param] = self._get_number(out, context_tokens,
                                                         is_last)
            elif param_type == "string":
                arg_prefix = f'"{param}":"'
                arg_tokens = self.model.encode(arg_prefix).tolist()[0]
                out.append(arg_tokens)
                generated_value = self._get_string(out, context_tokens,
                                                   is_last)
                if param == "regex":
                    generated_value = self._normalize_regex(generated_value)
                generated_args[param] = generated_value
            elif param_type == "boolean":
                arg_prefix = f'"{param}":'
                arg_tokens = self.model.encode(arg_prefix).tolist()[0]
                out.append(arg_tokens)
                generated_args[param] = self._get_bool(out, context_tokens)
        return generated_args

    def generate_function_call(self, user_prompt):
        functions_description = ""
        for fn in self.loaded_functions:
            params = []
            for param_name, param_definition in fn.parameters.items():
                param_type = param_definition["type"]
                params.append(f"{param_name}: {param_type}")
            params_str = ", ".join(params)
            functions_description += f"{fn.name}({params_str})\n"
        context_prompt = f"""
                        User request:
                        {user_prompt}
                        Available functions:
                        {functions_description}
                        Return JSON in the form:
                        {{
                        "name": "function_name",
                        "arguments": {{
                            ...
                        }}
                        }}
                        Output:
                        """
        context_tokens = self.model.encode(context_prompt).tolist()[0]
        out = OutputJSON()
        selected_function = self._select_function(out, user_prompt)
        json_prefix = ('{"name":' + selected_function + '", "arguments":{')
        prefix_tokens = self.model.encode(json_prefix).tolist()[0]
        out.append(prefix_tokens)
        for fn in self.loaded_functions:
            if fn.name == selected_function:
                args = self._get_args(fn, out, context_tokens)
                result = {
                    "prompt": user_prompt,
                    "function": fn.name,
                    "parameters": args
                }
                return result
        raise ValueError("No function selected")
