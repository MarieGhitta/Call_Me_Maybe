*This project has been created as part of the 42 curriculum by mghitta.*

# Call Me Maybe

## Description

**Call Me Maybe** is a Python project exploring **function calling with Large Language Models (LLMs)**.

The goal is to transform natural language prompts into structured function calls while guaranteeing **100% valid JSON output**, even when using a small language model such as **Qwen/Qwen3-0.6B**.

Unlike a traditional chatbot that directly answers a question, this program identifies:

- which function should be called
- which arguments should be passed to that function

Example:

User prompt:

```text
What is the sum of 40 and 2?
```

Expected output:

```json
{
    "name": "fn_add_numbers",
    "arguments": {
        "a": 40.0,
        "b": 2.0
    }
}
```

The system does **not execute the function**.  
It only generates the correct function call.

---

### Objectives

This project focuses on understanding:

- how LLM token generation works
- why structured output is difficult for small models
- how constrained decoding improves reliability
- how modern AI assistants perform function calling

---

# Instructions

Requirements:

- Python 3.10+
- uv

Install dependencies:

```bash
uv sync
```

Two input files:
- Functions definitions
- prompts

---

## Example Usage

Default execution:

```bash
uv run python -m src
```

Custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

This matches the execution format required by the subject.

---

# Design decisions

The project is organized into several modules.

```bash
.
├── src/
│   ├── __main__.py
│   ├── generator.py
│   ├── json_loader.py
│   ├── models.py
│   └── models_loader.py
│
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/
│
├── pyproject.toml
├── Makefile
└── README.md
```

---

## `models.py`

Contains Pydantic models used for validation.

### Prompt model

```python
class Prompt(BaseModel):
    prompt: str
```

Ensures each prompt is valid and non-empty.

### FunctionDefinition model

```python
class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict
    returns: dict
```

Each function definition includes:

- function name
- description
- parameters
- return type

Pydantic validation ensures malformed input files are rejected.

---

## `json_loader.py`

Responsible for safely loading JSON files.

It validates:

- file existence
- valid JSON syntax
- non-empty files

Errors are handled gracefully with explicit messages.

Examples:

```text
ERROR: File does not exist.
ERROR: invalid JSON.
ERROR: empty file.
```

---

## `models_loader.py`

Transforms raw JSON into Python objects.

Responsibilities:

- load prompts
- load function definitions
- instantiate Pydantic objects

This separates parsing from business logic.

---

## `generator.py`

This is the core of the project.

It contains the `Constrainator` class responsible for:

1. selecting the best function
2. generating valid arguments
3. enforcing JSON constraints

This module implements **constrained decoding**.

---

## `__main__.py`

Program entry point.

Responsibilities:

- parse CLI arguments
- load inputs
- initialize LLM
- run function generation
- write output file

The output directory is automatically created if missing.

---

## `functions_definition.json`

Contains all callable functions.

Example:

```json
[
    {
        "name": "fn_greet",
        "description": "Generate a greeting",
        "parameters": {
            "name": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    }
]
```

---

## `function_calling_tests.json`

Contains prompts to process.

Example:

```json
[
    {
        "prompt": "Greet Shrek"
    }
]
```

---

# Constrained Decoding

## Why is it needed?

Small LLMs frequently fail when asked to generate JSON directly.

Typical failures:

- missing quotes
- missing commas
- invalid braces
- wrong parameter types
- invented function names

Without constraints, output reliability is low.

---

## Algorithm explanation

Instead of allowing the model to generate any token, the decoder restricts generation to valid tokens only.

At each generation step:

1. retrieve model logits
2. compute allowed tokens
3. mask forbidden tokens using `-inf`
4. select best remaining token

Invalid tokens become impossible to generate.

---

### Function Selection

Function selection is performed using the LLM.

The model receives:

- available functions
- function descriptions
- user prompt

Example prompt:

```text
Available functions:
fn_add_numbers: Add two numbers
fn_greet: Generate greeting

User request:
Greet Shrek
```

The LLM generates the function name token by token.

Only tokens compatible with existing function names are allowed.

Example:

Available functions:

- fn_add_numbers
- fn_greet
- fn_reverse_string

If generated tokens already correspond to:

```text
fn_gr
```

Only tokens continuing toward `fn_greet` remain valid.

This guarantees the final function name belongs to the allowed list.

---

### Argument Generation

After function selection, arguments are generated according to parameter types.

Supported types:

- number
- string
- boolean

---

#### Number Generation

Allowed tokens include:

- digits
- decimal point
- minus sign
- separators

Example:

```json
"a": 42.0
```

---

#### String Generation

String generation stops when a closing quote is produced.

Forbidden tokens:

- line breaks
- invalid JSON separators

Example:

```json
"name": "Shrek"
```

---

#### Boolean Generation

Only two tokens are allowed:

- true
- false

Example:

```json
"enabled": true
```

---

#### Regex Handling

Some functions accept regex parameters.

Regex normalization is performed to clean model output.

Examples:

Input generated by model:

```text
/abc/g
```

Normalized output:

```text
abc
```

Escaped backslashes are also normalized.

This improves compatibility with expected test outputs.

---

### Output Format

Results are written as JSON in:

```bash
data/output/function_calling_results.json
```

Example:

```json
[
    {
        "name": "fn_reverse_string",
        "arguments": {
            "s": "hello"
        }
    }
]
```

Output always respects the expected schema.

---

### Error Handling

The program handles:

- invalid files
- empty prompts
- impossible generation
- malformed JSON
- invalid argument values

Graceful failure was a major design requirement.

The program should never crash unexpectedly.

---

# Performance analysis

I discussed with my peers to find good solution to have better speed. First of all, the use of numpy improves a lot the efficency.  
Another big problem with speed, was to run the programm on the school computer.  
I get a very good solution from a peer, which was to get the dependency installed in the sgoinfre folder, which has plenty of space.  

# Challenges faced

Main difficulties during development:

## Function selection

Ensuring the model selects functions using the LLM rather than heuristics required token-level constraints.

## Aguments selection

I add such difficulties to obtain the arguments. I had to do it in a totally different way of the function selection. I had to undertand how to use a mask and numpy. After this I used what I learned to improve the function selection.

## String generation

String decoding was difficult because generated strings could contain:

- escaped quotes
- regex characters
- commas

Special handling was required.

## Regex normalization

Regex prompts introduced escaping problems, especially backslashes.

---

# Testing strategy

I use the tests provided as soon as I can. When this tests worked, I add more complex situation to improve my  programm. 

---

# Ressouces:

## Documentation:

### LLM:
    [Deep Dive into LLMs like ChatGPT - Andrej Karpathy](https://www.youtube.com/watch?v=7xTGNNLPyMI)
### Json: 
    https://docs.python.org/fr/3.14/library/json.html
### Pydantic:
    https://pydantic.dev/docs/validation/latest/api/pydantic/base_model/
### Packaging:
    https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
### Regex:
    https://docs.python.org/fr/3/howto/regex.html
### Numpy:
    https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

## AI Usage:
    - explaining the different concepts of LLM.
    - creating tests.
    - README structure.