from pydantic import BaseModel, field_validator


class Prompt(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Prompt cannot be empty")
        return value


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict
    returns: dict

