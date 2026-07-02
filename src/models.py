from pydantic import BaseModel, field_validator


class Prompt(BaseModel):
    """Validate prompt with pydantic.

    Args:
        BaseModel: classe from pydantic.

    Raises:
        ValueError: empty prompt.

    Returns:
        _type_: str
    """
    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Check if a prompt is empty."""
        if not value.strip():
            raise ValueError("Prompt cannot be empty")
        return value


class FunctionDefinition(BaseModel):
    """Validate Functions with pydantic.

    Args:
        BaseModel : classe from pydantic.
    """
    name: str
    description: str
    parameters: dict
    returns: dict
