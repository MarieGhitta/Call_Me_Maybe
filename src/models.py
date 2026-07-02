"""Pydantic models used by the project."""

from pydantic import BaseModel, field_validator
from typing import Any


class Prompt(BaseModel):
    """Represent a user prompt to process with the LLM."""

    prompt: str

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Validate prompt with pydantic.

        Args:
            value (str): Prompt string.
        Raises:
            ValueError: If the prompt is empty or contains only spaces.
        Returns:
            str: Validated prompt.
        """
        if not value.strip():
            raise ValueError("Prompt cannot be empty")
        return value


class FunctionDefinition(BaseModel):
    """Represent a callable function definition.

    Attributes:
        name: Function name.
        description: Human-readable function description.
        parameters: Function parameters schema.
        returns: Return type schema.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]
