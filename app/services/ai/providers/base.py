from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class ModelInfo:
    id: str
    name: str
    supports_structured_output: bool = False


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return available models from this provider."""
        ...

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        inputs: list[str | dict[str, Any]],
        *,
        structured_output_schema: type[BaseModel] | None = None,
        temperature: float = 0.1,
        timeout: int | None = None,
    ) -> str:
        """Generate a response. Returns raw text.

        If structured_output_schema is provided and the provider supports native
        JSON mode, the provider should use it. Otherwise, the provider should
        inject the schema into the prompt as instructions.
        """
        ...
