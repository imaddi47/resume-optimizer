import json
from logging import getLogger
from typing import Any

import openai
from pydantic import BaseModel

from app.exceptions import ProviderAuthError, ProviderRateLimitError, ProviderModelError, ProviderError
from app.services.ai.providers.base import LLMProvider, ModelInfo

logger = getLogger(__name__)


def _schema_to_json_instruction(schema: type[BaseModel]) -> str:
    """Convert a Pydantic model to a JSON schema instruction string."""
    return (
        "\n\nYou MUST respond with valid JSON matching this schema exactly:\n"
        f"```json\n{json.dumps(schema.model_json_schema(), indent=2)}\n```\n"
        "Respond ONLY with the JSON object, no other text."
    )


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model_id: str, base_url: str | None = None):
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = base_url
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.AsyncOpenAI(**kwargs)

    async def list_models(self) -> list[ModelInfo]:
        try:
            result = await self.client.models.list()
            models = []
            for m in result.data:
                models.append(ModelInfo(
                    id=m.id,
                    name=m.id,
                    supports_structured_output="gpt" in m.id.lower() or "o1" in m.id.lower() or "o3" in m.id.lower(),
                ))
            return sorted(models, key=lambda x: x.id)
        except openai.AuthenticationError as e:
            raise ProviderAuthError(f"OpenAI authentication failed: {e}") from e
        except Exception as e:
            raise ProviderError(f"Failed to list OpenAI models: {e}") from e

    async def generate(
        self,
        system_prompt: str,
        inputs: list[str | dict[str, Any]],
        *,
        structured_output_schema: type[BaseModel] | None = None,
        temperature: float = 0.1,
        timeout: int | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        user_parts = []
        for inp in inputs:
            if isinstance(inp, str):
                user_parts.append(inp)
            elif isinstance(inp, dict):
                user_parts.append("[Image content not supported for this provider]")
        messages.append({"role": "user", "content": "\n".join(user_parts)})

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
        }

        if structured_output_schema:
            try:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": structured_output_schema.__name__,
                        "schema": structured_output_schema.model_json_schema(),
                        "strict": False,
                    },
                }
            except Exception:
                messages[0]["content"] += _schema_to_json_instruction(structured_output_schema)

        try:
            response = await self.client.chat.completions.create(
                **kwargs, timeout=timeout if timeout else openai.NOT_GIVEN,
            )
            text = response.choices[0].message.content
            if not text or not text.strip():
                raise ProviderError(
                    f"Model {self.model_id} returned empty response. "
                    "Try a different model."
                )
            return text.strip()
        except openai.AuthenticationError as e:
            raise ProviderAuthError(f"OpenAI authentication failed: {e}") from e
        except openai.RateLimitError as e:
            raise ProviderRateLimitError(f"OpenAI rate limit: {e}") from e
        except openai.NotFoundError as e:
            raise ProviderModelError(f"OpenAI model not found: {e}") from e
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"OpenAI error: {e}") from e
