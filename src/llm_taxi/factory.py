import os
from collections.abc import Mapping
from enum import Enum
from typing import cast

from llm_taxi.llms import (
    LLM,
    Anthropic,
    DashScope,
    DeepInfra,
    DeepSeek,
    Google,
    Groq,
    Mistral,
    OpenAI,
    OpenRouter,
    Perplexity,
    Together,
)


class Provider(Enum):
    OpenAI = "openai"
    Google = "google"
    Together = "together"
    Groq = "groq"
    Anthropic = "anthropic"
    Mistral = "mistral"
    Perplexity = "perplexity"
    DeepInfra = "deepinfra"
    DeepSeek = "deepseek"
    OpenRouter = "openrouter"
    DashScope = "dashscope"


MODEL_CLASSES: Mapping[Provider, type[LLM]] = {
    Provider.OpenAI: OpenAI,
    Provider.Google: Google,
    Provider.Together: Together,
    Provider.Groq: Groq,
    Provider.Anthropic: Anthropic,
    Provider.Mistral: Mistral,
    Provider.Perplexity: Perplexity,
    Provider.DeepInfra: DeepInfra,
    Provider.DeepSeek: DeepSeek,
    Provider.OpenRouter: OpenRouter,
    Provider.DashScope: DashScope,
}


def _get_env(key: str) -> str:
    if (value := os.getenv(key)) is None:
        msg = f"Required environment variable `{key}` not found"
        raise KeyError(msg)

    return value


def llm(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    call_kwargs: dict | None = None,
    **client_kwargs,
) -> LLM:
    """Initialize and return an instance of a specified LLM (Large Language Model) provider.

    Args:
        model (str): The model identifier in the format 'provider:model_name'.
        api_key (str | None, optional): The API key for authentication. Defaults to None.
        base_url (str | None, optional): The base URL for the API. Defaults to None.
        call_kwargs (dict | None, optional): Additional keyword arguments for the API call. Defaults to None.
        **client_kwargs: Additional keyword arguments for the LLM client initialization.

    Returns:
        LLM: An instance of the specified LLM provider.

    Raises:
        ValueError: If the specified provider is unknown.
        KeyError: If a required environment variable is not found.
    """
    provider_name, model = model.split(":", 1)

    try:
        provider = cast(Provider, Provider(provider_name))
    except ValueError as error:
        msg = f"Unknown LLM provider: {provider_name}"
        raise ValueError(msg) from error

    model_class = MODEL_CLASSES[provider]
    env_var_values: dict[str, str] = {}
    for key, env_name in model_class.env_vars.items():
        value = (
            params if (params := locals().get(key)) is not None else _get_env(env_name)
        )
        env_var_values[key] = value

    return model_class(
        model=model,
        **env_var_values,
        call_kwargs=call_kwargs,
        **client_kwargs,
    )
