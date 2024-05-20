import os
from collections.abc import Mapping
from enum import Enum
from typing import Any, TypeVar, cast

from llm_taxi.embeddings import (
    Embedding,
    GoogleEmbedding,
    MistralEmbedding,
    OpenAIEmbedding,
)
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

EMBEDDING_CLASSES: Mapping[Provider, type[Embedding]] = {
    Provider.OpenAI: OpenAIEmbedding,
    Provider.Mistral: MistralEmbedding,
    Provider.Google: GoogleEmbedding,
}


T = TypeVar("T")


def _get_env(key: str) -> str:
    if (value := os.getenv(key)) is None:
        msg = f"Required environment variable `{key}` not found"
        raise KeyError(msg)

    return value


def _get_class_name_and_class(
    model: str,
    class_dict: Mapping[Provider, type[T]],
) -> tuple[str, type[T]]:
    provider_name, model = model.split(":", 1)

    try:
        provider = cast(Provider, Provider(provider_name))
    except ValueError as error:
        msg = f"Unknown LLM provider: {provider_name}"
        raise ValueError(msg) from error

    return model, class_dict[provider]


def _get_params(
    cls: type[LLM | Embedding],
    local_vars: dict[str, Any],
) -> dict[str, str]:
    env_var_values: dict[str, str] = {}
    for key, env_name in cls.env_vars.items():
        value = (
            params
            if (params := local_vars.get(key)) is not None
            else _get_env(env_name)
        )
        env_var_values[key] = value

    return env_var_values


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
    model, model_class = _get_class_name_and_class(model, MODEL_CLASSES)
    env_var_values = _get_params(model_class, locals())

    return model_class(
        model=model,
        **env_var_values,
        call_kwargs=call_kwargs,
        **client_kwargs,
    )


def embedding(
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    call_kwargs: dict | None = None,
    **client_kwargs,
) -> Embedding:
    """Initialize and return an instance of a specified embedding provider.

    Args:
        model (str): The model identifier in the format 'provider:model_name'.
        api_key (str | None, optional): The API key for authentication. Defaults to None.
        base_url (str | None, optional): The base URL for the API. Defaults to None.
        call_kwargs (dict | None, optional): Additional keyword arguments for the API call. Defaults to None.
        **client_kwargs: Additional keyword arguments for the embedding client initialization.

    Returns:
        Embedding: An instance of the specified embedding provider.

    Raises:
        ValueError: If the specified provider is unknown.
        KeyError: If a required environment variable is not found.
    """
    model, embedding_class = _get_class_name_and_class(model, EMBEDDING_CLASSES)
    env_var_values = _get_params(embedding_class, locals())

    return embedding_class(
        model=model,
        **env_var_values,
        call_kwargs=call_kwargs,
        **client_kwargs,
    )
