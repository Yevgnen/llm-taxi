import os
from collections.abc import Mapping

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

MODEL_CLASSES: Mapping[str, type[LLM]] = {
    "openai": OpenAI,
    "google": Google,
    "together": Together,
    "groq": Groq,
    "anthropic": Anthropic,
    "mistral": Mistral,
    "perplexity": Perplexity,
    "deepinfra": DeepInfra,
    "deepseek": DeepSeek,
    "openrouter": OpenRouter,
    "dashscope": DashScope,
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
    provider, model = model.split(":", 1)
    if not (model_class := MODEL_CLASSES.get(provider)):
        msg = f"Unknown LLM provider: {provider}"
        raise ValueError(msg)

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
