import os
from collections.abc import Mapping

from llm_taxi.llm import LLM, Anthropic, Google, Groq, OpenAI, Together

MODEL_CLASSES: Mapping[str, type[LLM]] = {
    "openai": OpenAI,
    "google": Google,
    "together": Together,
    "groq": Groq,
    "anthropic": Anthropic,
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
    provider, model = model.split(":", 1)
    if not (model_class := MODEL_CLASSES.get(provider)):
        msg = f"Unknown LLM provider: {provider}"
        raise ValueError(msg)

    env_vars = model_class.env_vars
    env_var_values: dict[str, str] = {
        k: param if (param := locals().get(k)) is not None else _get_env(v)
        for k, v in env_vars.items()
    }

    return model_class(
        model=model,
        **env_var_values,
        call_kwargs=call_kwargs,
        **client_kwargs,
    )
