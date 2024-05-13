from llm_taxi.llms.openai import OpenAI


class OpenRouter(OpenAI):
    env_vars: dict[str, str] = {
        "api_key": "OPENROUTER_API_KEY",
        "base_url": "OPENROUTER_BASE_URL",
    }