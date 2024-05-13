from llm_taxi.llm.openai import OpenAI


class DeepSeek(OpenAI):
    env_vars: dict[str, str] = {
        "api_key": "DEEPSEEK_API_KEY",
        "base_url": "DEEPSEEK_BASE_URL",
    }
