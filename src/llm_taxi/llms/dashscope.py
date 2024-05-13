from llm_taxi.llms.openai import OpenAI


class DashScope(OpenAI):
    env_vars: dict[str, str] = {
        "api_key": "DASHSCOPE_API_KEY",
        "base_url": "DASHSCOPE_BASE_URL",
    }