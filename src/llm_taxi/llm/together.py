from typing import Any

from llm_taxi.llm import OpenAI


class Together(OpenAI):
    env_vars: dict[str, str] = {
        "api_key": "TOGETHER_API_KEY",
    }

    def _init_client(self, **kwargs) -> Any:
        from together import AsyncTogether

        return AsyncTogether(**kwargs)
