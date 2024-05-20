from typing import ClassVar

from together import AsyncTogether

from llm_taxi.clients.base import Client


class Together(Client[AsyncTogether]):
    env_vars: ClassVar[dict[str, str]] = {
        "api_key": "TOGETHER_API_KEY",
    }

    def _init_client(self, **kwargs) -> AsyncTogether:
        return AsyncTogether(**kwargs)
