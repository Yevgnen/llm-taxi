from typing import ClassVar

from openai import AsyncClient

from llm_taxi.clients.base import Client


class OpenAI(Client[AsyncClient]):
    env_vars: ClassVar[dict[str, str]] = {
        "api_key": "OPENAI_API_KEY",
    }

    def _init_client(self, **kwargs) -> AsyncClient:
        return AsyncClient(**kwargs)
