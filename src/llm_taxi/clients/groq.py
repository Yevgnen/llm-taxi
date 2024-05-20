from typing import ClassVar

from groq import AsyncGroq

from llm_taxi.clients.base import Client


class Groq(Client[AsyncGroq]):
    env_vars: ClassVar[dict[str, str]] = {
        "api_key": "GROQ_API_KEY",
    }

    def _init_client(self, **kwargs) -> AsyncGroq:
        return AsyncGroq(**kwargs)
