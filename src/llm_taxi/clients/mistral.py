from typing import ClassVar

from mistralai.async_client import MistralAsyncClient

from llm_taxi.clients.base import Client


class Mistral(Client[MistralAsyncClient]):
    env_vars: ClassVar[dict[str, str]] = {
        "api_key": "MISTRAL_API_KEY",
    }

    def _init_client(self, **kwargs) -> MistralAsyncClient:
        kwargs.pop("base_url", None)

        return MistralAsyncClient(**kwargs)
