from typing import ClassVar

from google import generativeai as genai
from google.generativeai import GenerativeModel

from llm_taxi.clients.base import Client


class Google(Client[GenerativeModel]):
    env_vars: ClassVar[dict[str, str]] = {
        "api_key": "GOOGLE_API_KEY",
    }

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        call_kwargs: dict | None = None,
        **client_kwargs,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            call_kwargs=call_kwargs,
            **client_kwargs,
        )
        genai.configure(api_key=api_key, **client_kwargs)

        self._call_kwargs.pop("model", None)

    def _init_client(self, **kwargs) -> GenerativeModel:
        kwargs = {k: v for k, v in kwargs.items() if k not in {"api_key", "base_url"}}

        return GenerativeModel(self.model, **kwargs)
