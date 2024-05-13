import abc
from collections.abc import AsyncGenerator
from typing import Any

from llm_taxi.conversation import Conversation


class LLM(metaclass=abc.ABCMeta):
    env_vars: dict[str, str] = {}

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        call_kwargs: dict | None = None,
        **client_kwargs,
    ) -> None:
        if not call_kwargs:
            call_kwargs = {}

        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._call_kwargs = call_kwargs | {"model": self.model}
        self._client = self._init_client(
            api_key=self._api_key,
            base_url=self._base_url,
            **client_kwargs,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> Any:
        return self._client

    def _init_client(self, **kwargs) -> Any:
        raise NotImplementedError

    def _convert_messages(self, conversation: Conversation) -> list[Any]:
        return [
            {
                "role": message.role.value,
                "content": message.content,
            }
            for message in conversation.messages
        ]

    def _get_call_kwargs(self, **kwargs) -> dict:
        return self._call_kwargs | kwargs

    @abc.abstractmethod
    async def streaming_response(
        self,
        conversation: Conversation,
        **kwargs,
    ) -> AsyncGenerator:
        raise NotImplementedError

    @abc.abstractmethod
    async def response(self, conversation: Conversation, **kwargs) -> str:
        raise NotImplementedError