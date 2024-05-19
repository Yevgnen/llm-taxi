import abc
from typing import Any, ClassVar


class Embedding(metaclass=abc.ABCMeta):
    env_vars: ClassVar[dict[str, str]] = {}

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

    @abc.abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    @abc.abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
