from typing import Any, ClassVar

from mistralai.async_client import MistralAsyncClient

from llm_taxi.embeddings.base import Embedding


class MistralEmbedding(Embedding):
    env_vars: ClassVar[dict[str, str]] = {
        "api_key": "MISTRAL_API_KEY",
    }

    def _init_client(self, **kwargs) -> Any:
        kwargs.pop("base_url", None)

        return MistralAsyncClient(**kwargs)

    async def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings(model=self.model, input=text)

        return response.data[0].embedding

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings(model=self.model, input=texts)

        return [x.embedding for x in response.data]
