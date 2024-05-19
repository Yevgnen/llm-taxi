from typing import Any, ClassVar

from llm_taxi.embeddings.base import Embedding


class OpenAIEmbedding(Embedding):
    env_vars: ClassVar[dict[str, str]] = {
        "api_key": "OPENAI_API_KEY",
    }

    def _init_client(self, **kwargs) -> Any:
        from openai import AsyncClient

        return AsyncClient(**kwargs)

    async def embed_text(self, text: str, **kwargs) -> list[float]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            **kwargs,
        )

        return response.data[0].embedding

    async def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            **kwargs,
        )

        return [x.embedding for x in response.data]
