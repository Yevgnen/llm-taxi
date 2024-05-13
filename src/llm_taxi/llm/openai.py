from collections.abc import AsyncGenerator
from typing import Any

from llm_taxi.conversation import Conversation
from llm_taxi.llm import LLM


class OpenAI(LLM):
    env_vars: dict[str, str] = {
        "api_key": "OPENAI_API_KEY",
    }

    def _init_client(self, **kwargs) -> Any:
        from openai import AsyncClient

        return AsyncClient(**kwargs)

    async def _streaming_response(self, response: Any) -> AsyncGenerator:
        async for chunk in response:
            if content := chunk.choices[0].delta.content:
                yield content

    async def streaming_response(
        self,
        conversation: Conversation,
        **kwargs,
    ) -> AsyncGenerator:
        messages = self._convert_messages(conversation)

        response = await self.client.chat.completions.create(
            messages=messages,
            stream=True,
            **self._get_call_kwargs(**kwargs),
        )

        return self._streaming_response(response)

    async def response(self, conversation: Conversation, **kwargs) -> str:
        messages = self._convert_messages(conversation)

        response = await self.client.chat.completions.create(
            messages=messages,
            **self._get_call_kwargs(**kwargs),
        )

        if content := response.choices[0].message.content:
            return content

        return ""
