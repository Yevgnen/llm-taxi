from collections.abc import AsyncGenerator
from typing import Any

from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage

from llm_taxi.conversation import Conversation
from llm_taxi.llm.openai import OpenAI


class Mistral(OpenAI):
    env_vars: dict[str, str] = {
        "api_key": "MISTRAL_API_KEY",
    }

    def _init_client(self, **kwargs) -> Any:
        kwargs.pop("base_url", None)

        return MistralAsyncClient(**kwargs)

    def _convert_messages(self, conversation) -> list[Any]:
        return [
            ChatMessage(role=x.role.value, content=x.content)
            for x in conversation.messages
        ]

    async def streaming_response(
        self,
        conversation: Conversation,
        **kwargs,
    ) -> AsyncGenerator:
        messages = self._convert_messages(conversation)

        response = self.client.chat_stream(
            messages=messages,
            **self._get_call_kwargs(**kwargs),
        )

        return self._streaming_response(response)

    async def response(self, conversation: Conversation, **kwargs) -> str:
        messages = self._convert_messages(conversation)

        response = await self.client.chat(
            messages=messages,
            **self._get_call_kwargs(**kwargs),
        )

        return response.choices[0].message.content
