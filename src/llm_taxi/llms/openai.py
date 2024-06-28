from collections.abc import AsyncGenerator, Iterable
from typing import Any

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from llm_taxi.clients.openai import OpenAI as OpenAIClient
from llm_taxi.conversation import Message, Role
from llm_taxi.llms.base import LLM


async def streaming_response(response: Any) -> AsyncGenerator:
    async for chunk in response:
        if content := chunk.choices[0].delta.content:
            yield content


_PARAM_TYPES: dict[Role, type] = {
    Role.User: ChatCompletionUserMessageParam,
    Role.Assistant: ChatCompletionAssistantMessageParam,
    Role.System: ChatCompletionSystemMessageParam,
}


class OpenAI(OpenAIClient, LLM):
    def _convert_messages(
        self,
        messages: list[Message],
    ) -> Iterable[ChatCompletionMessageParam]:
        return (
            _PARAM_TYPES[message.role](role=message.role.value, content=message.content)
            for message in messages
        )

    async def streaming_response(
        self,
        messages: list[Message],
        **kwargs,
    ) -> AsyncGenerator:
        response = await self.client.chat.completions.create(
            messages=self._convert_messages(messages),
            stream=True,
            **self._get_call_kwargs(**kwargs),
        )

        return streaming_response(response)

    async def response(self, messages: list[Message], **kwargs) -> str:
        response = await self.client.chat.completions.create(
            messages=self._convert_messages(messages),
            **self._get_call_kwargs(**kwargs),
        )

        if content := response.choices[0].message.content:
            return content

        return ""
