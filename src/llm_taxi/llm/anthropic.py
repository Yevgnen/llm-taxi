from collections.abc import AsyncGenerator
from typing import Any, Literal, cast

from anthropic import AsyncAnthropic
from anthropic._types import NOT_GIVEN, NotGiven
from anthropic.types import MessageParam

from llm_taxi.conversation import Conversation, Role
from llm_taxi.llm.base import LLM


class Anthropic(LLM):
    env_vars: dict[str, str] = {
        "api_key": "ANTHROPIC_API_KEY",
    }

    def _init_client(self, **kwargs) -> Any:
        return AsyncAnthropic(**kwargs)

    def _convert_messages(self, conversation: Conversation) -> list[MessageParam]:
        return [
            MessageParam(
                role=cast(Literal["user", "assistant"], message.role.value),
                content=message.content,
            )
            for message in conversation.messages
            if message.role in {Role.User, Role.Assistant}
        ]

    def _get_system_message_content(
        self,
        conversation: Conversation,
    ) -> str | NotGiven:
        if message := next(
            (x for x in reversed(conversation.messages) if x.role == Role.System),
            NOT_GIVEN,
        ):
            return message.content

        return NOT_GIVEN

    async def _streaming_response(self, response):
        async for chunk in response:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text

    async def streaming_response(
        self,
        conversation: Conversation,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator:
        system_message = self._get_system_message_content(conversation)
        messages = self._convert_messages(conversation)

        response = await self.client.messages.create(
            system=system_message,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            **self._get_call_kwargs(**kwargs),
        )

        return self._streaming_response(response)

    async def response(
        self,
        conversation: Conversation,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        system_message = self._get_system_message_content(conversation)
        messages = self._convert_messages(conversation)

        response = await self.client.messages.create(
            system=system_message,
            messages=messages,
            max_tokens=max_tokens,
            **self._get_call_kwargs(**kwargs),
        )

        return response.content[0].text
