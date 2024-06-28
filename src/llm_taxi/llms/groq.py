from collections.abc import AsyncGenerator
from typing import Any

from groq.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from groq.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from groq.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from llm_taxi.clients.groq import Groq as GroqClient
from llm_taxi.conversation import Message, Role
from llm_taxi.llms.openai import streaming_response

_PARAM_TYPES: dict[Role, type] = {
    Role.User: ChatCompletionUserMessageParam,
    Role.Assistant: ChatCompletionAssistantMessageParam,
    Role.System: ChatCompletionSystemMessageParam,
}


class Groq(GroqClient):
    def _convert_messages(self, messages: list[Message]) -> list[Any]:
        return [
            _PARAM_TYPES[message.role](
                role=message.role.value,
                content=message.content,
            )
            for message in messages
        ]

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
