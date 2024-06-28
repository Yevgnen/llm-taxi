from collections.abc import AsyncGenerator
from typing import cast

from together.types import ChatCompletionResponse

from llm_taxi.clients.together import Together as TogetherClient
from llm_taxi.conversation import Message
from llm_taxi.llms.openai import streaming_response


class Together(TogetherClient):
    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        return [
            {
                "role": message.role.value,
                "content": message.content,
            }
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
        response = cast(ChatCompletionResponse, response)

        if (choices := response.choices) and (data := choices[0]):
            if (message := data.message) and (content := message.content):
                return content

        return ""
