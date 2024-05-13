import itertools
from collections.abc import AsyncGenerator
from typing import Any

from llm_taxi.conversation import Conversation, Role
from llm_taxi.llms import LLM


class Google(LLM):
    env_vars: dict[str, str] = {
        "api_key": "GOOGLE_API_KEY",
    }

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        call_kwargs: dict | None = None,
        **client_kwargs,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            call_kwargs=call_kwargs,
            **client_kwargs,
        )

        from google import generativeai as genai

        genai.configure(api_key=api_key, **client_kwargs)

        self._call_kwargs.pop("model", None)

    def _init_client(self, **kwargs) -> Any:
        from google import generativeai as genai

        return genai.GenerativeModel(self.model)

    def _convert_messages(self, conversation: Conversation) -> list[Any]:
        role_mappping = {
            Role.System: "user",
            Role.User: "user",
            Role.Assistant: "model",
        }
        groups = itertools.groupby(
            conversation.messages,
            key=lambda x: role_mappping[Role(x.role)],
        )

        return [
            {
                "role": role,
                "parts": [x.content for x in parts],
            }
            for role, parts in groups
        ]

    async def _streaming_response(self, response):
        async for chunk in response:
            yield chunk.text

    async def streaming_response(
        self,
        conversation: Conversation,
        **kwargs,
    ) -> AsyncGenerator:
        from google import generativeai as genai

        messages = self._convert_messages(conversation)

        response = await self.client.generate_content_async(
            messages,
            stream=True,
            generation_config=genai.types.GenerationConfig(
                **self._get_call_kwargs(**kwargs),
            ),
        )

        return self._streaming_response(response)

    async def response(self, conversation: Conversation, **kwargs) -> str:
        from google import generativeai as genai

        messages = self._convert_messages(conversation)

        response = await self.client.generate_content_async(
            messages,
            generation_config=genai.types.GenerationConfig(
                **self._get_call_kwargs(**kwargs),
            ),
        )

        return response.text