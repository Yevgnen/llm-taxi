from typing import Any

from groq import AsyncGroq
from groq.types.chat.completion_create_params import Message

from llm_taxi.conversation import Conversation
from llm_taxi.llm.openai import OpenAI


class Groq(OpenAI):
    env_vars: dict[str, str] = {
        "api_key": "GROQ_API_KEY",
    }

    def _init_client(self, **kwargs) -> Any:
        return AsyncGroq(**kwargs)

    def _convert_messages(self, conversation: Conversation) -> list[Message]:
        return [
            Message(role=message.role.value, content=message.content)
            for message in conversation.messages
        ]
