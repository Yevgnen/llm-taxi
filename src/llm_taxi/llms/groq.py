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
from llm_taxi.llms.openai import OpenAI

_PARAM_TYPES: dict[Role, type] = {
    Role.User: ChatCompletionUserMessageParam,
    Role.Assistant: ChatCompletionAssistantMessageParam,
    Role.System: ChatCompletionSystemMessageParam,
}


class Groq(GroqClient, OpenAI):
    def _convert_messages(self, messages: list[Message]) -> list[Any]:
        return [
            _PARAM_TYPES[message.role](
                role=message.role.value,
                content=message.content,
            )
            for message in messages
        ]
