from enum import Enum

from pydantic import BaseModel, ConfigDict


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str


class Conversation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[Message]
