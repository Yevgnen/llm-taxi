import abc
from collections.abc import AsyncGenerator
from typing import Any, ClassVar, Generic, TypeVar

from llm_taxi.conversation import Message


T = TypeVar("T")


class LLM(Generic[T], metaclass=abc.ABCMeta):
    """Abstract base class for Large Language Models (LLMs).

    This class provides a template for LLM implementations, including methods for converting
    messages and generating responses, both streaming and non-streaming.

    Methods:
        streaming_response(messages: list[Message], **kwargs) -> AsyncGenerator:
            Abstract method to be implemented by subclasses to generate a streaming response.

        response(messages: list[Message], **kwargs) -> str:
            Abstract method to be implemented by subclasses to generate a non-streaming response.
    """

    call_kwargs_mapping: ClassVar[dict[str, str]] = {}

    def _convert_messages(self, messages: list[Message]) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    async def streaming_response(
        self,
        messages: list[Message],
        **kwargs,
    ) -> AsyncGenerator:
        raise NotImplementedError

    @abc.abstractmethod
    async def response(self, messages: list[Message], **kwargs) -> str:
        raise NotImplementedError
