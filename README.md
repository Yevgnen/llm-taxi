# llm-taxi

## Installation

```shell
pip install llm-taxi
```

## Usage

Use as a library

```python
import asyncio

from llm_taxi.conversation import Message, Role
from llm_taxi.factory import embedding, llm


async def main():
    client = llm(model="openai:gpt-3.5-turbo")
    messages = [
        Message(role=Role.User, content="What is the capital of France?"),
    ]

    response = await client.response(messages)
    print(response)

    client = llm(model="mistral:mistral-small")
    messages = [
        Message(role=Role.User, content="Tell me a joke."),
    ]
    response = await client.streaming_response(messages)
    async for chunk in response:
        print(chunk, end="", flush=True)
    print()

    embedder = embedding("openai:text-embedding-ada-002")
    embeddings = await embedder.embed_text("Hello, world!")
    print(embeddings[:10])

    embedder = embedding("mistral:mistral-embed")
    embeddings = await embedder.embed_texts(["Hello, world!"])
    print(embeddings[0][:10])



if __name__ == "__main__":
    asyncio.run(main())
```

## Command line interface

```shell
llm-taxi --model openai:gpt-3.5-turbo-16k
```

See all supported arguments

```shell
llm-taxi --help
```

## Supported Providers

|  Provider  | LLM | Embedding |
| :--------: | :-: | :-------: |
| Anthropic  | ✅  |           |
| DashScope  | ✅  |           |
| DeepInfra  | ✅  |           |
|  DeepSeek  | ✅  |           |
|   Google   | ✅  |    ✅     |
|    Groq    | ✅  |           |
|  Mistral   | ✅  |    ✅     |
|   OpenAI   | ✅  |    ✅     |
| OpenRouter | ✅  |           |
| Perplexity | ✅  |           |
|  Together  | ✅  |           |
|  BigModel  | ✅  |           |

### Embeddings

- Google
- Mistral
- OpenAI
