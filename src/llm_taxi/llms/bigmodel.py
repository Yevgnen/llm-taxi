from llm_taxi.clients.bigmodel import BigModel as BigModelClient
from llm_taxi.llms.openai import OpenAI


class BigModel(BigModelClient, OpenAI):
    pass
