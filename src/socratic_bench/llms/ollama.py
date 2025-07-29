from typing import Dict, List

import httpx
import ollama
from ollama import ResponseError

from socratic_bench.agents import LLM


class OllamaLLM(LLM):

    def __init__(self, model: str, client: ollama.Client, **options):
        self._model = model
        self._client = client
        self._options = options

    def query(self, messages: List[Dict[str, str]]) -> str:
        response = self._client.chat(model=self._model,
                                     messages=messages,
                                     options=self._options)
        return response["message"]["content"]

    def healthcheck(self) -> None:
        try:
            models = self._client.list()
        except httpx.ConnectError as e:
            raise ValueError("Unable to connect to Ollama server. Check server's address.", e)

        available_models = [m.model for m in models["models"]]

        if self._model not in available_models:
            try:
                print(f" === Pulling {self._model} from OllamaHub === ")
                self._client.pull(self._model)
            except ResponseError as e:
                raise ValueError("Model is unavailable. Unable to pull it.", e)

    @property
    def model_name(self) -> str:
        return self._model

    def unload(self) -> None:
        self._client.generate(self._model, keep_alive=0)
