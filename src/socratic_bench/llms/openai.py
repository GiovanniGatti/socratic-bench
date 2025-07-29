from typing import Union, Dict, List

import openai
from openai import NotGiven, NOT_GIVEN

from socratic_bench.agents import LLM


class OpenAILLM(LLM):

    def __init__(self, model: str, client: openai.OpenAI, temperature: Union[float, NotGiven] = NOT_GIVEN):
        self._model = model
        self._client = client
        self._temperature = temperature

    def query(self, messages: List[Dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=self._temperature  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""

    def healthcheck(self) -> None:
        try:
            models = self._client.models.list()
        except openai.AuthenticationError as e:
            raise ValueError("Unable to authenticate at OpenAI. Check if key is valid.", e)

        available_models = [m.id for m in models]
        if self._model not in available_models:
            raise ValueError(f"Invalid model. Expected one of {available_models}")

    @property
    def model_name(self) -> str:
        return self._model
