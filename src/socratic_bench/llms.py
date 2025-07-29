import gc
from typing import Union, Dict, List, Mapping, Any, Optional

import httpx
import ollama
import openai
import torch
from ollama import ResponseError
from openai import NotGiven, NOT_GIVEN
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

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


class HFLLM(LLM):

    def __init__(
            self,
            base_model: str,
            temperature: float = 0.15,
            max_new_tokens: int = 128,
            model_kwargs: Optional[Mapping[str, Any]] = None
    ):
        self._base_model = base_model
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._model_kwargs = model_kwargs or {"device_map": "cuda"}

        self.model = None
        self.tokenizer = None

    def query(self, messages: List[Dict[str, str]]) -> str:
        if getattr(self, "model", None) is None or self.tokenizer is None:
            self.load()

        raw_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([raw_prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, max_new_tokens=self._max_new_tokens, do_sample=True, temperature=self._temperature
        )
        generation = outputs[0, len(inputs['input_ids'][0]):]
        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        return decoded

    def load(self) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            self._base_model, **self._model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self._base_model)

    def healthcheck(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return self._base_model

    def unload(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
        gc.collect()
        torch.cuda.empty_cache()
