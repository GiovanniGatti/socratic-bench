import gc
from typing import Dict, List, Mapping, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

from socratic_bench.agents import LLM


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
