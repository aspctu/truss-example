from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.bettertransformer import BetterTransformer
import torch

class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self._tokenizer = None

    def load(self):
        self._model = AutoModelForCausalLM.from_pretrained(
            "aspctu/palmyra-20b-fp16",
            use_auth_token=self._secrets["hf_api_key"],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self._model = BetterTransformer.transform(self._model)
        self._tokenizer = AutoTokenizer.from_pretrained(
            "aspctu/palmyra-20b-fp16",
            use_auth_token=self._secrets["hf_api_key"]
        )

    def predict(self, request: Any) -> Any:
        prompt = request.pop("prompt")
        tokenized_text = self._tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
        generated_text = self._model.generate(tokenized_text, **request)
        response = self._tokenizer.decode(generated_text[0], skip_special_tokens=True)
        return {"completion" : response}
