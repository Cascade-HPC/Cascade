import time
import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .config import ModelConfig

@dataclass
class PrefillResult:
    past_key_values: tuple
    logits: torch.Tensor
    prefill_time_ms: float
    num_input_tokens: int

@dataclass
class DecodeResult:
    generated_ids: List[int]
    decode_time_ms: float
    tokens_per_second: float
    num_generated_tokens: int

class InferenceEngine:

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.config.name}")
        print(f"  dtype: {self.config.dtype}, TP: {self.config.tp_size}")

        t0 = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = getattr(torch, self.config.dtype)

        load_kwargs = dict(
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.config.quantization == "int4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )
            print(f"  Quantization: INT4 (bitsandbytes)")
        elif self.config.quantization == "int8":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            print(f"  Quantization: INT8 (bitsandbytes)")
        else:
            load_kwargs["dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            **load_kwargs,
        )
        self.model.eval()

        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.1f}s")
        print(f"  Device map: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'single'}")

    def tokenize(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt")

        device = next(self.model.parameters()).device
        return tokens["input_ids"].to(device)

    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
            )

        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

        return PrefillResult(
            past_key_values=outputs.past_key_values,
            logits=outputs.logits,
            prefill_time_ms=prefill_ms,
            num_input_tokens=input_ids.shape[1],
        )

    def decode(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple,
        max_new_tokens: int = 64,
    ) -> DecodeResult:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        generated_ids = []
        current_ids = input_ids[:, -1:]

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token.item())
                current_ids = next_token

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1000
        num_tokens = len(generated_ids)
        tps = (num_tokens / decode_ms * 1000) if decode_ms > 0 else 0

        return DecodeResult(
            generated_ids=generated_ids,
            decode_time_ms=decode_ms,
            tokens_per_second=tps,
            num_generated_tokens=num_tokens,
        )

    def prefill_and_decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> Tuple[PrefillResult, DecodeResult]:
        prefill = self.prefill(input_ids)
        decode = self.decode(input_ids, prefill.past_key_values, max_new_tokens)
        return prefill, decode
