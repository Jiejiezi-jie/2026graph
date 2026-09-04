from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class UsageSnapshot:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def __sub__(self, other: "UsageSnapshot") -> "UsageSnapshot":
        return UsageSnapshot(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            calls=self.calls - other.calls,
        )


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int


class TransformersChatClient:
    """Lazy, serialized text-only inference for local causal/VL chat models.

    The Qwen VL checkpoints are deliberately fed no image inputs. Loading is
    lazy so data preparation and unit tests never allocate GPU memory.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str,
        dtype: str = "bfloat16",
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        max_callback_new_tokens: int = 768,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.dtype_name = dtype
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.max_callback_new_tokens = int(max_callback_new_tokens)
        self.trust_remote_code = bool(trust_remote_code)
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._inference_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self._usage = UsageSnapshot()

    @property
    def model_name(self) -> str:
        return self.model_path.name

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Local model is missing: {self.model_path}")

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        dtype = getattr(torch, self.dtype_name)
        config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code
        )
        model_type = getattr(config, "model_type", "")
        model_kwargs = {
            "dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if model_type in {"qwen2_5_vl", "qwen3_vl"}:
            try:
                from transformers import AutoModelForImageTextToText

                model_cls = AutoModelForImageTextToText
            except ImportError:
                if model_type == "qwen2_5_vl":
                    from transformers import Qwen2_5_VLForConditionalGeneration

                    model_cls = Qwen2_5_VLForConditionalGeneration
                else:
                    from transformers import Qwen3VLForConditionalGeneration

                    model_cls = Qwen3VLForConditionalGeneration
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
            self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
            self._model = model_cls.from_pretrained(self.model_path, **model_kwargs)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **model_kwargs
            )
        self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def _messages(
        prompt: str,
        system_prompt: str | None,
        history_messages: Sequence[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for item in history_messages or []:
            if item.get("role") in {"system", "user", "assistant"}:
                messages.append(
                    {"role": item["role"], "content": str(item.get("content", ""))}
                )
        messages.append({"role": "user", "content": prompt})
        return messages

    def _render_inputs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if self._processor is not None:
            vl_messages = []
            for message in messages:
                content = message["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                vl_messages.append({"role": message["role"], "content": content})
            rendered = self._processor.apply_chat_template(
                vl_messages, tokenize=False, add_generation_prompt=True
            )
            encoded = self._processor(
                text=[rendered], padding=True, return_tensors="pt"
            )
        else:
            rendered = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            encoded = self._tokenizer(rendered, return_tensors="pt")
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }

    def _generate_sync(
        self,
        prompt: str,
        system_prompt: str | None,
        history_messages: Sequence[dict[str, Any]] | None,
        max_new_tokens: int | None,
    ) -> Generation:
        import torch

        with self._inference_lock:
            self._load()
            inputs = self._render_inputs(
                self._messages(prompt, system_prompt, history_messages)
            )
            input_tokens = int(inputs["input_ids"].shape[-1])
            requested_tokens = int(max_new_tokens or self.max_new_tokens)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": min(requested_tokens, self.max_callback_new_tokens),
                "do_sample": self.temperature > 0,
                "pad_token_id": self._tokenizer.eos_token_id,
            }
            if self.temperature > 0:
                generation_kwargs["temperature"] = self.temperature
            with torch.inference_mode():
                generated = self._model.generate(**inputs, **generation_kwargs)
            continuation = generated[:, input_tokens:]
            output_tokens = int(continuation.shape[-1])
            text = self._tokenizer.batch_decode(
                continuation, skip_special_tokens=True
            )[0].strip()
        with self._usage_lock:
            self._usage = UsageSnapshot(
                self._usage.input_tokens + input_tokens,
                self._usage.output_tokens + output_tokens,
                self._usage.calls + 1,
            )
        return Generation(text=text, input_tokens=input_tokens, output_tokens=output_tokens)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
    ) -> Generation:
        return await asyncio.to_thread(
            self._generate_sync,
            prompt,
            system_prompt,
            history_messages,
            max_new_tokens,
        )

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Callback compatible with official LightRAG and PathRAG."""

        max_tokens = (
            kwargs.get("max_tokens")
            or kwargs.get("max_new_tokens")
            or self.max_callback_new_tokens
        )
        result = await self.generate(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            max_new_tokens=max_tokens,
        )
        return result.text

    def snapshot(self) -> UsageSnapshot:
        with self._usage_lock:
            return self._usage

    def unload(self) -> None:
        if self._model is None:
            return
        import torch

        self._model = None
        self._processor = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TransformersEmbeddingClient:
    """Shared BGE/E5-style sentence embedding client."""

    def __init__(
        self,
        model_path: str | Path,
        device: str,
        batch_size: int = 16,
        max_length: int = 512,
        normalize: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self._tokenizer: Any = None
        self._model: Any = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Dedicated embedding model is missing: {self.model_path}. "
                "Do not substitute LLM/VLM hidden states."
            )
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self._model.eval()
        self._dimension = int(self._model.config.hidden_size)

    @property
    def dimension(self) -> int:
        with self._lock:
            self._load()
            assert self._dimension is not None
            return self._dimension

    def _embed_sync(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches: list[np.ndarray] = []
        with self._lock:
            self._load()
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    hidden = self._model(**encoded).last_hidden_state
                vectors = hidden[:, 0]
                if self.normalize:
                    vectors = functional.normalize(vectors, p=2, dim=1)
                batches.append(vectors.float().cpu().numpy())
        return np.concatenate(batches, axis=0).astype(np.float32, copy=False)

    async def embed(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        return await asyncio.to_thread(self._embed_sync, list(texts))

    def unload(self) -> None:
        if self._model is None:
            return
        import torch

        self._model = None
        self._tokenizer = None
        self._dimension = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


ANSWER_SYSTEM_PROMPT = (
    "Answer the medical question using only the supplied context. Give a direct, "
    "concise answer. If the context is insufficient, say I don't know."
)


async def generate_grounded_answer(
    client: TransformersChatClient,
    question: str,
    contexts: Sequence[str],
    system_prompt: str = ANSWER_SYSTEM_PROMPT,
    max_context_tokens: int = 5000,
) -> Generation:
    import tiktoken

    numbered = "\n\n".join(
        f"[Context {index}]\n{text}" for index, text in enumerate(contexts, start=1)
    )
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    context_tokens = encoding.encode(numbered)
    if len(context_tokens) > max_context_tokens:
        numbered = encoding.decode(context_tokens[:max_context_tokens])
    prompt = f"Medical question: {question}\n\nRetrieved context:\n{numbered}"
    return await client.generate(prompt, system_prompt=system_prompt)
