from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import urllib.error
import urllib.request
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
    finish_reason: str | None = None


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


class OpenAICompatibleError(RuntimeError):
    """An API request failed after the configured retry policy."""


class OpenAICompatibleChatClient:
    """Async-compatible client for OpenAI-style chat completion endpoints.

    The client intentionally uses the standard library HTTP stack so the API
    path does not require the OpenAI SDK and remains compatible with DeepSeek
    and other OpenAI-compatible providers.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        max_callback_new_tokens: int = 768,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("An API key is required for the chat client")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.max_callback_new_tokens = int(max_callback_new_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))
        self._usage_lock = threading.Lock()
        self._usage = UsageSnapshot()

    @property
    def model_path(self) -> Path:
        """Compatibility attribute used only for diagnostics by adapters."""

        return Path(self.model_name)

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _messages(
        prompt: str,
        system_prompt: str | None,
        history_messages: Sequence[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for item in history_messages or []:
            role = item.get("role")
            if role in {"system", "user", "assistant"}:
                messages.append({"role": role, "content": str(item.get("content", ""))})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _request_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(),
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise OpenAICompatibleError("Chat API returned a non-object JSON payload")
                return parsed
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                try:
                    detail = json.loads(raw)
                except json.JSONDecodeError:
                    detail = raw[:500]
                last_error = OpenAICompatibleError(
                    f"Chat API HTTP {exc.code}: {detail}"
                )
                if exc.code not in {408, 409, 429} and not 500 <= exc.code < 600:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = OpenAICompatibleError(f"Chat API request failed: {exc}")
            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))
        raise last_error or OpenAICompatibleError("Chat API request failed")

    def _generate_sync(
        self,
        prompt: str,
        system_prompt: str | None,
        history_messages: Sequence[dict[str, Any]] | None,
        max_new_tokens: int | None,
        kwargs: dict[str, Any],
    ) -> Generation:
        requested_tokens = int(max_new_tokens or self.max_new_tokens)
        if (
            kwargs.get("keyword_extraction") or kwargs.get("entity_extraction")
        ) and kwargs.get("response_format") is None:
            kwargs["response_format"] = {"type": "json_object"}
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._messages(prompt, system_prompt, history_messages),
            "temperature": self.temperature,
            "max_tokens": min(requested_tokens, self.max_callback_new_tokens),
        }
        for name in ("response_format", "top_p", "stop", "seed"):
            if kwargs.get(name) is not None:
                payload[name] = kwargs[name]
        response = self._request_sync(payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenAICompatibleError("Chat API response contains no choices")
        message = choices[0].get("message", {})
        text = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(text, list):
            text = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in text
            )
        text = str(text or "").strip()
        usage = response.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        with self._usage_lock:
            self._usage = UsageSnapshot(
                self._usage.input_tokens + input_tokens,
                self._usage.output_tokens + output_tokens,
                self._usage.calls + 1,
            )
        return Generation(text, input_tokens, output_tokens, choices[0].get("finish_reason"))

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> Generation:
        return await asyncio.to_thread(
            self._generate_sync,
            prompt,
            system_prompt,
            history_messages,
            max_new_tokens,
            kwargs,
        )

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        # Upstream graph extractors call complete() without a token override;
        # they need the larger callback budget rather than the short answer budget.
        max_tokens = (
            kwargs.pop("max_tokens", None)
            or kwargs.pop("max_new_tokens", None)
            or self.max_callback_new_tokens
        )
        result = await self.generate(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            max_new_tokens=max_tokens,
            **kwargs,
        )
        return result.text

    def snapshot(self) -> UsageSnapshot:
        with self._usage_lock:
            return self._usage

    def unload(self) -> None:
        return None


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

    @property
    def model_name(self) -> str:
        """Stable display name shared by local and API embedding clients."""

        return self.model_path.name

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


class OpenAICompatibleEmbeddingClient:
    """Batching and normalization adapter for OpenAI-compatible embeddings."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimension: int,
        dimensions: int | None = None,
        batch_size: int = 16,
        max_length: int = 8191,
        normalize: bool = True,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("An API key is required for the embedding client")
        if int(dimension) <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.model_path = Path(model)
        self._dimension = int(dimension)
        self.dimensions = int(dimensions) if dimensions is not None else None
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(0, int(max_retries))

    @property
    def dimension(self) -> int:
        return self._dimension

    def _endpoint(self) -> str:
        return f"{self.base_url}/embeddings"

    def _request_sync(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(),
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise OpenAICompatibleError("Embedding API returned a non-object JSON payload")
                return parsed
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                try:
                    detail = json.loads(raw)
                except json.JSONDecodeError:
                    detail = raw[:500]
                last_error = OpenAICompatibleError(
                    f"Embedding API HTTP {exc.code}: {detail}"
                )
                if exc.code not in {408, 409, 429} and not 500 <= exc.code < 600:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = OpenAICompatibleError(f"Embedding API request failed: {exc}")
            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))
        raise last_error or OpenAICompatibleError("Embedding API request failed")

    def _embed_sync(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            payload: dict[str, Any] = {"model": self.model_name, "input": batch}
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions
            response = self._request_sync(payload)
            data = response.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise OpenAICompatibleError(
                    f"Embedding API returned {len(data) if isinstance(data, list) else 0} vectors "
                    f"for {len(batch)} inputs"
                )
            ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
            vectors = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
                raise OpenAICompatibleError(
                    f"Expected embedding shape (*, {self.dimension}), got {vectors.shape}"
                )
            if self.normalize:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise OpenAICompatibleError("Embedding API returned a zero vector")
                vectors = vectors / norms
            batches.append(vectors)
        return np.concatenate(batches, axis=0).astype(np.float32, copy=False)

    async def embed(self, texts: Sequence[str], **_: Any) -> np.ndarray:
        return await asyncio.to_thread(self._embed_sync, list(texts))

    def unload(self) -> None:
        return None


ANSWER_SYSTEM_PROMPT = (
    "Answer the medical question using only the supplied context. Give a direct, "
    "concise answer. If the context is insufficient, say I don't know."
)


async def generate_grounded_answer(
    client: Any,
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


def _api_key(config: dict[str, Any], kind: str) -> str:
    env_name = str(config.get("api_key_env", ""))
    if not env_name:
        raise ValueError(f"{kind} API config must set api_key_env")
    value = os.getenv(env_name)
    if not value:
        raise RuntimeError(f"Environment variable {env_name} is not set")
    return value


def build_chat_client(config: dict[str, Any]) -> Any:
    backend = str(config.get("backend", "transformers")).lower()
    if backend in {"api", "openai", "openai_compatible"}:
        return OpenAICompatibleChatClient(
            base_url=str(config["base_url"]),
            api_key=_api_key(config, "LLM"),
            model=str(config["model"]),
            temperature=float(config.get("temperature", 0.0)),
            max_new_tokens=int(config.get("max_new_tokens", 256)),
            max_callback_new_tokens=int(config.get("max_callback_new_tokens", 768)),
            timeout_seconds=float(config.get("timeout_seconds", 120)),
            max_retries=int(config.get("max_retries", 3)),
        )
    if backend != "transformers":
        raise ValueError(f"Unsupported LLM backend: {backend}")
    return TransformersChatClient(
        model_path=config["model_path"],
        device=config["device"],
        dtype=config.get("dtype", "bfloat16"),
        temperature=float(config.get("temperature", 0.0)),
        max_new_tokens=int(config.get("max_new_tokens", 256)),
        max_callback_new_tokens=int(config.get("max_callback_new_tokens", 768)),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
    )


def build_embedding_client(config: dict[str, Any]) -> Any:
    backend = str(config.get("backend", "transformers")).lower()
    if backend in {"api", "openai", "openai_compatible"}:
        return OpenAICompatibleEmbeddingClient(
            base_url=str(config["base_url"]),
            api_key=_api_key(config, "Embedding"),
            model=str(config["model"]),
            dimension=int(config["dimension"]),
            dimensions=(
                int(config["dimensions"])
                if config.get("dimensions") is not None
                else None
            ),
            batch_size=int(config.get("batch_size", 16)),
            max_length=int(config.get("max_length", 8191)),
            normalize=bool(config.get("normalize", True)),
            timeout_seconds=float(config.get("timeout_seconds", 120)),
            max_retries=int(config.get("max_retries", 3)),
        )
    if backend != "transformers":
        raise ValueError(f"Unsupported embedding backend: {backend}")
    return TransformersEmbeddingClient(
        model_path=config["model_path"],
        device=config["device"],
        batch_size=int(config.get("batch_size", 16)),
        max_length=int(config.get("max_length", 512)),
        normalize=bool(config.get("normalize", True)),
    )
