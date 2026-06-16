import asyncio
import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple
from opd_evolver.base.engine.logs import logger
class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
class OpenRouterEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str = "qwen/qwen3-embedding-8b",
        api_key: Optional[str] = None,
        cache_path: Optional[str] = None,
        max_batch_size: int = 100,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.cache_path = cache_path
        self.max_batch_size = max_batch_size
        self._cache: Dict[str, List[float]] = {}
        if cache_path:
            self._load_cache()
        self._client = None
    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1",
                )
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client
    @property
    def dimension(self) -> int:
        dimensions = {
            "qwen/qwen3-embedding-8b": 4096,
            "qwen/qwen3-embedding-4b": 2560,
            "qwen/qwen3-embedding-0.6b": 1024,
        }
        return dimensions.get(self.model, 4096)
    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()[:16]
    async def embed(self, text: str) -> List[float]:
        text_hash = self._text_hash(text)
        if text_hash in self._cache:
            return self._cache[text_hash]
        try:
            client = self._get_client()
            response = await client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float",
                extra_headers={
                    "HTTP-Referer": "https://github.com/opd_evolver",
                    "X-OpenRouter-Title": "OPD Evolver Memory",
                },
            )
            embedding = response.data[0].embedding
            self._cache[text_hash] = embedding
            if self.cache_path:
                self._save_cache()
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding via OpenRouter: {e}")
            raise
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            if text_hash in self._cache:
                results[i] = self._cache[text_hash]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        if uncached_texts:
            client = self._get_client()
            for batch_start in range(0, len(uncached_texts), self.max_batch_size):
                batch_end = min(batch_start + self.max_batch_size, len(uncached_texts))
                batch = uncached_texts[batch_start:batch_end]
                try:
                    response = await client.embeddings.create(
                        model=self.model,
                        input=batch,
                        encoding_format="float",
                        extra_headers={
                            "HTTP-Referer": "https://github.com/opd_evolver",
                            "X-OpenRouter-Title": "OPD Evolver Memory",
                        },
                    )
                    for j, emb_data in enumerate(response.data):
                        idx = uncached_indices[batch_start + j]
                        text = texts[idx]
                        embedding = emb_data.embedding
                        results[idx] = embedding
                        text_hash = self._text_hash(text)
                        self._cache[text_hash] = embedding
                except Exception as e:
                    logger.error(f"Failed to generate batch embeddings via OpenRouter: {e}")
                    raise
            if self.cache_path:
                self._save_cache()
        return [r for r in results if r is not None]
    def _load_cache(self) -> None:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.debug(f"Loaded {len(self._cache)} cached embeddings")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self._cache = {}
    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except IOError as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    def clear_cache(self) -> None:
        self._cache.clear()
        if self.cache_path and os.path.exists(self.cache_path):
            os.remove(self.cache_path)
class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_path: Optional[str] = None,
        max_batch_size: int = 100,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.cache_path = cache_path
        self.max_batch_size = max_batch_size
        self._cache: Dict[str, List[float]] = {}
        if cache_path:
            self._load_cache()
        self._client = None
    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        return self._client
    @property
    def dimension(self) -> int:
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)
    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()[:16]
    async def embed(self, text: str) -> List[float]:
        text_hash = self._text_hash(text)
        if text_hash in self._cache:
            return self._cache[text_hash]
        try:
            client = self._get_client()
            response = await client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = response.data[0].embedding
            self._cache[text_hash] = embedding
            if self.cache_path:
                self._save_cache()
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            if text_hash in self._cache:
                results[i] = self._cache[text_hash]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        if uncached_texts:
            client = self._get_client()
            for batch_start in range(0, len(uncached_texts), self.max_batch_size):
                batch_end = min(batch_start + self.max_batch_size, len(uncached_texts))
                batch = uncached_texts[batch_start:batch_end]
                try:
                    response = await client.embeddings.create(
                        model=self.model,
                        input=batch,
                    )
                    for j, emb_data in enumerate(response.data):
                        idx = uncached_indices[batch_start + j]
                        text = texts[idx]
                        embedding = emb_data.embedding
                        results[idx] = embedding
                        text_hash = self._text_hash(text)
                        self._cache[text_hash] = embedding
                except Exception as e:
                    logger.error(f"Failed to generate batch embeddings: {e}")
                    raise
            if self.cache_path:
                self._save_cache()
        return [r for r in results if r is not None]
    def _load_cache(self) -> None:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            logger.debug(f"Loaded {len(self._cache)} cached embeddings")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self._cache = {}
    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except IOError as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    def clear_cache(self) -> None:
        self._cache.clear()
        if self.cache_path and os.path.exists(self.cache_path):
            os.remove(self.cache_path)
class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension: Optional[int] = None
    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                self._dimension = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for local embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._get_model()
        return self._dimension or 384
    async def embed(self, text: str) -> List[float]:
        model = self._get_model()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, convert_to_numpy=True).tolist()
        )
        return embedding
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, convert_to_numpy=True).tolist()
        )
        return embeddings
def local_hf_embedding_settings_from_env() -> Tuple[str, int, str]:
    device = os.getenv("OPD_EVOLVER_EMBEDDING_DEVICE", "cuda")
    max_len = int(os.getenv("OPD_EVOLVER_EMBEDDING_MAX_LENGTH", "2048"))
    dtype = os.getenv("OPD_EVOLVER_EMBEDDING_DTYPE", "float16")
    return device, max_len, dtype
class LocalHFEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cpu",
        torch_dtype: str = "auto",
        max_length: int = 2048,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._dimension: Optional[int] = None
        self._lock = asyncio.Lock()
        self._embed_batch_delay_s = 0.005
        self._pending_embed_requests: List[Tuple[str, "asyncio.Future[List[float]]"]] = []
        self._pending_embed_lock = asyncio.Lock()
        self._pending_embed_flush_handle: Optional[asyncio.TimerHandle] = None
    def _get_model_and_tokenizer(self) -> Tuple[object, object]:
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "transformers + torch are required for LocalHFEmbeddingProvider. "
                    "Install with: pip install transformers torch"
                ) from e
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
            dtype = self.torch_dtype
            if dtype == "auto":
                torch_dtype = None
            else:
                torch_dtype = {
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                }.get(str(dtype).lower())
            model = AutoModel.from_pretrained(self.model_id, torch_dtype=torch_dtype)
            model = model.to(self.device)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._dimension = int(getattr(model.config, "hidden_size", 0) or 0) or None
        return self._model, self._tokenizer
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._get_model_and_tokenizer()
        return self._dimension or 1024
    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        import torch
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    def _embed_batch_sync(self, texts: Sequence[str]) -> List[List[float]]:
        import torch
        import torch.nn.functional as F
        model, tokenizer = self._get_model_and_tokenizer()
        batch = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            param0 = next(model.parameters())
            model_dtype = param0.dtype
            if device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16):
                amp_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float16
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            pooled = self._last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            return pooled.detach().cpu().to(torch.float32).numpy().tolist()
    async def embed(self, text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[List[float]] = loop.create_future()
        async with self._pending_embed_lock:
            self._pending_embed_requests.append((text, fut))
            if self._pending_embed_flush_handle is None:
                self._pending_embed_flush_handle = loop.call_later(
                    self._embed_batch_delay_s,
                    lambda: asyncio.create_task(self._flush_pending_embeds()),
                )
        return await fut
    async def _flush_pending_embeds(self) -> None:
        async with self._pending_embed_lock:
            batch = self._pending_embed_requests
            self._pending_embed_requests = []
            self._pending_embed_flush_handle = None
        if not batch:
            return
        texts = [text for text, _future in batch]
        loop = asyncio.get_running_loop()
        try:
            async with self._lock:
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: self._embed_batch_sync(texts),
                )
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedding batch returned {len(embeddings)} results for {len(batch)} inputs"
                )
        except Exception as exc:
            for _text, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return
        for embedding, (_text, future) in zip(embeddings, batch):
            if not future.done():
                future.set_result(embedding)
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(None, lambda: self._embed_batch_sync(texts))
