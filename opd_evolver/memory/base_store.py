from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import uuid
def _dbg_log(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: Dict[str, Any],
    run_id: str = "pre-fix",
) -> None:
    try:
        import json as _json
        import time as _time
        payload = {
            "sessionId": "42fd61",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(_time.time() * 1000),
        }
        with open(
            ".cursor/debug-42fd61.log",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
@dataclass
class MemoryItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_task: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    success_count: int = 0
    usage_count: int = 0
    last_used: Optional[str] = None
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "source_task": self.source_task,
            "created_at": self.created_at,
            "success_count": self.success_count,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            source_task=data.get("source_task"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            success_count=data.get("success_count", 0),
            usage_count=data.get("usage_count", 0),
            last_used=data.get("last_used"),
        )
@dataclass
class RetrievedItem:
    item: MemoryItem
    similarity: float
    tier: str
    tag: str
    def format_for_context(self) -> str:
        return f"{self.tag}\nTier: {self.tier}\nContent: {self.item.content}\nSimilarity: {self.similarity:.3f}"
class GpuEmbeddingIndex:
    _CUDA_DTYPE = "float16"
    _CPU_DTYPE  = "float32"
    def __init__(self, device: str = "auto") -> None:
        self._requested_device = device
        self._matrix: Any = None
        self._use_torch: bool = True
        self._n: int = 0
        self._dirty: bool = True
    def _resolve_device(self) -> str:
        if self._requested_device != "auto":
            return self._requested_device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    def _torch_dtype(self, device: str):
        import torch
        return torch.float16 if device.startswith("cuda") else torch.float32
    def mark_dirty(self) -> None:
        self._dirty = True
    def build(self, embeddings: List[List[float]]) -> None:
        self._dirty = False
        self._n = 0
        self._matrix = None
        if not embeddings:
            return
        device = self._resolve_device()
        try:
            import torch
            dtype = self._torch_dtype(device)
            try:
                mat = torch.tensor(embeddings, dtype=dtype, device=device)
            except RuntimeError:
                device = "cpu"
                dtype = torch.float32
                mat = torch.tensor(embeddings, dtype=dtype, device=device)
            norms = mat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
            self._matrix = mat / norms
            self._n = self._matrix.shape[0]
            self._use_torch = True
            _dbg_log(
                hypothesis_id="H3",
                location="opd_evolver/memory/base_store.py:GpuEmbeddingIndex.build",
                message="build_done_torch",
                data={
                    "requested_device": self._requested_device,
                    "resolved_device": str(getattr(self._matrix, "device", device)),
                    "dtype": str(getattr(self._matrix, "dtype", dtype)),
                    "n": int(self._n),
                    "use_torch": bool(self._use_torch),
                },
            )
            return
        except ImportError:
            pass
        import numpy as np
        mat = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._matrix = mat / np.maximum(norms, 1e-9)
        self._n = self._matrix.shape[0]
        self._use_torch = False
        _dbg_log(
            hypothesis_id="H3",
            location="opd_evolver/memory/base_store.py:GpuEmbeddingIndex.build",
            message="build_done_numpy",
            data={
                "requested_device": self._requested_device,
                "resolved_device": "cpu",
                "dtype": "float32",
                "n": int(self._n),
                "use_torch": bool(self._use_torch),
            },
        )
    def topk(
        self,
        query: List[float],
        k: int,
        embeddings: List[List[float]],
    ) -> Tuple[List[int], List[float]]:
        if self._dirty or self._matrix is None:
            self.build(embeddings)
        if self._n == 0:
            return [], []
        k = min(k, self._n)
        if self._use_torch:
            try:
                import torch
                device = self._matrix.device
                dtype  = self._matrix.dtype
                q = torch.tensor(query, dtype=dtype, device=device)
                q = q / q.norm(p=2).clamp(min=1e-9)
                sims = self._matrix @ q
                topk_vals, topk_idx = torch.topk(sims, k)
                _dbg_log(
                    hypothesis_id="H1",
                    location="opd_evolver/memory/base_store.py:GpuEmbeddingIndex.topk",
                    message="topk_done_torch",
                    data={
                        "k": int(k),
                        "n": int(self._n),
                        "device": str(device),
                        "dtype": str(dtype),
                        "top0_idx": int(topk_idx[0].item()) if int(k) > 0 else None,
                        "top0_sim": float(topk_vals[0].item()) if int(k) > 0 else None,
                    },
                )
                return topk_idx.cpu().tolist(), topk_vals.cpu().to(torch.float32).tolist()
            except Exception:
                self._use_torch = False
                self.build(embeddings)
        import numpy as np
        q = np.array(query, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 1e-9:
            q = q / norm
        sims = self._matrix @ q
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        _dbg_log(
            hypothesis_id="H1",
            location="opd_evolver/memory/base_store.py:GpuEmbeddingIndex.topk",
            message="topk_done_numpy",
            data={
                "k": int(k),
                "n": int(self._n),
                "device": "cpu",
                "dtype": "float32",
                "top0_idx": int(idx[0]) if int(k) > 0 else None,
                "top0_sim": float(sims[idx[0]]) if int(k) > 0 else None,
            },
        )
        return idx.tolist(), sims[idx].tolist()
class BaseMemoryStore(ABC):
    def __init__(
        self,
        tier_name: str,
        storage_path: Optional[str] = None,
        embedding_provider: Optional[Any] = None,
        index_device: str = "auto",
    ):
        self.tier_name = tier_name
        self.storage_path = storage_path
        self.embedding_provider = embedding_provider
        self._items: Dict[str, MemoryItem] = {}
        self._embeddings: List[List[float]] = []
        self._item_ids: List[str] = []
        self._gpu_index = GpuEmbeddingIndex(device=index_device)
        if storage_path:
            self._load_from_storage()
    @abstractmethod
    def _create_item(self, content: str, metadata: Dict[str, Any]) -> MemoryItem:
        pass
    async def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_task: Optional[str] = None,
    ) -> MemoryItem:
        metadata = metadata or {}
        item = self._create_item(content, metadata)
        item.source_task = source_task
        if self.embedding_provider:
            item.embedding = await self.embedding_provider.embed(content)
        self._items[item.id] = item
        if item.embedding:
            self._embeddings.append(item.embedding)
            self._item_ids.append(item.id)
            self._gpu_index.mark_dirty()
            _dbg_log(
                hypothesis_id="H1",
                location="opd_evolver/memory/base_store.py:BaseMemoryStore.add",
                message="add_mark_dirty",
                data={
                    "tier": self.tier_name,
                    "item_id": str(item.id),
                    "n_after": int(len(self._embeddings)),
                },
            )
        if self.storage_path:
            self._save_to_storage()
        return item
    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[RetrievedItem]:
        if not self._embeddings:
            return []
        indices, sims = self._gpu_index.topk(
            query=query_embedding,
            k=top_k + max(0, int(len(self._embeddings) * 0.1)),
            embeddings=self._embeddings,
        )
        results: List[RetrievedItem] = []
        for rank, (idx, sim) in enumerate(zip(indices, sims)):
            if sim < min_similarity:
                break
            if len(results) >= top_k:
                break
            item_id = self._item_ids[idx]
            item = self._items.get(item_id)
            if item:
                tag = f"[RETRIEVED_{self.tier_name.upper()}_{rank + 1:02d}]"
                results.append(RetrievedItem(
                    item=item,
                    similarity=float(sim),
                    tier=self.tier_name,
                    tag=tag,
                ))
        _dbg_log(
            hypothesis_id="H4",
            location="opd_evolver/memory/base_store.py:BaseMemoryStore.search_by_embedding",
            message="search_by_embedding_done",
            data={
                "tier": self.tier_name,
                "n": int(len(self._embeddings)),
                "top_k": int(top_k),
                "min_similarity": float(min_similarity),
                "returned": int(len(results)),
                "top0_item_id": str(results[0].item.id) if results else None,
                "top0_sim": float(results[0].similarity) if results else None,
            },
        )
        return results
    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[RetrievedItem]:
        if not self._embeddings or not self.embedding_provider:
            return []
        query_embedding = await self.embedding_provider.embed(query)
        return self.search_by_embedding(query_embedding, top_k, min_similarity)
    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)
    def get_all(self) -> List[MemoryItem]:
        return list(self._items.values())
    def count(self) -> int:
        return len(self._items)
    def increment_usage(self, item_id: str) -> None:
        if item_id in self._items:
            self._items[item_id].usage_count += 1
            if self.storage_path:
                self._save_to_storage()
    def increment_success(self, item_id: str) -> None:
        if item_id in self._items:
            self._items[item_id].success_count += 1
            if self.storage_path:
                self._save_to_storage()
    def _load_from_storage(self) -> None:
        import os
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item_data in data.get("items", []):
                item = MemoryItem.from_dict(item_data)
                self._items[item.id] = item
                if item.embedding:
                    self._embeddings.append(item.embedding)
                    self._item_ids.append(item.id)
            self._gpu_index.mark_dirty()
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load memory store from {self.storage_path}: {e}")
    def _save_to_storage(self) -> None:
        import os
        if not self.storage_path:
            return
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "tier": self.tier_name,
            "count": len(self._items),
            "items": [item.to_dict() for item in self._items.values()],
        }
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Warning: Failed to save memory store to {self.storage_path}: {e}")
    def clear(self) -> None:
        self._items.clear()
        self._embeddings.clear()
        self._item_ids.clear()
        self._gpu_index.mark_dirty()
        if self.storage_path:
            self._save_to_storage()
