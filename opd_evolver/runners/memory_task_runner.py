from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from opd_evolver.runners.task_runner import (
    MemoryAugmentedTaskRunner,
    SimpleTaskRunner,
)


def _memory_config_to_dict(memory_config: Any) -> Dict[str, Any]:
    if memory_config is None:
        return {}
    if isinstance(memory_config, dict):
        return dict(memory_config)
    config: Dict[str, Any] = {}
    for key in (
        "storage_dir",
        "embedding_provider",
        "embedding_model",
        "cold_start_threshold",
        "retrieval_top_k",
        "memory_tiers",
        "writer_dataset_path",
        "selector_dataset_path",
    ):
        if hasattr(memory_config, key):
            value = getattr(memory_config, key)
            if value is not None:
                config[key] = value
    return config


class LegacyMemoryAugmentedTaskRunner(MemoryAugmentedTaskRunner):
    def __init__(
        self,
        main_model: str,
        sub_models: Optional[List[str]] = None,
        env_type: str = "bash",
        max_attempts: int = 10,
        trajectory_dir: Path | None = None,
        csv_summary_path: Path | None = None,
        memory_enabled: bool = True,
        memory_config: Any = None,
        memory_manager: Any = None,
        reflection_model: Optional[str] = None,
        selector_model: Optional[str] = None,
    ):
        config = _memory_config_to_dict(memory_config)
        if reflection_model:
            config["reflection_model"] = reflection_model
        if selector_model:
            config["selector_model"] = selector_model
        super().__init__(
            model=main_model,
            env_type=env_type,
            memory_config=config,
            max_steps=max_attempts,
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_summary_path,
        )
        self.main_model = main_model
        self.sub_models = sub_models or []
        self.memory_enabled = memory_enabled
        self._legacy_memory_manager = memory_manager

    async def _get_memory_pipeline(self):
        if not self.memory_enabled:
            return None
        return await super()._get_memory_pipeline()

    def _get_provider_adapter(self) -> Any | None:
        if not self.memory_enabled:
            return None
        return super()._get_provider_adapter()

    def get_memory_manager(self) -> Any:
        return self._legacy_memory_manager


MemoryAugmentedInterCodeRunner = LegacyMemoryAugmentedTaskRunner
MemoryAugmentedIntercodeRunner = LegacyMemoryAugmentedTaskRunner

__all__ = [
    "SimpleTaskRunner",
    "MemoryAugmentedTaskRunner",
    "LegacyMemoryAugmentedTaskRunner",
    "MemoryAugmentedInterCodeRunner",
    "MemoryAugmentedIntercodeRunner",
]
