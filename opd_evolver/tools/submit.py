from __future__ import annotations
from typing import Any, Dict
from pydantic import Field
from opd_evolver.base.agent.base_action import BaseAction
from opd_evolver.base.engine.logs import logger
class SubmitTool(BaseAction):
    name: str = "submit"
    description: str = "Run tests in current container to verify task completion"
    parameters: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Why ready for submission"},
        },
        "required": ["reason"]
    })
    env: Any = Field(default=None, exclude=True)
    class Config:
        arbitrary_types_allowed = True
    def __init__(self, env):
        super().__init__()
        self.env = env
    async def __call__(self, reason: str = "") -> Dict:
        logger.info(f"[SubmitTool] Called with reason: {reason}")
        logger.info(f"[SubmitTool] Container status: _container_started={getattr(self.env, '_container_started', 'N/A')}")
        if not hasattr(self.env, '_container_started') or not self.env._container_started:
            logger.error("[SubmitTool] No container running! Cannot run tests.")
            return {"success": False, "reward": 0.0, "done": True, "error": "No container running"}
        logger.info("[SubmitTool] Running verification tests...")
        submit_action = {"action": "submit", "params": {}}
        obs, reward, done, info = await self.env.step(submit_action)
        logger.info(f"[SubmitTool] Tests completed: reward={reward}, done={done}")
        return {
            "success": reward == 1,
            "reward": float(reward),
            "done": done,
            "observation": obs,
            "info": info,
        }
