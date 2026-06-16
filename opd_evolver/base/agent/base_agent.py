from abc import abstractmethod
from typing import List, Optional, Any, Dict
from pydantic import Field, BaseModel
from opd_evolver.base.agent.base_action import BaseAction
from opd_evolver.base.engine.async_llm import AsyncLLM
class BaseAgent(BaseAction, BaseModel):
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )
    llm: Optional[AsyncLLM] = Field(default=None, description="Language model instance")
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    class Config:
        arbitrary_types_allowed = True
    @abstractmethod
    async def step(self):
        pass
    @abstractmethod
    async def run(self, request: Optional[str] = None) -> str:
        pass
    async def __call__(self, **kwargs) -> Any:
        return await self.run(**kwargs)
    def to_param(self) -> Dict[str, Any]:
        return {
            "type": "agent-as-function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
