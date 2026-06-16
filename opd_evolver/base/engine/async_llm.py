import os
import re
import threading
import yaml
from openai import AsyncOpenAI
from pathlib import Path
from typing import Dict, Optional, Any
from opd_evolver.base.engine.cost_monitor import record_cost
from opd_evolver.base.engine.logs import logger, LogLevel
def _assistant_message_text(message: Any) -> str:
    raw = getattr(message, "content", None)
    if isinstance(raw, str) and raw.strip():
        return raw
    if isinstance(raw, list):
        chunks: list[str] = []
        for block in raw:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    chunks.append(block["text"])
            elif isinstance(block, str):
                chunks.append(block)
        merged = "".join(chunks).strip()
        if merged:
            return merged
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    if hasattr(message, "model_dump"):
        extra = message.model_dump()
        for key in ("reasoning_content", "reasoning"):
            val = extra.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return raw if isinstance(raw, str) else ""
def _chat_template_extra_body(config: "LLMConfig") -> dict[str, Any] | None:
    enable_thinking = getattr(config, "enable_thinking", None)
    if enable_thinking is None:
        return None
    return {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
class LLMConfig:
    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 1)
        self.key = config.get("key", None)
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.top_p = config.get("top_p", 1)
        self.enable_thinking = config.get("enable_thinking")
class LLMsConfig:
    _instance = None
    _default_config = None
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self.configs = config_dict or {}
    @classmethod
    def default(cls):
        if cls._default_config is None:
            config_data: Optional[Dict[str, Any]] = None
            config_paths = [
                Path("config/global_config.yaml"),
                Path("config/global_config2.yaml"),
                Path("./config/global_config.yaml"),
                Path("config/model_config.yaml"),
            ]
            config_file = next((path for path in config_paths if path.exists() and path.stat().st_size > 0), None)
            if config_file is not None:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = cls._load_config_from_env()
            if not config_data:
                raise FileNotFoundError(
                    "No default configuration file found in the expected locations and no environment-based fallback is configured."
                )
            if "models" in config_data:
                config_data = config_data["models"] or {}
            cls._default_config = cls(config_data)
        return cls._default_config
    @classmethod
    def _load_config_from_env(cls) -> Optional[Dict[str, Any]]:
        inline_config = os.getenv("AUTOENV_MODEL_CONFIG_JSON")
        if inline_config:
            try:
                data = yaml.safe_load(inline_config)
            except yaml.YAMLError:
                logger.log_to_file(
                    LogLevel.WARNING,
                    "Failed to parse AUTOENV_MODEL_CONFIG_JSON; falling back to explicit env vars.",
                )
            else:
                if isinstance(data, dict):
                    return data.get("models", data) or {}
        api_key = os.getenv("AUTOENV_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        base_url = (
            os.getenv("AUTOENV_OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        def _get_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                logger.log_to_file(
                    LogLevel.WARNING,
                    f"Invalid float value for {name}: {raw!r}; using {default}.",
                )
                return default
        temperature = _get_float("AUTOENV_OPENAI_TEMPERATURE", 1)
        top_p = _get_float("AUTOENV_OPENAI_TOP_P", 1)
        models_env = os.getenv("AUTOENV_OPENAI_MODELS", "o3")
        models = [m.strip() for m in models_env.split(",") if m.strip()]
        if not models:
            models = ["o3"]
        config: Dict[str, Any] = {}
        for model_name in models:
            normalized = model_name.upper().replace('-','_').replace('/','_')
            env_key_name = f"AUTOENV_{normalized}_API_KEY"
            env_base_name = f"AUTOENV_{normalized}_BASE_URL"
            model_api_key = os.getenv(env_key_name, api_key)
            model_base_url = os.getenv(env_base_name, base_url)
            config[model_name] = {
                "api_key": model_api_key,
                "base_url": model_base_url,
                "temperature": temperature,
                "top_p": top_p,
            }
        return config
    def get(self, llm_name: str) -> LLMConfig:
        if llm_name not in self.configs:
            raise ValueError(f"Configuration for {llm_name} not found")
        config = self.configs[llm_name]
        llm_config = {
            "model": config.get("model") or llm_name,
            "temperature": config.get("temperature", 1),
            "key": config.get("api_key"),
            "base_url": config.get("base_url", "https://oneapi.deepwisdom.ai/v1"),
            "top_p": config.get("top_p", 1),
            "enable_thinking": config.get("enable_thinking"),
        }
        return LLMConfig(llm_config)
    def add_config(self, name: str, config: Dict[str, Any]) -> None:
        self.configs[name] = config
    def get_all_names(self) -> list:
        return list(self.configs.keys())
class ModelPricing:
    PRICES = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "o3": {"input": 0.002, "output": 0.008},
        "o3-mini": {"input": 0.0011, "output": 0.0044},
        "gpt-5": {"input": 0.00125, "output": 0.01},
        "gpt-5-mini": {"input":0.00025, "output": 0.002},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "moonshotai/kimi-k2": {"input": 0.000296, "output": 0.001185},
        "deepseek/deepseek-chat-v3.1": {"input":0.00025 , "output":0.001},
        "deepseek-chat": {"input":0.00025 , "output":0.001},
        "deepseek-v3": {"input": 0.00025, "output": 0.001},
        "deepseek-v3.1": {"input": 0.00025, "output": 0.001},
        "deepseek-v3.2": {"input": 0.00025, "output": 0.001},
        "deepseek-r1": {"input": 0.00055, "output": 0.00219},
        "z-ai/glm-4.5": {"input": 0.00033, "output": 0.00132},
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
        "claude-4-sonnet": {"input": 0.003, "output": 0.015},
        "claude-4-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
        "claude-4-5-haiku": {"input": 0.00088, "output": 0.0044},
        "claude-4-sonnet-20250514": {"input": 0.003, "output": 0.015},
        "gemini-2.5-flash": {"input": 0.0003, "output": 0.00252},
        "gemini-3-flash-preview": {"input": 0.0005, "output": 0.003},
        "gemini-3-pro-preview": {"input": 0.002, "output": 0.004},
        "gemini-2.5-flash-image": {"input": 0.0003, "output": 0.03},
        "x-ai/grok-4-fast": {"input": 0.0002, "output": 0.0005}
    }
    @classmethod
    def get_price(cls, model_name, token_type):
        if model_name in cls.PRICES:
            return cls.PRICES[model_name][token_type]
        for key in cls.PRICES:
            if key in model_name:
                return cls.PRICES[key][token_type]
        return 0
class TokenUsageTracker:
    def __init__(self, model: str = ""):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.usage_history = []
    def add_usage(self, model, input_tokens, output_tokens):
        input_cost = (input_tokens / 1000) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000) * ModelPricing.get_price(model, "output")
        total_cost = input_cost + output_cost
        usage_record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": ModelPricing.get_price(model, "input"),
                "output_price": ModelPricing.get_price(model, "output")
            }
        }
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total_cost
        self.usage_history.append(usage_record)
        return usage_record
    def get_summary(self):
        return {
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": len(self.usage_history),
            "history": self.usage_history
        }
class AsyncLLM:
    def __init__(self, config, system_msg:str = None, max_completion_tokens:int = None):
        if isinstance(config, str):
            llm_name = config
            config = LLMsConfig.default().get(llm_name)
        self.config = config
        self.aclient = AsyncOpenAI(api_key=self.config.key, base_url=self.config.base_url)
        self.sys_msg = system_msg
        self.usage_tracker = TokenUsageTracker(model=self.config.model)
        self._usage_lock = threading.Lock()
        self.max_completion_tokens = max_completion_tokens
    async def __call__(self, prompt, max_tokens=None):
        try:
            message = []
            if self.sys_msg is not None:
                message.append({
                    "content": self.sys_msg,
                    "role": "system"
                })
            if isinstance(prompt, str):
                message.append({"role": "user", "content": prompt})
            elif isinstance(prompt, list):
                message.append({"role": "user", "content": prompt})
            else:
                raise ValueError(f"prompt must be str or list, got {type(prompt)}")
            tokens_to_use = max_tokens if max_tokens is not None else self.max_completion_tokens
            is_claude = "claude" in self.config.model.lower()
            sampling_params = (
                {"temperature": self.config.temperature}
                if is_claude
                else {"temperature": self.config.temperature, "top_p": self.config.top_p}
            )
            template_extra = _chat_template_extra_body(self.config)
            create_kwargs: dict[str, Any] = {}
            if template_extra is not None:
                create_kwargs["extra_body"] = template_extra
            if self.config.model == "gemini-3-flash-preview":
                response = await self.aclient.chat.completions.create(
                    model=self.config.model,
                    messages=message,
                    max_tokens=tokens_to_use,
                    **sampling_params,
                    reasoning_effort="high",
                    **create_kwargs,
                )
            elif tokens_to_use is not None and "o3" in self.config.model:
                response = await self.aclient.chat.completions.create(
                    model=self.config.model,
                    messages=message,
                    max_completion_tokens=tokens_to_use,
                    **sampling_params,
                    **create_kwargs,
                )
            elif self.config.model == "gemini-3-flash-preview":
                response = await self.aclient.chat.completions.create(
                    model=self.config.model,
                    messages=message,
                    max_tokens=tokens_to_use,
                    **sampling_params,
                    reasoning_effort="high",
                    **create_kwargs,
                )
            elif tokens_to_use is not None and "o3" not in self.config.model:
                response = await self.aclient.chat.completions.create(
                    model=self.config.model,
                    messages=message,
                    max_tokens=tokens_to_use,
                    **sampling_params,
                    **create_kwargs,
                )
            else:
                response = await self.aclient.chat.completions.create(
                    model=self.config.model,
                    messages=message,
                    **sampling_params,
                    **create_kwargs,
                )
            if response is None:
                logger.error("LLM API returned None response")
                return ""
            if not hasattr(response, 'choices') or not response.choices:
                logger.error("LLM API returned response without choices")
                return ""
            if not hasattr(response, 'usage') or response.usage is None:
                logger.warning("LLM API response missing usage data")
                ret = _assistant_message_text(response.choices[0].message)
                if ret:
                    logger.log_to_file(LogLevel.INFO, f"LLM Response: {ret}")
                    return ret
                return ""
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            with self._usage_lock:
                usage_record = self.usage_tracker.add_usage(
                    self.config.model,
                    input_tokens,
                    output_tokens
                )
            record_cost(self.config.model, input_tokens, output_tokens, usage_record["total_cost"])
            ret = _assistant_message_text(response.choices[0].message)
            logger.log_to_file(LogLevel.INFO, f"LLM Response: {ret}")
            return ret
        except Exception as e:
            logger.error(f"LLM API call failed: {type(e).__name__}: {e}")
            return ""
        ret = _assistant_message_text(response.choices[0].message)
        logger.log_to_file(LogLevel.INFO, f"LLM Response: {ret}")
        return ret
    def get_usage_summary(self):
        with self._usage_lock:
            s = self.usage_tracker.get_summary()
            return {
                "model": s["model"],
                "total_input_tokens": s["total_input_tokens"],
                "total_output_tokens": s["total_output_tokens"],
                "total_tokens": s["total_tokens"],
                "total_cost": s["total_cost"],
                "call_count": s["call_count"],
                "history": list(s["history"]),
            }
    async def generate_text_to_image(self, prompt: str) -> dict[str, Any]:
        try:
            response = await self(prompt)
            image_b64 = self._extract_image_from_response(response)
            if not image_b64:
                return {
                    "success": False,
                    "image_base64": None,
                    "prompt": prompt,
                    "error": "No image found in response",
                }
            return {
                "success": True,
                "image_base64": image_b64,
                "prompt": prompt,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "image_base64": None,
                "prompt": prompt,
                "error": str(e),
            }
    async def generate_image_to_image(
        self, prompt: str, reference_images: list[str]
    ) -> dict[str, Any]:
        try:
            content = []
            for img_b64 in reference_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    }
                )
            content.append({"type": "text", "text": prompt})
            response = await self(content)
            image_b64 = self._extract_image_from_response(response)
            if not image_b64:
                return {
                    "success": False,
                    "image_base64": None,
                    "prompt": prompt,
                    "error": "No image found in response",
                }
            return {
                "success": True,
                "image_base64": image_b64,
                "prompt": prompt,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "image_base64": None,
                "prompt": prompt,
                "error": str(e),
            }
    def _extract_image_from_response(self, response: str) -> str | None:
        if not isinstance(response, str):
            return None
        match = re.search(r"data:image/[^;]+;base64,([^)]+)", response)
        return match.group(1) if match else None
def create_llm_instance(
    llm_config,
    max_completion_tokens: Optional[int] = None,
) -> AsyncLLM:
    if isinstance(llm_config, LLMConfig):
        return AsyncLLM(llm_config, max_completion_tokens=max_completion_tokens)
    elif isinstance(llm_config, str):
        return AsyncLLM(llm_config, max_completion_tokens=max_completion_tokens)
    elif isinstance(llm_config, dict):
        llm_config = LLMConfig(llm_config)
        return AsyncLLM(llm_config, max_completion_tokens=max_completion_tokens)
    else:
        raise TypeError("llm_config must be an LLMConfig instance, a string, or a dictionary")
