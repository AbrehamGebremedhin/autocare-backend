from app.services.base_service import BaseService
from langchain_ollama import OllamaLLM
from typing import Optional, Any, Dict, Callable
import logging
from app.utils.redis_cache import redis_cache
from tenacity import retry, stop_after_attempt, wait_fixed
from string import Template

class LLMService(BaseService):
    """
    Service for interacting with a locally running Ollama LLM using LangChain.
    Handles retries, prompt templates, Redis caching, and versioning.
    Makes it easy to swap/update LLM backends.
    """
    def __init__(self, model_name: str = "gemma3:12b", version: str = "v1", websocket_manager=None, **default_params):
        super().__init__(websocket_manager=websocket_manager)
        self.model_name = model_name
        self.version = version
        self.default_params = default_params
        self.llm = OllamaLLM(model=model_name, **default_params)
        self.logger = logging.getLogger(__name__)

    def set_model(self, model_name: str, version: str = None, **params) -> None:
        self.model_name = model_name
        if version:
            self.version = version
        self.default_params.update(params)
        self.llm = OllamaLLM(model=model_name, **self.default_params)
        self.logger.info(f"Switched Ollama model to {model_name} (version {self.version}) with params {self.default_params}")

    def render_prompt(self, template: str, variables: dict) -> str:
        return Template(template).safe_substitute(variables)

    def _cache_key(self, prompt: str, params: dict) -> str:
        import hashlib, json
        key = f"llm:{self.model_name}:{self.version}:{prompt}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key.encode()).hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def _call_llm(self, prompt: str, params: dict, stream: bool = False):
        import asyncio
        loop = asyncio.get_event_loop()
        if stream and hasattr(self.llm, 'stream'):
            def stream_fn():
                return self.llm.stream(prompt, **params)
            return await loop.run_in_executor(None, stream_fn)
        else:
            return await loop.run_in_executor(None, lambda: self.llm.invoke(prompt, **params))

    async def generate_response(self, prompt: str, stream: bool = False, use_cache: bool = True, **kwargs) -> str:
        await self._rate_limit()
        params = {**self.default_params, **kwargs}
        cache_key = self._cache_key(prompt, params)
        if use_cache:
            cached = await redis_cache.get(cache_key)
            if cached:
                return cached
        try:
            response = await self._call_llm(prompt, params, stream=stream)
            if use_cache:
                await redis_cache.set(cache_key, response)
            return response
        except Exception as e:
            self.logger.error(f"LLMService error: {e}")
            return f"Error: {e}"

    async def generate_response_with_template(self, template: str, variables: dict, **kwargs) -> str:
        prompt = self.render_prompt(template, variables)
        return await self.generate_response(prompt, **kwargs)

    async def perform_action(self, prompt: str, **kwargs) -> Any:
        return await self.generate_response(prompt, **kwargs)

    def generate_response_sync(self, prompt: str, **kwargs) -> str:
        params = {**self.default_params, **kwargs}
        try:
            return self.llm.invoke(prompt, **params)
        except Exception as e:
            self.logger.error(f"LLMService sync error: {e}")
            return f"Error: {e}"

    def get_llm(self):
        """
        Returns the underlying LLM instance for advanced LangChain integrations.
        """
        return self.llm
