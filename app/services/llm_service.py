from app.services.base_service import BaseService
from langchain_ollama import OllamaLLM
from typing import Optional, Any, Dict, Callable, Union
from app.core.interfaces import IWebSocketManager, ILogger
import logging
from app.utils.redis_cache import redis_cache
from tenacity import retry, stop_after_attempt, wait_fixed
from string import Template
from app.utils.message_types import MessageSource

class LLMService(BaseService):
    """
    Service for interacting with a locally running Ollama LLM using LangChain.
    Handles retries, prompt templates, Redis caching, and versioning.
    Makes it easy to swap/update LLM backends.
    """
    def __init__(
        self,
        model_name: str = "gemma3:12b",
        version: str = "v1",
        websocket_manager: IWebSocketManager = None,
        logger: Optional[ILogger] = None,
        llm: Optional[Any] = None,
        **default_params
    ):
        super().__init__(websocket_manager=websocket_manager)
        self.model_name = model_name
        self.version = version
        self.default_params = default_params
        self.llm = llm or OllamaLLM(model=model_name, **default_params)
        self.logger = logger or logging.getLogger(__name__)

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
        # Direct LLM call without tracing
        if stream and hasattr(self.llm, 'stream'):
            def stream_fn():
                return self.llm.stream(prompt, **params)
            return await loop.run_in_executor(None, stream_fn)
        else:
            return await loop.run_in_executor(None, lambda: self.llm.invoke(prompt, **params))

    async def generate_response(self, prompt: str, stream: bool = False, use_cache: bool = True, websocket=None, session_id=None, output_schema: Union[dict, Callable]=None, **kwargs) -> str:
        await self._rate_limit()
        params = {**self.default_params, **kwargs}
        cache_key = self._cache_key(prompt, params)
        if use_cache:
            cached = await redis_cache.get(cache_key)
            if cached:
                if websocket:
                    await self.send_ws_info(websocket, "LLM cache hit", MessageSource.CHAT_SERVICE, session_id=session_id, details={"cache_key": cache_key})
                # Validate cached output if schema is provided
                if output_schema:
                    return self._enforce_output_schema(cached, output_schema)
                return cached
        try:
            if websocket:
                await self.send_ws_progress(websocket, "LLM generating response", MessageSource.CHAT_SERVICE, 0.5, session_id=session_id)
            response = await self._call_llm(prompt, params, stream=stream)
            if use_cache:
                await redis_cache.set(cache_key, response)
            if websocket:
                await self.send_ws_result(websocket, "LLM response ready", MessageSource.CHAT_SERVICE, session_id=session_id, details={"response": response})
            # Enforce output structure if schema is provided
            if output_schema:
                return self._enforce_output_schema(response, output_schema)
            return response
        except Exception as e:
            self.logger.error(f"LLMService error: {e}")
            if websocket:
                await self.send_ws_error(websocket, f"LLMService error: {e}", MessageSource.CHAT_SERVICE, session_id=session_id, details={"error": str(e)})
            return f"Error: {e}"

    def _enforce_output_schema(self, response: str, schema: Union[dict, Callable]) -> str:
        """
        Enforce that the response matches the required output schema.
        If schema is a dict, check required keys. If callable, use as validator.
        Returns a JSON string matching the schema, or a default structure if invalid.
        """
        import json
        try:
            parsed = json.loads(response) if isinstance(response, str) else response
        except Exception:
            parsed = None
        # If schema is a callable (e.g., pydantic model or custom validator)
        if callable(schema):
            try:
                valid = schema(parsed)
                return json.dumps(valid) if not isinstance(valid, str) else valid
            except Exception:
                return json.dumps(self._default_structure(schema))
        # If schema is a dict of required keys
        if isinstance(schema, dict):
            if isinstance(parsed, dict) and all(k in parsed for k in schema.keys()):
                return json.dumps(parsed)
            else:
                return json.dumps(self._default_structure(schema))
        # If no schema matched, return as is
        return response

    def _default_structure(self, schema: Union[dict, Callable]) -> dict:
        """
        Return a default structure based on the schema.
        """
        if isinstance(schema, dict):
            return {k: v for k, v in schema.items()}
        if callable(schema):
            # Try to get default from callable, else return empty dict
            try:
                return schema()
            except Exception:
                return {}
        return {}

    async def generate_response_with_template(self, template: str, variables: dict, output_schema: Union[dict, Callable]=None, **kwargs) -> str:
        prompt = self.render_prompt(template, variables)
        return await self.generate_response(prompt, output_schema=output_schema, **kwargs)

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
