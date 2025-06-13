from app.services.base_service import BaseService
from langchain_community.llms import Ollama
from typing import Optional, Any, Dict, Callable
import logging

class LLMService(BaseService):
    """
    Service for interacting with a locally running Ollama LLM using LangChain.
    Supports dynamic model switching, streaming, and advanced parameterization.
    """
    def __init__(self, model_name: str = "llama3", websocket_manager=None, **default_params):
        super().__init__(websocket_manager=websocket_manager)
        self.model_name = model_name
        self.default_params = default_params
        self.llm = Ollama(model=model_name, **default_params)
        self.logger = logging.getLogger(__name__)

    def set_model(self, model_name: str, **params) -> None:
        """
        Dynamically switch the LLM model and update parameters.
        """
        self.model_name = model_name
        self.default_params.update(params)
        self.llm = Ollama(model=model_name, **self.default_params)
        self.logger.info(f"Switched Ollama model to {model_name} with params {self.default_params}")

    async def generate_response(self, prompt: str, stream: bool = False, **kwargs) -> str:
        """
        Generate a response from the LLM for a given prompt.
        Supports streaming if enabled.
        """
        await self._rate_limit()
        import asyncio
        loop = asyncio.get_event_loop()
        params = {**self.default_params, **kwargs}
        try:
            if stream and hasattr(self.llm, 'stream'):
                # Streaming response generator
                def stream_fn():
                    return self.llm.stream(prompt, **params)
                response = await loop.run_in_executor(None, stream_fn)
                return response  # Caller should handle the generator
            else:
                response = await loop.run_in_executor(
                    None, lambda: self.llm.invoke(prompt, **params)
                )
                return response
        except Exception as e:
            self.logger.error(f"LLMService error: {e}")
            return f"Error: {e}"

    async def perform_action(self, prompt: str, **kwargs) -> Any:
        """
        Implementation of the abstract method from BaseService.
        Generates a response for the given prompt.
        """
        return await self.generate_response(prompt, **kwargs)

    def generate_response_sync(self, prompt: str, **kwargs) -> str:
        """
        Synchronous version of generate_response for non-async environments.
        """
        params = {**self.default_params, **kwargs}
        try:
            return self.llm.invoke(prompt, **params)
        except Exception as e:
            self.logger.error(f"LLMService sync error: {e}")
            return f"Error: {e}"
