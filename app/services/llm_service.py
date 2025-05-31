from app.services.base_service import BaseService
from langchain_community.llms import Ollama
from typing import Optional, Any

class LLMService(BaseService):
    """
    Service for interacting with a locally running Ollama LLM using LangChain.
    """
    def __init__(self, model_name: str = "llama3", websocket_manager=None):
        super().__init__(websocket_manager=websocket_manager)
        self.model_name = model_name
        self.llm = Ollama(model=model_name)

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM for a given prompt.
        """
        await self._rate_limit()
        # LangChain's Ollama is synchronous, so run in thread executor
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.llm.invoke(prompt, **kwargs)
        )
        return response

    async def perform_action(self, prompt: str, **kwargs) -> Any:
        """
        Implementation of the abstract method from BaseService.
        Generates a response for the given prompt.
        """
        return await self.generate_response(prompt, **kwargs)
