from typing import Any, Dict
from langchain.prompts import PromptTemplate
from app.services.llm_service import LLMService
from app.utils.logger import Logger
from app.utils.websocket import manager  # WebSocket manager for broadcasting stages

class UserInteractionAgent:
    """
    Agentic RAG that takes the result from the diagnostic agent and creates a user-facing message.
    """
    def __init__(self, **kwargs):
        self.llm_service = LLMService()
        self.logger = Logger("UserInteractionAgent")
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert automotive assistant. Given the user's message and the diagnostic result, generate a clear, empathetic, and actionable message for the user.

            - Summarize the main diagnosis in simple terms.
            - Reference supporting evidence only if helpful.
            - Provide clear next steps or recommendations.
            - Be concise, friendly, and avoid technical jargon unless necessary.
            - If the diagnosis is inconclusive or more info is needed, politely ask the user for more details.

            Input:
            - User message: {user_message}
            - Diagnosis result: {diagnosis_result}

            Output:
            - user_message: A message to display to the user.
            """
        )

    async def generate_user_message(self, user_message: str, diagnosis_result: Any) -> Dict[str, Any]:
        await manager.broadcast("Stage: Generating user-facing message")
        try:
            prompt_vars = {
                "user_message": user_message,
                "diagnosis_result": str(diagnosis_result)
            }
            prompt = self.prompt.format(**prompt_vars)
            response = await self.llm_service.generate_response(prompt)
            await manager.broadcast("Stage: User message generated")
            return {"user_message": response, "success": True}
        except Exception as e:
            await manager.broadcast(f"Stage: Error in user message generation - {type(e).__name__}")
            await self.logger.error(f"UserInteractionAgent error: {e}")
            return {
                "user_message": "Sorry, I couldn't generate a response at this time. Please try again later.",
                "success": False
            }

    def get_langchain_llm(self):
        """
        For advanced LangChain integrations (e.g., chains), use this accessor.
        """
        return self.llm_service.get_llm()

    async def process(self, user_message: str, diagnosis_result: Any) -> dict:
        """
        Accepts the user message and diagnosis result, generates a user-facing message, and returns the result with success status and error handling.
        Args:
            user_message (str): The user's original message.
            diagnosis_result (Any): The result from the diagnostic agent.
        Returns:
            dict: Contains 'success', 'result', and optionally 'error' and 'error_type'.
        """
        try:
            result = await self.generate_user_message(user_message, diagnosis_result)
            return {
                "success": result.get("success", False),
                "result": result.get("user_message"),
                "error": None,
                "error_type": None
            }
        except Exception as e:
            await self.logger.error(f"Process error: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "error_type": type(e).__name__
            }
