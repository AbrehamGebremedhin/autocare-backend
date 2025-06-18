from typing import Any, Dict
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from app.utils.logger import Logger

class UserInteractionAgent:
    """
    Agentic RAG that takes the result from the diagnostic agent and creates a user-facing message.
    """
    def __init__(self, **kwargs):
        self.llm = OllamaLLM(model="gemma3:12b")
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
        """
        Generate a user-facing message based on the diagnosis and user message.
        Args:
            user_message (str): The user's original message.
            diagnosis_result (Any): The result from the diagnostic agent (structured or string).
        Returns:
            Dict[str, Any]: Contains the generated user message and success status.
        """
        try:
            prompt_vars = {
                "user_message": user_message,
                "diagnosis_result": str(diagnosis_result)
            }
            chain = self.llm | self.prompt
            response = await chain.ainvoke(prompt_vars) if hasattr(chain, "ainvoke") else chain.invoke(prompt_vars)
            return {"user_message": response, "success": True}
        except Exception as e:
            await self.logger.error(f"UserInteractionAgent error: {e}")
            return {
                "user_message": "Sorry, I couldn't generate a response at this time. Please try again later.",
                "success": False
            }

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
