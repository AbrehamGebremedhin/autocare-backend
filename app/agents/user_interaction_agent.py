from typing import Any, Dict
from langchain.prompts import PromptTemplate
from app.services.llm_service import LLMService
from app.utils.logger import Logger
from app.utils.websocket import manager  # WebSocket manager for broadcasting stages
import json

class UserInteractionAgent:
    """
    Agentic RAG that takes the result from the diagnostic agent and creates a user-facing message.
    """
    def __init__(self, **kwargs):
        self.llm_service = LLMService()
        self.logger = Logger("UserInteractionAgent")
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert automotive assistant. Your goal is to help the user diagnose and, if possible, resolve the issue themselves. Provide clear, step-by-step instructions for safe DIY troubleshooting and minor repairs. Only recommend seeing a mechanic if the issue is dangerous, requires specialized tools, or cannot be safely addressed by a typical car owner.

            Always include safety warnings before any potentially hazardous steps. Use simple language and explain technical terms. Do NOT recommend visiting a mechanic unless absolutely necessary. Try to empower the user to understand and address the problem first.

            Given the user's message and the diagnostic result, generate a clear, empathetic, and actionable message for the user.

            - Summarize the main diagnosis in simple, user-friendly terms.
            - Reference supporting evidence only if helpful.
            - Provide clear, step-by-step next actions or recommendations, including safety tips and what to check or do first.
            - Be concise, friendly, and avoid technical jargon unless necessary, but be specific and detailed in instructions.
            - If the diagnosis is inconclusive or more info is needed, politely and clearly ask the user for the exact information required to proceed, and explain why it is needed.
            - If the issue could be urgent or dangerous, highlight this and advise the user accordingly.
            - If the diagnosis includes a list of 'Other Possible Causes' (from the diagnosis tree), mention these to the user as additional things to consider or discuss with a mechanic.
            - Always end with actionable next steps, and if the user should consult a professional mechanic, say so.

            Input:
            - User message: {user_message}
            - Diagnosis result: {diagnosis_result}

            Output:
            - user_message: A message to display to the user.
            """
        )

    async def generate_user_message(self, user_message: str, diagnosis_result: Any) -> Dict[str, Any]:
        await manager.broadcast(json.dumps({"type": "stage", "stage": "Generating user-facing message"}))
        try:
            # Extract step_by_step_guide if present
            step_by_step_guide = None
            if isinstance(diagnosis_result, dict):
                step_by_step_guide = diagnosis_result.get('step_by_step_guide')
            elif isinstance(diagnosis_result, str):
                try:
                    parsed = json.loads(diagnosis_result)
                    step_by_step_guide = parsed.get('step_by_step_guide')
                except Exception:
                    step_by_step_guide = None
            prompt_vars = {
                "user_message": user_message,
                "diagnosis_result": str(diagnosis_result)
            }
            prompt = self.prompt.format(**prompt_vars)
            # Use the LLMService for direct prompt calls
            response = await self.llm_service.generate_response(prompt)
            await manager.broadcast(json.dumps({"type": "stage", "stage": "User message generated"}))
            return {
                "user_message": response,
                "success": True,
                "step_by_step_guide": step_by_step_guide
            }
        except Exception as e:
            await manager.broadcast(json.dumps({"type": "stage", "stage": f"Error in user message generation - {type(e).__name__}"}))
            await self.logger.error(f"UserInteractionAgent error: {e}")
            return {
                "user_message": "Sorry, I couldn't generate a response at this time. Please try again later.",
                "success": False,
                "step_by_step_guide": None
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
