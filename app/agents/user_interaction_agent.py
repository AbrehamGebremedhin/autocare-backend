from typing import Any, Dict
from langchain.prompts import PromptTemplate
from app.services.llm_service import LLMService
from app.utils.logger import Logger
from app.utils.websocket import manager  # WebSocket manager for broadcasting stages
from app.agents.base_agent import BaseAgent
import json
import re
import traceback

class UserInteractionAgent(BaseAgent):
    """
    Agentic RAG that takes the result from the diagnostic agent and creates a user-facing message.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_service = LLMService()
        self.logger = Logger("UserInteractionAgent")
        self.prompt = PromptTemplate.from_template(
"""
You are an expert automotive assistant. Your goal is to help the user diagnose and, if possible, resolve the issue themselves. Provide clear, step-by-step instructions for safe DIY troubleshooting and minor repairs. Only recommend seeing a mechanic if the issue is dangerous, requires specialized tools, or cannot be safely addressed by a typical car owner.

Always include safety warnings before any potentially hazardous steps. Use simple language and explain technical terms. Do NOT recommend visiting a mechanic unless absolutely necessary. Try to empower the user to understand and address the problem first.

IMPORTANT:
- Do NOT ask the user for car make, model, or year; you already have this information.
- Do NOT tell the user to refer to the owner's manual; use the information from the manual directly in your response.

Given the user's message and the diagnostic result, generate a clear, empathetic, and actionable message for the user.

Return your response as a JSON object with the following fields:
- diagnosis: A summary of the main diagnosis in simple, user-friendly terms.
- actionable_steps: A list of clear, step-by-step next actions or recommendations for the user, including what to check or do first.
- needed_tools: A list of tools or materials the user will need to perform the steps (if any).
- safety_note: Important safety warnings or notes relevant to the steps.
- followup_questions: A list of follow-up questions for the user if more information is needed, or an empty list if not.
- confidence: Your confidence in the diagnosis (e.g., High, Medium, Low) and a brief explanation.

Input:
- User message: {user_message}
- Diagnosis result: {diagnosis_result}

Output:
{{"diagnosis": "...",
"actionable_steps": ["...", "..."],
"needed_tools": ["...", "..."],
"safety_note": "...",
"followup_questions": ["...", "..."],
"confidence": "..."
}}
"""
        )

    def _extract_first_json_object(self, text: str) -> str:
        """
        Extract the first valid JSON object from a string, handling nested braces.
        Returns the JSON string or None if not found.
        """
        start = text.find('{')
        if start == -1:
            return None
        stack = []
        for i in range(start, len(text)):
            if text[i] == '{':
                stack.append('{')
            elif text[i] == '}':
                if stack:
                    stack.pop()
                if not stack:
                    return text[start:i+1]
        return None

    async def generate_user_message(self, user_message: str, diagnosis_result: Any) -> Dict[str, Any]:
        await manager.broadcast(json.dumps({"type": "stage", "stage": "Generating user-facing message"}))
        await self.logger.info(f"UserInteractionAgent - Generating user-facing message for: {user_message}")
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
            response = self._sanitize_output(response)
            # Try to parse the response as JSON, robustly
            raw_response = response.strip()
            parsed_response = None
            try:
                # Try direct JSON parse first
                parsed_response = json.loads(raw_response)
            except Exception:
                # Use stack-based extraction for nested JSON
                json_str = self._extract_first_json_object(raw_response)
                if json_str:
                    try:
                        parsed_response = json.loads(json_str)
                    except Exception as e2:
                        await self.logger.error(f"[error] - Failed to parse extracted JSON: {e2}\nRaw LLM response: {raw_response}\nTraceback: {traceback.format_exc()}")
                else:
                    await self.logger.error(f"[error] - No JSON object found in LLM response. Raw response: {raw_response}")
            if not parsed_response:
                await self.logger.error(f"[error] - Using fallback response. Raw LLM response: {raw_response}")
                parsed_response = {
                    "diagnosis": "Could not parse response.",
                    "actionable_steps": [],
                    "needed_tools": [],
                    "safety_note": "",
                    "followup_questions": [],
                    "confidence": "Low: Could not parse response."
                }
            await manager.broadcast(json.dumps({"type": "stage", "stage": "User message generated"}))
            return {
                **parsed_response,
                "success": True,
                "step_by_step_guide": step_by_step_guide
            }
        except Exception as e:
            await manager.broadcast(json.dumps({"type": "stage", "stage": f"Error in user message generation - {type(e).__name__}"}))
            await self.logger.error(f"UserInteractionAgent error: {e}\nTraceback: {traceback.format_exc()}")
            return {
                "diagnosis": None,
                "actionable_steps": [],
                "needed_tools": [],
                "safety_note": "",
                "followup_questions": [],
                "confidence": "Low: Exception occurred.",
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
            dict: Contains 'success', 'diagnosis', 'actionable_steps', 'needed_tools', 'safety_note', 'followup_questions', 'confidence', and optionally 'error' and 'error_type'.
        """
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = await self.generate_user_message(user_message, diagnosis_result)
                # Compose a user-facing message from all fields
                user_message_text = {
                    "diagnosis": result.get("diagnosis", ""),
                    "actionable_steps": result.get("actionable_steps", []),
                    "needed_tools": result.get("needed_tools", []),
                    "safety_note": result.get("safety_note", ""),
                    "followup_questions": result.get("followup_questions", []),
                    "confidence": result.get("confidence", "")
                }
                return {
                    "success": result.get("success", False),
                    "user_message": user_message_text,
                    "error": None,
                    "error_type": None
                }
            except Exception as e:
                await self.logger.error(f"Process error (attempt {attempt+1}): {e}")
                if attempt == max_retries:
                    return {
                        "success": False,
                        "user_message": "",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
