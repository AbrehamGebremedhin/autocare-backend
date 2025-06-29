from typing import Any, Dict
from langchain.prompts import PromptTemplate
from app.services.llm_service import LLMService
from app.utils.logger import Logger
from app.core.interfaces import IWebSocketManager
from app.agents.base_agent import BaseAgent
from app.utils.message_types import MessageSource
from app.utils.monitoring import monitor_and_handle
from app.utils.json_utils import serialize_datetimes
import json
import re
import traceback

class UserInteractionAgent(BaseAgent):
    """
    Agentic RAG that takes the result from the diagnostic agent and creates a user-facing message.
    """
    def __init__(self, websocket_manager: IWebSocketManager = None, **kwargs):
        super().__init__(websocket_manager=websocket_manager, logger_name="UserInteractionAgent", **kwargs)
        self.llm_service = LLMService()
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert automotive assistant. Your goal is to help the user diagnose and, if possible, resolve the issue themselves. Provide clear, step-by-step instructions for safe DIY troubleshooting and minor repairs. Only recommend seeing a mechanic if the issue is dangerous, requires specialized tools, or cannot be safely addressed by a typical car owner.

            Always include safety warnings before any potentially hazardous steps. Use simple language and explain technical terms. Do NOT recommend visiting a mechanic unless absolutely necessary. Try to empower the user to understand and address the problem first.

            IMPORTANT:
            - Do NOT ask the user for car make, model, or year; you already have this information.
            - Do NOT tell the user to refer to the owner's manual; use the information from the manual directly in your response.

            Given the user's message and the diagnostic result, generate a clear, empathetic, and actionable message for the user.

            Return your response as a JSON object with the following fields:
            - diagnosis: A detailed diagnosis of the main issue of the problem.
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
            Ensure the JSON is valid and well-formed. Do NOT include any additional text, markdown, or explanations outside the JSON object.
            Return ONLY a valid JSON object as specified above. Do NOT include any extra text, markdown, explanations outside the JSON or surrounded by backticks.
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

    async def generate_user_message(self, user_message: str, diagnosis_result: Any, websocket=None, session_id=None) -> dict:
        if websocket:
            await self.send_ws_stage(websocket, "Generating user-facing message", MessageSource.CHAT_SERVICE, session_id=session_id)
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
            # --- NEW: Unpack and merge rich diagnosis fields ---
            merged_diagnosis = ""
            merged_actionable_steps = []
            merged_needed_tools = []
            merged_safety_note = ""
            merged_followup_questions = []
            merged_confidence = ""
            if isinstance(diagnosis_result, dict):
                # Main summary
                main_diag = diagnosis_result.get("diagnosis_summary") or diagnosis_result.get("diagnosis") or ""
                merged_diagnosis += main_diag
                # Add alternative diagnoses
                alt_diags = diagnosis_result.get("alternative_diagnoses", [])
                if alt_diags:
                    merged_diagnosis += "\n\nOther possible scenarios to consider:"
                    for alt in alt_diags:
                        name = alt.get("name", "Unknown")
                        likelihood = alt.get("likelihood", "")
                        features = ", ".join(alt.get("distinguishing_features", []))
                        notes = alt.get("notes", "")
                        merged_diagnosis += f"\n- {name} (Likelihood: {likelihood})"
                        if features:
                            merged_diagnosis += f". Distinguishing features: {features}."
                        if notes:
                            merged_diagnosis += f" {notes}"
                        # Optionally add actionable steps for each alt
                        alt_steps = alt.get("actionable_steps", [])
                        if alt_steps:
                            merged_actionable_steps.append(f"If {name}: " + "; ".join(alt_steps))
                # Add uncommon but important scenarios
                uncommon = diagnosis_result.get("uncommon_but_important_scenarios", [])
                if uncommon:
                    merged_diagnosis += "\n\nUncommon but important scenarios:"
                    for u in uncommon:
                        merged_diagnosis += f"\n- {u.get('name', 'Unknown')}: {u.get('description', '')} (Likelihood: {u.get('likelihood', '')})"
                # Add main actionable steps
                main_steps = diagnosis_result.get("step_by_step_guide") or diagnosis_result.get("actionable_steps") or []
                if isinstance(main_steps, list):
                    merged_actionable_steps = main_steps + merged_actionable_steps
                # Merge other fields
                merged_needed_tools = diagnosis_result.get("needed_tools", [])
                merged_safety_note = diagnosis_result.get("safety_note", "")
                merged_followup_questions = diagnosis_result.get("followup_questions", [])
                merged_confidence = diagnosis_result.get("confidence", diagnosis_result.get("confidence", ""))
            else:
                merged_diagnosis = str(diagnosis_result)
            prompt_vars = {
                "user_message": user_message,
                "diagnosis_result": merged_diagnosis
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
                    "diagnosis": merged_diagnosis or "Could not parse response.",
                    "actionable_steps": merged_actionable_steps,
                    "needed_tools": merged_needed_tools,
                    "safety_note": merged_safety_note,
                    "followup_questions": merged_followup_questions,
                    "confidence": merged_confidence or "Low: Could not parse response."
                }
            # Serialize datetimes before sending to websocket or returning
            parsed_response = serialize_datetimes(parsed_response)
            if websocket:
                await self.send_ws_result(websocket, "User message generated", MessageSource.CHAT_SERVICE, session_id=session_id, details=parsed_response)
            await self.broadcast_stage(json.dumps({"type": "stage", "stage": "User message generated"}))
            return {
                **parsed_response,
                "success": True,
                "step_by_step_guide": step_by_step_guide
            }
        except Exception as e:
            if websocket:
                await self.send_ws_error(websocket, f"Error in user message generation - {type(e).__name__}", MessageSource.CHAT_SERVICE, session_id=session_id, details={"error": str(e)})
            await self.broadcast_stage(json.dumps({"type": "stage", "stage": f"Error in user message generation - {type(e).__name__}"}))
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

    @monitor_and_handle("UserInteractionAgent")
    async def process(self, user_message: str, diagnosis_result: Any, websocket=None, session_id=None) -> dict:
        """
        Accepts the user message and diagnosis result, generates a user-facing message, and returns the result with success status and error handling.
        Args:
            user_message (str): The user's original message.
            diagnosis_result (Any): The result from the diagnostic agent.
        Returns:
            dict: Contains 'success', 'diagnosis', 'actionable_steps', 'needed_tools', 'safety_note', 'followup_questions', 'confidence', and optionally 'error' and 'error_type'.
        """
        # Convert DiagnosisTreeNode to dict if present in diagnosis_result
        def serialize(obj):
            from app.utils.diagnosis_tree import DiagnosisTreeNode
            if isinstance(obj, DiagnosisTreeNode):
                return obj.to_dict()
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [serialize(i) for i in obj]
            return obj
        diagnosis_result = serialize(diagnosis_result)
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = await self.generate_user_message(user_message, diagnosis_result, websocket=websocket, session_id=session_id)
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
                    if websocket:
                        await self.send_ws_error(websocket, f"Error in user message generation - {type(e).__name__}", MessageSource.CHAT_SERVICE, session_id=session_id, details={"error": str(e)})
                    return {
                        "success": False,
                        "user_message": "",
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
