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
    def __init__(
        self,
        websocket_manager: IWebSocketManager = None,
        llm_service: LLMService = None,
        **kwargs
    ):
        super().__init__(websocket_manager=websocket_manager, logger_name="UserInteractionAgent", **kwargs)
        self.llm_service = llm_service or LLMService()
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert automotive assistant specializing in empowering DIY car repair enthusiasts. Your primary mission is to transform complex automotive diagnostic information into clear, actionable guidance that enables users to successfully diagnose and repair their vehicles themselves.

            Your users are motivated DIY enthusiasts who want to:
            - Understand what's wrong with their car at a technical level
            - Learn how to fix it themselves with detailed guidance
            - Save money by doing their own repairs
            - Build their automotive knowledge and skills
            - Only seek professional help when absolutely necessary

            CORE APPROACH:
            - Assume the user wants to do the repair themselves
            - Provide detailed technical explanations that build confidence
            - Break down complex procedures into manageable steps
            - Include professional tips and tricks
            - Explain the reasoning behind each step
            - Provide multiple approaches when possible
            - Focus on practical, hands-on guidance

            IMPORTANT:
            - Do NOT ask for car make, model, or year; this information is already available
            - Do NOT refer users to manuals; extract and use information directly
            - Present complex diagnostic data in an organized, accessible format
            - Prioritize practical DIY solutions over professional recommendations
            - Build user confidence through detailed explanations

            Transform the comprehensive diagnostic result into practical DIY guidance that empowers the user to successfully complete their own repairs.

            CRITICAL: Always include relevant follow-up questions to gather more information for better diagnosis and repair guidance. Consider:
            - Clarifying symptoms or timing
            - Understanding user's experience level
            - Confirming environmental factors
            - Gathering repair history details
            - Assessing available tools/workspace
            - Understanding urgency and constraints

            Return your response as a JSON object with the following DIY-focused fields:
            - technical_diagnosis: Detailed technical explanation of the problem for DIY understanding
            - primary_repair_procedure: The most likely repair approach with complete details
            - diy_difficulty_assessment: Honest assessment of difficulty level and what's involved
            - required_tools_and_parts: Comprehensive tool and parts list with specifications
            - step_by_step_repair_guide: Detailed repair instructions with professional insights
            - alternative_approaches: Other ways to tackle the problem if the primary approach doesn't work
            - diagnostic_verification: How to confirm the diagnosis before starting repairs
            - cost_and_time_analysis: Realistic cost and time expectations vs professional service
            - safety_protocols: Specific safety measures for this repair (not generic warnings)
            - troubleshooting_guide: What to do when things don't go as expected
            - quality_verification: How to test and verify the repair was successful
            - learning_insights: Technical knowledge gained from this repair that applies elsewhere
            - upgrade_opportunities: Performance or reliability improvements possible during repair
            - maintenance_prevention: How to prevent this problem from recurring
            - followup_questions: Array of specific questions to gather more diagnostic information and improve repair guidance

            Input:
            - User message: {user_message}
            - Diagnosis result: {diagnosis_result}

            Output:
            {{"technical_diagnosis": "...",
            "primary_repair_procedure": {{"name": "...", "difficulty": "...", "overview": "..."}},
            "diy_difficulty_assessment": {{"level": "...", "challenges": ["..."], "success_factors": ["..."]}},
            "required_tools_and_parts": {{"tools": [{{...}}], "parts": [{{...}}], "consumables": [{{...}}]}},
            "step_by_step_repair_guide": [{{...}}],
            "alternative_approaches": [{{...}}],
            "diagnostic_verification": [{{...}}],
            "safety_protocols": [{{...}}],
            "troubleshooting_guide": [{{...}}],
            "quality_verification": [{{...}}],
            "learning_insights": [{{...}}],
            "upgrade_opportunities": [{{...}}],
            "maintenance_prevention": [{{...}}],
            "followup_questions": [
                {{"question": "...", "purpose": "...", "category": "symptom_clarification|experience_level|environmental_factors|repair_history|tools_workspace|urgency_constraints"}},
                {{"question": "...", "purpose": "...", "category": "..."}}
            ]
            }}
            Ensure the JSON is valid and well-formed. Focus on empowering DIY repair success.
            Return ONLY a valid JSON object. Do NOT include any extra text, markdown, explanations outside the JSON or surrounded by backticks.
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
            # --- Enhanced DIY-Focused: Unpack and merge rich diagnosis fields ---
            merged_diagnosis = ""
            merged_repair_procedures = []
            merged_tools_and_parts = {"tools": [], "parts": [], "consumables": []}
            merged_diagnostic_procedures = []
            merged_safety_protocols = []
            merged_quality_assurance = []
            merged_maintenance_schedule = {}
            
            if isinstance(diagnosis_result, dict):
                # Main technical summary
                main_diag = diagnosis_result.get("diagnosis_summary") or diagnosis_result.get("diagnosis") or ""
                merged_diagnosis += main_diag
                
                # Add system education for better understanding
                system_edu = diagnosis_result.get("system_education", {})
                if system_edu:
                    merged_diagnosis += "\n\n**🔧 TECHNICAL BACKGROUND:**"
                    if system_edu.get("affected_system"):
                        merged_diagnosis += f"\n**System:** {system_edu.get('affected_system')}"
                    if system_edu.get("how_it_works"):
                        merged_diagnosis += f"\n**How it works:** {system_edu.get('how_it_works')}"
                    if system_edu.get("common_failure_modes"):
                        merged_diagnosis += f"\n**Common failures:** {', '.join(system_edu.get('common_failure_modes', []))}"
                    merged_system_education = system_edu
                
                # Add supporting evidence with DIY insights
                evidence = diagnosis_result.get("supporting_evidence", [])
                if evidence:
                    merged_diagnosis += "\n\n**📚 SUPPORTING EVIDENCE:**"
                    for ev in evidence:
                        source = ev.get("source", "Unknown")
                        evidence_text = ev.get("evidence", "")
                        diy_tips = ev.get("diy_tips", "")
                        specific_details = ev.get("specific_details", "")
                        
                        if evidence_text:
                            merged_diagnosis += f"\n• **{source.title()}:** {evidence_text}"
                            if specific_details:
                                merged_diagnosis += f" | Details: {specific_details}"
                            if diy_tips:
                                merged_diagnosis += f" | DIY Tip: {diy_tips}"
                
                # Process DIY repair procedures
                repair_procedures = diagnosis_result.get("diy_repair_procedures", [])
                for proc in repair_procedures:
                    proc_info = {
                        "name": proc.get("procedure_name", "Repair Procedure"),
                        "difficulty": proc.get("difficulty_level", "Unknown"),
                        "time": proc.get("estimated_time", "Unknown"),
                        "steps": proc.get("detailed_steps", []),
                        "troubleshooting": proc.get("troubleshooting", [])
                    }
                    merged_repair_procedures.append(proc_info)
                    
                    # Merge tools and parts
                    if proc.get("required_tools"):
                        merged_tools_and_parts["tools"].extend(proc.get("required_tools", []))
                    if proc.get("required_parts"):
                        merged_tools_and_parts["parts"].extend(proc.get("required_parts", []))
                
                # Process alternative diagnoses with repair focus
                alt_diags = diagnosis_result.get("alternative_diagnoses", [])
                if alt_diags:
                    merged_diagnosis += "\n\n**🔍 ALTERNATIVE DIAGNOSES:**"
                    for alt in alt_diags:
                        name = alt.get("name", "Unknown")
                        likelihood = alt.get("likelihood", "")
                        repair_approach = alt.get("repair_approach", "")
                        difficulty = alt.get("difficulty_assessment", "")
                        cost = alt.get("cost_estimate", "")
                        
                        merged_diagnosis += f"\n\n• **{name}** (Likelihood: {likelihood})"
                        if repair_approach:
                            merged_diagnosis += f"\n  Repair approach: {repair_approach}"
                        if difficulty:
                            merged_diagnosis += f"\n  DIY difficulty: {difficulty}"
                        if cost:
                            merged_diagnosis += f"\n  Estimated cost: {cost}"
                
                # Merge diagnostic procedures
                merged_diagnostic_procedures = diagnosis_result.get("diagnostic_procedures", [])
                
                # Merge safety protocols
                merged_safety_protocols = diagnosis_result.get("safety_protocols", [])
                
                # Merge quality assurance
                merged_quality_assurance = diagnosis_result.get("quality_assurance", [])
                
                # Merge maintenance schedule
                merged_maintenance_schedule = diagnosis_result.get("maintenance_schedule", {})
                
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
                    "technical_diagnosis": merged_diagnosis or "Could not parse response.",
                    "primary_repair_procedure": {"name": "Unknown", "difficulty": "Unknown", "overview": "Unable to determine repair procedure"},
                    "diy_difficulty_assessment": {"level": "Unknown", "challenges": [], "success_factors": []},
                    "required_tools_and_parts": merged_tools_and_parts,
                    "step_by_step_repair_guide": [],
                    "alternative_approaches": [],
                    "diagnostic_verification": merged_diagnostic_procedures,
                    "safety_protocols": merged_safety_protocols,
                    "troubleshooting_guide": [],
                    "quality_verification": merged_quality_assurance,
                    "learning_insights": [],
                    "upgrade_opportunities": [],
                    "maintenance_prevention": merged_maintenance_schedule,
                    "followup_questions": [
                        {"question": "Can you provide more details about when this problem occurs?", "purpose": "Better understand symptom timing", "category": "symptom_clarification"},
                        {"question": "What is your experience level with car repairs?", "purpose": "Tailor repair instructions appropriately", "category": "experience_level"}
                    ]
                }
            
            # Ensure followup_questions is always present
            if "followup_questions" not in parsed_response or not parsed_response["followup_questions"]:
                # Generate default follow-up questions based on diagnosis content
                default_questions = []
                
                # Check if we need more symptom clarification
                if "diagnosis" in str(diagnosis_result).lower() and len(user_message.split()) < 10:
                    default_questions.append({
                        "question": "Can you describe the symptoms in more detail - when do they occur and what exactly happens?",
                        "purpose": "Get more specific symptom information for accurate diagnosis",
                        "category": "symptom_clarification"
                    })
                
                # Check experience level
                if "repair" in str(diagnosis_result).lower() or "fix" in str(diagnosis_result).lower():
                    default_questions.append({
                        "question": "What's your experience level with automotive repairs?",
                        "purpose": "Customize repair instructions to your skill level",
                        "category": "experience_level"
                    })
                
                # Check for tools/workspace
                if any(tool_word in str(diagnosis_result).lower() for tool_word in ["tool", "wrench", "socket", "jack"]):
                    default_questions.append({
                        "question": "Do you have access to basic automotive tools and a suitable workspace?",
                        "purpose": "Ensure you have necessary equipment before starting",
                        "category": "tools_workspace"
                    })
                
                # Check urgency
                default_questions.append({
                    "question": "How urgent is this repair - are you able to drive the car safely right now?",
                    "purpose": "Assess safety and prioritize repair approach",
                    "category": "urgency_constraints"
                })
                
                parsed_response["followup_questions"] = default_questions[:3]  # Limit to 3 questions
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
                "followup_questions": [
                    {"question": "Can you provide more details about the symptoms you're experiencing?", "purpose": "Better understand the problem", "category": "symptom_clarification"},
                    {"question": "When did this problem first start occurring?", "purpose": "Understand problem timeline", "category": "symptom_clarification"}
                ],
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
                # Compose a comprehensive DIY-focused user message from all fields
                user_message_text = {
                    "technical_diagnosis": result.get("technical_diagnosis", ""),
                    "primary_repair_procedure": result.get("primary_repair_procedure", {}),
                    "diy_difficulty_assessment": result.get("diy_difficulty_assessment", {}),
                    "required_tools_and_parts": result.get("required_tools_and_parts", {}),
                    "step_by_step_repair_guide": result.get("step_by_step_repair_guide", []),
                    "alternative_approaches": result.get("alternative_approaches", []),
                    "diagnostic_verification": result.get("diagnostic_verification", []),
                    "safety_protocols": result.get("safety_protocols", []),
                    "troubleshooting_guide": result.get("troubleshooting_guide", []),
                    "quality_verification": result.get("quality_verification", []),
                    "learning_insights": result.get("learning_insights", []),
                    "upgrade_opportunities": result.get("upgrade_opportunities", []),
                    "maintenance_prevention": result.get("maintenance_prevention", {}),
                    "followup_questions": result.get("followup_questions", [])
                }
                return {
                    "success": result.get("success", False),
                    "user_message": user_message_text,
                    "followup_questions": result.get("followup_questions", []),  # Also include at top level for backwards compatibility
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
                        "followup_questions": [
                            {"question": "Could you please describe the problem you're experiencing with your car?", "purpose": "Get basic problem description", "category": "symptom_clarification"},
                            {"question": "When did this issue first occur?", "purpose": "Understand problem timeline", "category": "symptom_clarification"}
                        ],
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
