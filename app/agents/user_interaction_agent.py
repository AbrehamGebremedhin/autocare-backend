from typing import Any, Dict, Optional
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
        websocket_manager: Optional[IWebSocketManager] = None,
        llm_service: Optional[LLMService] = None,
        **kwargs
    ):
        """
        Initialize the UserInteractionAgent with dependency injection for testability.
        """
        super().__init__(websocket_manager=websocket_manager, logger_name="UserInteractionAgent", **kwargs)
        self.llm_service = llm_service or LLMService()
        
        # Simple response prompt for basic questions
        self.simple_response_prompt = PromptTemplate.from_template(
            """
            You are an automotive expert providing helpful, concise answers to general automotive questions.
            
            USER MESSAGE: "{user_message}"
            SESSION CONTEXT: "{session_context}"
            CONVERSATION CONTEXT: "{conversation_context}"
            
            Provide a helpful, informative response that directly answers the user's question.
            Keep it conversational but informative. Include practical tips where relevant.
            
            IMPORTANT: If conversation context is provided, use it to understand what the user is referring to.
            For example, if they previously asked about "diesel and benzene" and now ask "which one is more flammable",
            answer specifically about diesel vs benzene flammability, not general automotive fluids.
            
            If the question relates to an ongoing diagnostic session, reference that context appropriately.
            If you need more specific information to provide a better answer, ask targeted follow-up questions.
            
            Format your response as JSON:
            {{
                "response": "Your helpful response to the user's question",
                "followup_questions": [
                    {{"question": "...", "purpose": "...", "category": "clarification"}}
                ]
            }}
            
            Return ONLY valid JSON.
            """
        )
        
        self.prompt = PromptTemplate.from_template(
            """
            You are an expert automotive assistant specializing in empowering DIY car repair enthusiasts. Your primary mission is to transform complex automotive diagnostic information into clear, actionable guidance that enables users to successfully diagnose and repair their vehicles themselves.

            {session_context}

            IMPORTANT: If session context is provided above, ALWAYS keep your response focused on that original issue. 
            When users ask follow-up questions or mention additional symptoms, interpret them within the context 
            of the original problem. Do not deviate to unrelated automotive topics unless explicitly asked.

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

    def _clean_json_response(self, response_text: str) -> str:
        """Clean LLM response to extract valid JSON."""
        if not response_text:
            return "{}"
        
        # Remove markdown formatting
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Find JSON content
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
            return response_text[start_idx:end_idx + 1]
        
        return "{}"

    async def generate_simple_response(self, user_message: str, session_context: str = "", conversation_history: list = None) -> dict:
        """
        Generate a simple, conversational response for general questions.
        Now includes conversation history for context-aware responses.
        """
        try:
            # Build conversation context from history
            conversation_context = ""
            if conversation_history:
                await self.logger.info(f"[SIMPLE_RESPONSE] Processing {len(conversation_history)} conversation history messages")
                recent_messages = conversation_history[-3:]  # Last 3 messages for context
                for msg in recent_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if isinstance(content, dict):
                        # Extract the main text from complex responses
                        content = content.get('technical_diagnosis', str(content)[:200])
                    elif isinstance(content, str):
                        content = content[:200]  # Limit length
                    conversation_context += f"{role}: {content}\n"
                await self.logger.info(f"[SIMPLE_RESPONSE] Built conversation context: {conversation_context[:300]}")
            else:
                await self.logger.info("[SIMPLE_RESPONSE] No conversation history provided")
            
            formatted_prompt = self.simple_response_prompt.format(
                user_message=user_message,
                session_context=session_context or "No session context available",
                conversation_context=conversation_context or "No previous conversation context"
            )
            
            response = await self.llm_service.generate_response(formatted_prompt)
            
            # Handle AIMessage objects from LangChain
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Try to parse JSON, fall back to plain text if needed
            try:
                clean_response = self._clean_json_response(response_text)
                result = json.loads(clean_response)
                response_content = result.get("response", response_text)
                followup_questions = result.get("followup_questions", [])
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, use the raw response
                response_content = response_text
                followup_questions = []
            
            # Return in the expected format but only with simple response
            return {
                "technical_diagnosis": response_content,
                "primary_repair_procedure": {},
                "diy_difficulty_assessment": {},
                "required_tools_and_parts": {},
                "step_by_step_repair_guide": [],
                "alternative_approaches": [],
                "diagnostic_verification": [],
                "safety_protocols": [],
                "troubleshooting_guide": [],
                "quality_verification": [],
                "learning_insights": [],
                "upgrade_opportunities": [],
                "maintenance_prevention": {},
                "followup_questions": followup_questions,
                "response_type": "simple",
                "success": True
            }
            
        except Exception as e:
            await self.logger.error(f"Failed to generate simple response: {e}")
            raise e

    async def generate_user_message(self, user_message: str, diagnosis_result: Any, websocket=None, session_id=None) -> dict:
        # Stage 1: Starting user message generation
        if websocket:
            await self.send_ws_stage(websocket, "Starting user message generation", MessageSource.USER_INTERACTION, session_id=session_id)
        await self.logger.info(f"UserInteractionAgent - Generating user-facing message for: {user_message}")
        
        # Stage 2: Gathering session context
        if websocket:
            await self.send_ws_stage(websocket, "Gathering session context", MessageSource.USER_INTERACTION, session_id=session_id)
        
        # Get session context if available - try multiple sources
        session_context = ""
        if session_id:
            try:
                from app.agents.session_context_agent import SessionContextAgent
                context_manager = SessionContextAgent()
                session_context = context_manager.get_context_reminder(session_id)
            except Exception as e:
                await self.logger.warning(f"Could not retrieve session context for {session_id}: {e}")
        
        # If no session context from session_id, check if it's passed in diagnosis_result
        if not session_context and isinstance(diagnosis_result, dict):
            session_context = diagnosis_result.get('session_context', "")
        
        # Ensure we have some session guidance even if context loading fails
        if not session_context and session_id:
            session_context = "\nIMPORTANT: Maintain focus on the user's original automotive issue. Interpret follow-up questions and symptoms within the context of their primary concern."
        
        # Stage 3: Determining response type
        if websocket:
            response_type = "comprehensive" if not (isinstance(diagnosis_result, dict) and diagnosis_result.get("simple_response")) else "simple"
            await self.send_ws_stage(
                websocket, 
                f"Determining response type: {response_type}", 
                MessageSource.USER_INTERACTION, 
                session_id=session_id,
                details={"response_type": response_type}
            )
        
        # Check if this is a simple response request from orchestrator
        if isinstance(diagnosis_result, dict) and diagnosis_result.get("simple_response"):
            await self.logger.info("Processing simple response request from orchestrator")
            if websocket:
                await self.send_ws_stage(websocket, "Generating simple response", MessageSource.USER_INTERACTION, session_id=session_id)
            # Get conversation history from diagnosis_result if available
            conversation_history = diagnosis_result.get('conversation_history', [])
            return await self.generate_simple_response(user_message, session_context, conversation_history)
        
        # Otherwise, proceed with full diagnostic response
        await self.logger.info("Generating full diagnostic response")
        if websocket:
            await self.send_ws_stage(websocket, "Generating comprehensive diagnostic response", MessageSource.USER_INTERACTION, session_id=session_id)
        
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
                "diagnosis_result": merged_diagnosis,
                "session_context": session_context
            }
            prompt = self.prompt.format(**prompt_vars)
            
            # Stage 4: Processing with AI assistant
            if websocket:
                await self.send_ws_stage(websocket, "Processing with AI assistant to generate user response", MessageSource.USER_INTERACTION, session_id=session_id)
            
            # Use the LLMService for direct prompt calls
            response = await self.llm_service.generate_response(prompt)
            
            # Stage 5: Parsing AI response
            if websocket:
                await self.send_ws_stage(websocket, "Parsing AI response and formatting output", MessageSource.USER_INTERACTION, session_id=session_id)
            
            # Handle AIMessage objects from LangChain and dict responses
            if hasattr(response, 'content'):
                response = response.content
            elif isinstance(response, dict) and 'content' in response:
                response = response['content']
            elif isinstance(response, dict):
                response = str(response)
                
            response = self._sanitize_output(response)
            
            # Try to parse the response as JSON, robustly
            raw_response = response.strip() if isinstance(response, str) else str(response).strip()
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
            
            # Stage 4: Finalizing user message
            if websocket:
                await self.send_ws_stage(websocket, "Finalizing user message and formatting response", MessageSource.USER_INTERACTION, session_id=session_id)
            
            # Serialize datetimes before sending to websocket or returning
            parsed_response = serialize_datetimes(parsed_response)
            
            # Stage 5: User message generation complete
            if websocket:
                # Include tree data if available in diagnosis result
                tree_data = None
                if isinstance(diagnosis_result, dict) and 'diagnosis_tree' in diagnosis_result:
                    tree = diagnosis_result['diagnosis_tree']
                    if tree and hasattr(tree, 'children'):
                        tree_data = {
                            "total_nodes": len(tree.children),
                            "root_issue": tree.issue_name if hasattr(tree, 'issue_name') else "Unknown",
                            "children": [
                                {
                                    "issue_name": child.issue_name,
                                    "likelihood": round(child.likelyhood * 100, 1),
                                    "type": child.data.get("issue_type") if child.data else "Unknown",
                                    "category": child.data.get("issue_category") if child.data else "Unknown",
                                    "description": child.data.get("description") if child.data else None,
                                    "severity": child.data.get("severity") if child.data else "Unknown"
                                }
                                for child in tree.children
                            ]
                        }
                
                result_summary = {
                    "has_diagnosis": bool(parsed_response.get("technical_diagnosis")),
                    "repair_procedures_count": len(parsed_response.get("step_by_step_repair_guide", [])),
                    "safety_protocols_count": len(parsed_response.get("safety_protocols", [])),
                    "followup_questions_count": len(parsed_response.get("followup_questions", [])),
                    "difficulty_level": parsed_response.get("diy_difficulty_assessment", {}).get("level", "Unknown"),
                    "final_tree_data": tree_data  # Include complete tree data in final user message
                }
                await self.send_ws_result(
                    websocket, 
                    "User message generated successfully", 
                    MessageSource.USER_INTERACTION, 
                    session_id=session_id, 
                    details=result_summary
                )
            
            return {
                **parsed_response,
                "success": True,
                "step_by_step_guide": step_by_step_guide
            }
        except Exception as e:
            if websocket:
                await self.send_ws_error(websocket, f"Error in user message generation - {type(e).__name__}", MessageSource.USER_INTERACTION, session_id=session_id, details={"error": str(e)})
            error_stage = f"Error in user message generation - {type(e).__name__}"
            stage_msg = {"type": "stage", "stage": error_stage}
            await self.logger.info(f"Broadcasting error stage: {stage_msg['stage']}")
            await self.broadcast_stage(json.dumps(stage_msg))
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
        await self._log_entry("process", message_length=len(user_message), session_id=session_id)
        
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
                
                process_result = {
                    "success": result.get("success", False),
                    "user_message": user_message_text,
                    "followup_questions": result.get("followup_questions", []),  # Also include at top level for backwards compatibility
                    "error": None,
                    "error_type": None
                }
                
                await self._log_exit("process", success=process_result["success"], attempt=attempt + 1)
                return process_result
                
            except Exception as e:
                await self.logger.error(f"Process error (attempt {attempt+1}): {e}")
                if attempt == max_retries:
                    if websocket:
                        await self.send_ws_error(websocket, f"Error in user message generation - {type(e).__name__}", MessageSource.CHAT_SERVICE, session_id=session_id, details={"error": str(e)})
                    
                    error_result = {
                        "success": False,
                        "user_message": "",
                        "followup_questions": [
                            {"question": "Could you please describe the problem you're experiencing with your car?", "purpose": "Get basic problem description", "category": "symptom_clarification"},
                            {"question": "When did this issue first occur?", "purpose": "Understand problem timeline", "category": "symptom_clarification"}
                        ],
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    
                    await self._log_exit("process", success=False, error=str(e), max_attempts_reached=True)
                    return error_result

    def close(self) -> None:
        """
        Optional cleanup method for the agent.
        """
        pass
