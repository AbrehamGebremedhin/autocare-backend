"""
Session Context Agent for managing conversation focus and original issue tracking.
Focused specifically on session context management - delegates other tasks to specialized agents.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.agents.base_agent import BaseAgent
from app.services.llm_service import LLMService
from app.CRUD.car_crud import CarCRUD
from app.core.interfaces import IWebSocketManager
from app.utils.message_types import MessageSource
from app.utils.monitoring import monitor_and_handle


@dataclass
class OriginalIssueContext:
    """Simplified original issue context focused on session management."""
    primary_issue: str
    symptoms: List[str]
    issue_category: str  # e.g., "engine", "braking", "electrical", etc.
    initial_message: str
    timestamp: datetime
    keywords: List[str]  # Key diagnostic terms from the original issue
    severity: str  # "low", "medium", "high", "critical"
    urgency: str  # "not_urgent", "moderate", "urgent", "emergency"
    confidence_score: float  # LLM confidence in analysis (0.0-1.0)
    car_make: Optional[str] = None
    car_model: Optional[str] = None
    car_year: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OriginalIssueContext':
        # Handle datetime conversion
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class SessionContextAgent(BaseAgent):
    """
    Session Context Agent focused on managing conversation context and original issue tracking.
    
    RESPONSIBILITIES:
    - Extract and store original issue context from initial messages
    - Determine if follow-up messages are relevant to the original issue
    - Provide context summaries for other agents
    - Maintain session focus and detect off-topic conversations
    
    DELEGATES TO OTHER AGENTS:
    - Symptom extraction → SymptomExtractionAgent
    - Diagnosis analysis → DiagnosticAgent
    - User interaction → UserInteractionAgent
    """
    
    def __init__(
        self,
        car_id: Optional[str] = None,
        car_make: Optional[str] = None,
        car_model: Optional[str] = None,
        car_year: Optional[str] = None,
        websocket_manager: Optional[IWebSocketManager] = None,
        llm_service: Optional[LLMService] = None,
        **kwargs
    ):
        super().__init__(
            car_crud=CarCRUD() if car_id else None,
            car_id=car_id,
            car_make=car_make,
            car_model=car_model,
            car_year=car_year,
            logger_name="SessionContextAgent",
            websocket_manager=websocket_manager
        )
        self.llm_service = llm_service or LLMService()
        self.original_issue_contexts: Dict[str, OriginalIssueContext] = {}
        
        # Simplified prompts focused on session context only
        self.context_extraction_prompt = PromptTemplate.from_template(
            """
            You are analyzing a user's initial automotive problem description to extract basic session context.
            
            Your ONLY job is to identify the core issue for session tracking - NOT to diagnose or extract detailed symptoms.
            
            USER MESSAGE: "{user_message}"
            CAR DETAILS: {car_make} {car_model} {car_year}
            
            Extract basic context information in JSON format:
            
            {{
                "primary_issue": "Simple description of the main problem mentioned",
                "symptoms": ["Only the symptoms explicitly mentioned by the user"],
                "issue_category": "Primary automotive system affected (engine, braking, transmission, electrical, suspension, cooling, fuel, exhaust, drivetrain, hvac, body, general)",
                "keywords": ["Key automotive terms from the message"],
                "severity": "Basic severity assessment (low, medium, high, critical)",
                "urgency": "Basic urgency level (not_urgent, moderate, urgent, emergency)",
                "confidence_score": 0.0-1.0
            }}
            
            KEEP IT SIMPLE - focus on what the user explicitly stated, not detailed diagnosis.
            Return ONLY valid JSON.
            """
        )
        
        self.relevance_analysis_prompt = PromptTemplate.from_template(
            """
            You are determining if a follow-up message is relevant to an ongoing automotive diagnostic session.
            
            ORIGINAL ISSUE CONTEXT:
            - Primary Issue: {original_issue}
            - Category: {issue_category}
            - Initial Message: "{initial_message}"
            
            NEW MESSAGE: "{new_message}"
            
            Is this new message relevant to the original automotive issue?
            
            Consider it RELEVANT if:
            - Provides more info about the same problem
            - Asks questions about the same automotive system
            - Mentions related symptoms or behaviors
            - Follows up on suggested troubleshooting
            
            Consider it IRRELEVANT if:
            - Completely different automotive problem
            - Non-automotive topic
            - Unrelated to the original issue category
            
            Return JSON:
            {{
                "is_relevant": true/false,
                "relevance_score": 0.0-1.0,
                "reason": "Brief explanation of why it is/isn't relevant"
            }}
            
            Return ONLY valid JSON.
            """
        )

    async def extract_original_issue(self, user_message: str, session_id: str, car_make: str = "", car_model: str = "", car_year: str = "") -> OriginalIssueContext:
        """
        Extract original issue context from the user's initial message.
        This focuses only on basic context extraction for session management.
        """
        try:
            # Format the prompt with user message and car details
            formatted_prompt = self.context_extraction_prompt.format(
                user_message=user_message,
                car_make=car_make,
                car_model=car_model,
                car_year=car_year
            )

            # Get LLM response
            response = await self.llm_service.generate_response(formatted_prompt)

            # Handle different response types
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, dict):
                # Handle dict responses like {'content': '...', 'type': 'ai'}
                response_text = response.get('content', str(response))
            else:
                response_text = str(response)

            # Clean and parse JSON response
            clean_response = self._clean_json_response(response_text)
            try:
                # Try parsing as JSON first
                result = json.loads(clean_response)
            except json.JSONDecodeError as e:
                # Try to fix single quotes to double quotes
                fixed_response = clean_response.replace("'", '"')
                try:
                    result = json.loads(fixed_response)
                except Exception as e2:
                    await self.logger.error(f"Raw LLM response for context extraction (session {session_id}): {response_text}")
                    await self.logger.error(f"Failed to parse context JSON: {e2}")
                    result = {}

            # Create context object
            context = OriginalIssueContext(
                primary_issue=result.get('primary_issue', user_message[:100]),
                symptoms=result.get('symptoms', []),
                issue_category=result.get('issue_category', 'general'),
                initial_message=user_message,
                timestamp=datetime.now(),
                keywords=result.get('keywords', []),
                severity=result.get('severity', 'medium'),
                urgency=result.get('urgency', 'moderate'),
                confidence_score=result.get('confidence_score', 0.5),
                car_make=car_make,
                car_model=car_model,
                car_year=car_year
            )

            # Store the context
            self.original_issue_contexts[session_id] = context

            await self.logger.info(f"Session {session_id}: Extracted context - {context.issue_category} issue: {context.primary_issue}")
            return context

        except Exception as e:
            await self.logger.error(f"Failed to extract original issue context: {e}")
            # Fallback context
            fallback_context = OriginalIssueContext(
                primary_issue=user_message[:100],
                symptoms=[],
                issue_category='general',
                initial_message=user_message,
                timestamp=datetime.now(),
                keywords=[],
                severity='medium',
                urgency='moderate',
                confidence_score=0.1,
                car_make=car_make,
                car_model=car_model,
                car_year=car_year
            )
            self.original_issue_contexts[session_id] = fallback_context
            return fallback_context

    async def is_message_relevant(self, message: str, session_id: str, relevance_threshold: float = 0.3) -> bool:
        """
        Check if a follow-up message is relevant to the original issue.
        This is the core session context responsibility.
        """
        try:
            if session_id not in self.original_issue_contexts:
                await self.logger.warning(f"No original context found for session {session_id}")
                return True  # Default to relevant if no context
            
            context = self.original_issue_contexts[session_id]
            
            # Format the prompt
            formatted_prompt = self.relevance_analysis_prompt.format(
                original_issue=context.primary_issue,
                issue_category=context.issue_category,
                initial_message=context.initial_message,
                new_message=message
            )
            
            # Get LLM response
            response = await self.llm_service.generate_response(formatted_prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, dict):
                # Handle dict responses like {'content': '...', 'type': 'ai'}
                response_text = response.get('content', str(response))
            else:
                response_text = str(response)
            
            # Parse response
            clean_response = self._clean_json_response(response_text)
            result = json.loads(clean_response)
            
            is_relevant = result.get('is_relevant', True)
            relevance_score = result.get('relevance_score', 0.5)
            
            # Apply threshold
            final_relevance = is_relevant and relevance_score >= relevance_threshold
            
            await self.logger.info(f"Session {session_id}: Message relevance = {final_relevance} (score: {relevance_score})")
            return final_relevance
            
        except Exception as e:
            await self.logger.error(f"Failed to check message relevance: {e}")
            return True  # Default to relevant on error

    def get_original_context(self, session_id: str) -> Optional[OriginalIssueContext]:
        """Get the stored original issue context for a session."""
        return self.original_issue_contexts.get(session_id)

    def get_context_reminder(self, session_id: str) -> str:
        """
        Get a context reminder string for other agents to use.
        This provides session context to other agents without them needing to know the details.
        """
        if session_id not in self.original_issue_contexts:
            return ""
        
        context = self.original_issue_contexts[session_id]
        return f"Session is focused on a {context.issue_category} issue: '{context.primary_issue}' (severity: {context.severity}, urgency: {context.urgency})"

    def clear_session_context(self, session_id: str) -> None:
        """Clear stored context for a session."""
        if session_id in self.original_issue_contexts:
            del self.original_issue_contexts[session_id]

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

    @monitor_and_handle("SessionContextAgent")
    async def process(self, user_message: str, websocket=None, session_id=None, action="extract_context", **kwargs) -> Dict[str, Any]:
        """
        Main processing method for session context operations.
        """
        try:
            if action == "extract_context":
                car_make = kwargs.get('car_make', '')
                car_model = kwargs.get('car_model', '')
                car_year = kwargs.get('car_year', '')
                context = await self.extract_original_issue(user_message, session_id, car_make, car_model, car_year)
                return {
                    "success": True,
                    "context": context.to_dict(),
                    "session_id": session_id
                }
            elif action == "check_relevance":
                is_relevant = await self.is_message_relevant(user_message, session_id)
                return {
                    "success": True,
                    "is_relevant": is_relevant,
                    "session_id": session_id
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
        except Exception as e:
            await self.logger.error(f"Session context processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def perform_action(self, *args, **kwargs):
        """Base agent interface implementation."""
        return await self.run_with_notification(self._perform_action_impl, *args, **kwargs)

    async def _perform_action_impl(self, *args, **kwargs):
        """Actual action implementation."""
        pass


# Backward compatibility alias
SessionContextManager = SessionContextAgent
