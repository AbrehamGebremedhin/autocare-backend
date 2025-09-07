import json
from typing import Optional, Dict, Any
from datetime import datetime
from langchain.prompts import PromptTemplate
from app.agents.symptom_extraction_agent import SymptomExtractorAgent
from app.agents.diagnostic_agent import DiagnosisAgent
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.utils.diagnosis_tree_factory import get_diagnosis_tree
from app.agents.user_interaction_agent import UserInteractionAgent
from app.core.interfaces import IWebSocketManager
from app.agents.base_agent import BaseAgent, AgentCommand, AgentState
from app.utils.message_types import MessageSource
from app.utils.monitoring import monitor_and_handle
from app.services.llm_service import LLMService
from app.utils.orchestrator_websocket import OrchestratorWebSocketMixin
from app.utils.tree_formatter import TreeDataFormatter

class SymptomExtractionCommand(AgentCommand):
    """Command for symptom extraction operations"""
    
    def __init__(self, agent_class, user_interaction_agent):
        self.agent_class = agent_class
        self.user_interaction_agent = user_interaction_agent
    
    def validate(self, context: Dict[str, Any]) -> bool:
        return 'user_request' in context and 'car_id' in context
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_request = context['user_request']
        car_id = context['car_id']
        diagnosis_tree = context.get('diagnosis_tree')
        websocket = context.get('websocket')
        session_id = context.get('session_id')
        
        if car_id is None:
            user_response = await self.user_interaction_agent.process(
                user_request, 'car_id is required for symptom extraction.', session_id=session_id
            )
            return {
                'response': user_response.get('user_message'), 
                'success': user_response.get('success', True)
            }
        
        agent = self.agent_class(car_id, diagnosis_tree=diagnosis_tree)
        
        process_result = await agent.process(user_request, websocket=websocket, session_id=session_id)
        
        result = process_result.get('result')
        success = process_result.get('success', True)
        updated_tree = process_result.get('diagnosis_tree')
        
        if isinstance(result, dict) and result.get('need_more_info'):
            info_type = result.get('info_type', 'additional information')
            user_response = await self.user_interaction_agent.process(
                user_request, f"Could you please provide more details about: {info_type}?", session_id=session_id
            )
            return {
                'response': user_response.get('user_message'),
                'need_more_info': True,
                'info_type': info_type,
                'success': user_response.get('success', True),
                'diagnosis_tree': updated_tree
            }
        
        if not success:
            user_response = await self.user_interaction_agent.process(
                user_request, 'An error occurred during processing.', session_id=session_id
            )
            return {
                'response': user_response.get('user_message'), 
                'success': user_response.get('success', True), 
                'diagnosis_tree': updated_tree
            }
        
        return {
            'result': result,
            'success': success,
            'diagnosis_tree': updated_tree
        }

class DiagnosisCommand(AgentCommand):
    """Command for diagnosis operations"""
    
    def __init__(self, diagnosis_agent_class, user_interaction_agent):
        self.diagnosis_agent_class = diagnosis_agent_class
        self.user_interaction_agent = user_interaction_agent
    
    def validate(self, context: Dict[str, Any]) -> bool:
        return 'user_request' in context and 'car_id' in context
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_request = context['user_request']
        car_id = context['car_id']
        diagnosis_tree = context.get('diagnosis_tree')
        websocket = context.get('websocket')
        session_id = context.get('session_id')
        
        diagnostic_agent = self.diagnosis_agent_class(car_id, diagnosis_tree=diagnosis_tree)
        
        diagnosis_result = await diagnostic_agent.process(user_request, websocket=websocket, session_id=session_id)
        
        # Add tree data to diagnosis result for user interaction agent
        if diagnosis_tree and 'diagnosis_tree' not in diagnosis_result:
            diagnosis_result['diagnosis_tree'] = diagnosis_tree
        
        user_response = await self.user_interaction_agent.process(
            user_request, 
            diagnosis_result, 
            websocket=websocket,  # Pass websocket to user interaction agent
            session_id=session_id
        )
        
        return {
            'response': user_response.get('user_message'),
            'success': user_response.get('success', True),
            'step_by_step_guide': diagnosis_result.get('step_by_step_guide'),
            'diagnosis_tree': diagnosis_tree,
            'diagnosis_result': diagnosis_result
        }

class InitialProcessingCommand(AgentCommand):
    """Command for initial message processing (symptom extraction + diagnosis)"""
    
    def __init__(self, symptom_command, diagnosis_command):
        self.symptom_command = symptom_command
        self.diagnosis_command = diagnosis_command
    
    def validate(self, context: Dict[str, Any]) -> bool:
        # For initial processing, we can use a default car_id if not provided
        # This fixes the "Command 'initial_processing' validation failed" error
        if 'user_request' in context and 'car_id' not in context:
            # Add a default car_id for initial processing - will be replaced with real one later
            context['car_id'] = 'default_car'
            return True
        return self.symptom_command.validate(context) and self.diagnosis_command.validate(context)
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Execute symptom extraction
        symptom_result = await self.symptom_command.execute(context)
        
        if 'diagnosis_tree' in symptom_result:
            context['diagnosis_tree'] = symptom_result['diagnosis_tree']
        
        # Execute diagnosis with updated tree
        diagnosis_result = await self.diagnosis_command.execute(context)
        
        return {
            'response': diagnosis_result.get('response'),
            'success': diagnosis_result.get('success', True),
            'step_by_step_guide': diagnosis_result.get('step_by_step_guide'),
            'diagnosis_tree': context.get('diagnosis_tree')
        }

class OrchestratorAgent(BaseAgent, OrchestratorWebSocketMixin):
    """
    Orchestrates the flow between agents for end-to-end diagnosis and user interaction.
    Uses command pattern for better separation of concerns and maintainability.
    """
    def __init__(
        self,
        websocket_manager: Optional[IWebSocketManager] = None,
        symptom_extractor_agent_class=SymptomExtractorAgent,
        diagnosis_agent_class=DiagnosisAgent,
        user_interaction_agent: Optional[UserInteractionAgent] = None,
        **kwargs
    ):
        """
        Initialize the OrchestratorAgent with agent class dependencies for testability.
        """
        super().__init__(websocket_manager=websocket_manager, **kwargs)
        
        # Agent dependencies
        self.symptom_extractor_agent_class = symptom_extractor_agent_class
        self.diagnosis_agent_class = diagnosis_agent_class
        self.user_interaction_agent = user_interaction_agent or UserInteractionAgent()
        
        # LLM service for message classification
        self.llm_service = LLMService()
        
        # Message classification prompt
        self.classification_prompt = PromptTemplate.from_template(
            """
            You are analyzing a user message to determine what type of response is needed.
            
            USER MESSAGE: "{user_message}"
            SESSION CONTEXT: "{session_context}"
            
            Classify the message type to determine appropriate response level.
            
            SIMPLE RESPONSE (no diagnostic details needed):
            - General automotive questions that can be answered from knowledge
            - Educational questions (signs, symptoms, explanations) even if related to current issue
            - Clarification questions about terms or concepts  
            - Questions about maintenance schedules or general procedures
            - Follow-up questions about previously provided information
            - Basic "how-to" questions without specific problem context
            - Questions about tools, parts, or general automotive knowledge
            - Simple yes/no or factual questions
            - Questions asking for explanation of concepts or terminology
            - "What are the signs of..." type questions (educational, not diagnostic)
            
            FULL DIAGNOSTIC RESPONSE (comprehensive repair guidance needed):
            - Specific problem symptoms requiring diagnosis ("My car is making...")
            - NEW symptoms reported for existing issues ("Now it's also...")
            - Requests for repair procedures for identified issues
            - Troubleshooting requests for ongoing problems
            - Questions that require detailed repair guidance
            - Symptom-based inquiries requiring diagnostic analysis
            - Complex repair planning or procedure questions
            - Updates to existing symptoms ("Getting worse", "Now also...")
            
            IMPORTANT: Educational questions like "What are the signs of..." should be SIMPLE even 
            if they relate to an ongoing diagnostic session. Only classify as FULL DIAGNOSTIC if 
            the user is reporting actual symptoms they're experiencing.
            
            Respond with ONLY one word: "simple" or "full_diagnostic"
            """
        )
        
        # Initialize commands
        self._setup_commands()
    
    def _setup_commands(self) -> None:
        """Setup command pattern for orchestrator operations"""
        symptom_command = SymptomExtractionCommand(
            self.symptom_extractor_agent_class,
            self.user_interaction_agent
        )
        
        diagnosis_command = DiagnosisCommand(
            self.diagnosis_agent_class,
            self.user_interaction_agent
        )
        
        initial_processing_command = InitialProcessingCommand(
            symptom_command,
            diagnosis_command
        )
        
        # Register commands
        self.register_command('symptom_extraction', symptom_command)
        self.register_command('diagnosis', diagnosis_command)
        self.register_command('initial_processing', initial_processing_command)
    
    async def _ensure_diagnosis_tree(self, context: Dict[str, Any]) -> DiagnosisTreeNode:
        """Ensure diagnosis tree exists, creating if necessary"""
        await self._log_entry("_ensure_diagnosis_tree")
        
        diagnosis_tree = context.get('diagnosis_tree')
        car_id = context.get('car_id')
        websocket = context.get('websocket')
        session_id = context.get('session_id')
        
        # Case 1: No tree exists yet - create a new one
        if diagnosis_tree is None and car_id is not None:
            diagnosis_tree = get_diagnosis_tree(issue_name='root', likelyhood=1.0)
            context['diagnosis_tree'] = diagnosis_tree
            
            await self.logger.info(f"Created new diagnosis tree for car_id: {car_id}")
            
            # Send tree initialization using standardized method
            if websocket:
                await self.send_tree_initialization(websocket, diagnosis_tree, session_id)
        # Case 2: Tree exists but might be in dict form instead of object form
        elif diagnosis_tree is not None:
            # Ensure it's a proper DiagnosisTreeNode object
            if not isinstance(diagnosis_tree, DiagnosisTreeNode):
                await self.logger.warning(f"Diagnosis tree is not a DiagnosisTreeNode but a {type(diagnosis_tree)}")
                try:
                    # Try to deserialize if it's a dict
                    if isinstance(diagnosis_tree, dict):
                        from app.schemas.Chat_Session import ChatSession
                        diagnosis_tree = ChatSession.deserialize_diagnosis_tree(diagnosis_tree)
                        context['diagnosis_tree'] = diagnosis_tree
                        
                        # Verify the deserialized object
                        if hasattr(diagnosis_tree, 'children'):
                            await self.logger.info(f"Successfully deserialized tree with {len(diagnosis_tree.children)} children")
                        else:
                            raise ValueError("Deserialized object missing 'children' attribute")
                except Exception as e:
                    await self.logger.error(f"Failed to deserialize tree: {str(e)}")
                    # Create a new tree if deserialization failed
                    diagnosis_tree = get_diagnosis_tree(issue_name='root', likelyhood=1.0)
                    context['diagnosis_tree'] = diagnosis_tree
                    await self.logger.info("Created replacement diagnosis tree after deserialization failure")
                    
                    # Send tree initialization for replacement tree
                    if websocket:
                        await self.send_tree_initialization(websocket, diagnosis_tree, session_id)
            else:
                # It's a valid DiagnosisTreeNode
                if hasattr(diagnosis_tree, 'children'):
                    await self.logger.info(f"Using existing diagnosis tree with {len(diagnosis_tree.children)} children")
                    
                    # Verify children are accessible
                    child_count = len(diagnosis_tree.children)
                    if child_count > 0:
                        child_names = [child.issue_name for child in diagnosis_tree.children]
                        await self.logger.info(f"Tree has {child_count} children: {child_names}")
                    else:
                        await self.logger.info("Tree has no children")
                else:
                    await self.logger.error(f"Diagnosis tree object missing 'children' attribute: {type(diagnosis_tree)}")
                    # Replace with new tree
                    diagnosis_tree = get_diagnosis_tree(issue_name='root', likelyhood=1.0)
                    context['diagnosis_tree'] = diagnosis_tree
                    await self.logger.info("Created replacement diagnosis tree due to missing children attribute")
        
        # Final safety check
        if not hasattr(diagnosis_tree, 'children'):
            await self.logger.error(f"Final diagnosis tree still invalid: {type(diagnosis_tree)}")
            diagnosis_tree = get_diagnosis_tree(issue_name='root', likelyhood=1.0)
            context['diagnosis_tree'] = diagnosis_tree
            await self.logger.info("Created final fallback diagnosis tree")
        
        await self._log_exit("_ensure_diagnosis_tree", children_count=len(diagnosis_tree.children) if diagnosis_tree and hasattr(diagnosis_tree, 'children') else 0)
        return diagnosis_tree
    
    async def route_request(self, user_request: str, user_id: str = None, context: dict = None, websocket=None, session_id=None):
        """
        PERFORMANCE OPTIMIZED: Main entry point with improved parallel processing and reduced overhead.
        """
        await self._log_entry("route_request", user_request=user_request[:100], user_id=user_id)
        
        # Log WebSocket connection status
        if websocket:
            await self.logger.info(f"WebSocket connection active for session {session_id}")
        else:
            await self.logger.info("No WebSocket connection provided")
            
        await self.send_orchestrator_stage(websocket, "Routing request", session_id=session_id)
        
        # Prepare context for command execution
        command_context = {
            'user_request': user_request,
            'user_id': user_id,
            'websocket': websocket,
            'session_id': session_id,
            **(context or {})
        }
        
        # Ensure diagnosis tree exists
        original_tree = command_context.get('diagnosis_tree')
        if original_tree:
            # Check if tree is valid before accessing children
            if hasattr(original_tree, 'children'):
                await self.logger.info(f"Received existing tree with {len(original_tree.children)} children")
                # Log some info about the existing tree's structure
                if original_tree.children:
                    child_issues = [c.issue_name for c in original_tree.children]
                    await self.logger.info(f"Existing tree child issues: {child_issues}")
            else:
                await self.logger.info(f"Received tree object but it's not a DiagnosisTreeNode (type: {type(original_tree)})")
        else:
            await self.logger.info("No existing tree found in command context")
        
        await self._ensure_diagnosis_tree(command_context)
        
        # After ensuring the tree exists, log its state
        current_tree = command_context.get('diagnosis_tree')
        await self.logger.info(f"Tree after ensure_diagnosis_tree has {len(current_tree.children) if current_tree else 0} children")
        
        # Get session context for classification
        session_context = ""
        if session_id:
            try:
                from app.agents.session_context_agent import SessionContextAgent
                context_manager = SessionContextAgent()
                session_context = context_manager.get_context_reminder(session_id)
            except Exception as e:
                await self.logger.warning(f"Could not retrieve session context for {session_id}: {e}")
        
        # OPTIMIZATION: Classify message early to determine processing path
        await self.send_orchestrator_stage(websocket, "Classifying message type", session_id=session_id)
        classification = await self.classify_message(user_request, session_context)
        
        # Send classification result via standardized WebSocket
        if websocket:
            await self.send_classification_result(websocket, classification, session_id)
        
        # If it's a simple question and we have high confidence, skip the full diagnostic pipeline
        if (classification.get("response_type") == "simple" and 
            classification.get("confidence", 0.0) > 0.7):
            
            await self.logger.info("[ORCHESTRATOR] Taking simple response path - skipping full diagnostic pipeline")
            await self.send_orchestrator_stage(websocket, "Generating simple response", session_id=session_id)
            
            # For simple responses, still extract session context if it's an initial message
            if command_context.get('is_initial_message') and session_id:
                try:
                    from app.agents.session_context_agent import SessionContextAgent
                    context_agent = SessionContextAgent()
                    await context_agent.extract_original_issue(
                        user_request, 
                        session_id, 
                        command_context.get('car_make', ''),
                        command_context.get('car_model', ''),
                        command_context.get('car_year', '')
                    )
                    # Get updated session context
                    session_context = context_agent.get_context_reminder(session_id)
                except Exception as e:
                    await self.logger.warning(f"Failed to extract session context for simple response: {e}")
            
            # Generate simple response directly through user interaction agent
            # Pass conversation history for context
            conversation_history = command_context.get('conversation_history', [])
            await self.logger.info(f"[ORCHESTRATOR] Simple response - conversation history length: {len(conversation_history)}")
            if conversation_history:
                await self.logger.info(f"[ORCHESTRATOR] Last message in history: {conversation_history[-1].get('content', 'No content')[:100]}")
            
            simple_response = await self.user_interaction_agent.generate_simple_response(
                user_request, 
                session_context, 
                conversation_history
            )
            
            # Format response to match expected structure
            user_response = await self.user_interaction_agent.process(
                user_request, 
                {
                    "simple_response": True,
                    "conversation_history": conversation_history
                }, 
                websocket=websocket, 
                session_id=session_id
            )
            
            # Override the user_message with our simple response
            if user_response.get('success'):
                user_response['user_message'] = simple_response
                user_response['response_type'] = 'simple'
                user_response['tokens_saved'] = True  # Indicator that we saved processing
            
            await self.send_orchestrator_stage(websocket, "Simple response complete", session_id=session_id)
            await self._log_exit("route_request", success=True, result_type="simple_response")
            return {'response': user_response.get('user_message'), 'success': user_response.get('success', True)}
        
        # Continue with full diagnostic pipeline for complex questions
        await self.logger.info("[ORCHESTRATOR] Taking full diagnostic path")
        
        # Stage: Full diagnostic pipeline starting
        if websocket:
            details = {
                "analysis_type": "comprehensive",
                "classification_confidence": classification.get("confidence", 0.0),
                "expected_stages": ["symptom_extraction", "tree_building", "diagnosis", "user_message_generation"]
            }
            await self.send_orchestrator_stage(
                websocket, 
                "Starting comprehensive diagnostic analysis", 
                session_id=session_id,
                details=details
            )
        
        car_id = command_context.get('car_id')
        
        # Route based on request type and context
        try:
            if command_context.get('is_initial_message'):
                await self.send_ws_stage(websocket, "Processing initial message with full analysis", MessageSource.ORCHESTRATOR, session_id=session_id)
                result = await self.execute_command('initial_processing', command_context)
                await self._log_exit("route_request", success=True, result_type="initial_processing")
                return result
            
            elif car_id is not None:
                # Check if we have an existing diagnosis tree with children
                diagnosis_tree = command_context.get('diagnosis_tree')
                is_tree_empty = diagnosis_tree is None or len(diagnosis_tree.children) == 0
                
                # Always run symptom extraction first for non-initial messages if tree is empty
                # or if the message suggests new symptoms
                user_request_lower = user_request.lower()
                symptom_keywords = ['issue', 'problem', 'symptom', 'wrong', 'noise', 'doesn\'t work', 'not working', 'failed']
                needs_symptom_extraction = is_tree_empty or any(keyword in user_request_lower for keyword in symptom_keywords)
                
                # PERFORMANCE OPTIMIZATION: Start context preparation for diagnosis in parallel
                import asyncio
                
                if needs_symptom_extraction:
                    if websocket:
                        tree_context = {
                            "tree_empty": is_tree_empty,
                            "existing_symptoms": len(diagnosis_tree.children) if diagnosis_tree else 0,
                            "analysis_trigger": "new_symptoms_detected" if not is_tree_empty else "empty_tree"
                        }
                        await self.send_orchestrator_stage(
                            websocket, 
                            "Extracting new symptoms from follow-up message", 
                            session_id=session_id,
                            details=tree_context
                        )
                    
                    # Start both symptom extraction and diagnosis preparation in parallel
                    symptom_task = asyncio.create_task(self.execute_command('symptom_extraction', command_context))
                    
                    # Wait for symptom extraction to complete
                    symptom_result = await symptom_task
                    
                    # Update context with new tree
                    if 'diagnosis_tree' in symptom_result:
                        command_context['diagnosis_tree'] = symptom_result['diagnosis_tree']
                        
                        # Send initial tree data immediately after symptom extraction
                        if websocket:
                            try:
                                tree = symptom_result['diagnosis_tree']
                                new_symptoms_count = len(symptom_result.get('result', [])) if 'result' in symptom_result else 0
                                await self.send_symptom_extraction_complete(websocket, tree, new_symptoms_count, session_id)
                            except Exception as e:
                                await self.logger.error(f"Failed to send symptom extraction tree data: {e}")
                        
                        # Tree update notification is now handled by send_symptom_extraction_complete
                else:
                    await self.logger.info("Skipping symptom extraction - tree has content and no new symptoms detected")
                
                # Now run the diagnosis with the updated tree
                if websocket:
                    current_tree = command_context.get('diagnosis_tree')
                    
                    # Send pre-diagnosis tree data using standardized method
                    if current_tree:
                        try:
                            await self.send_diagnosis_preparation(websocket, current_tree, session_id)
                        except Exception as e:
                            await self.logger.error(f"Failed to send pre-diagnosis tree data: {e}")
                    
                    diagnosis_context = {
                        "tree_available": current_tree is not None,
                        "symptoms_count": len(current_tree.children) if current_tree else 0,
                        "analysis_stage": "comprehensive_diagnosis"
                    }
                    await self.send_orchestrator_stage(
                        websocket, 
                        "Starting comprehensive diagnosis analysis", 
                        session_id=session_id,
                        details=diagnosis_context
                    )
                
                diagnosis_result = await self.execute_command('diagnosis', command_context)

                if 'diagnosis_tree' not in diagnosis_result and 'diagnosis_tree' in command_context:
                    diagnosis_result['diagnosis_tree'] = command_context['diagnosis_tree']
                    await self.logger.info("Explicitly adding diagnosis tree to result")
                
                # Final completion stage
                if websocket:
                    # Send final diagnosis completion with standardized tree data
                    tree = None
                    if 'diagnosis_tree' in diagnosis_result and diagnosis_result['diagnosis_tree']:
                        tree = diagnosis_result['diagnosis_tree']
                    elif 'diagnosis_tree' in command_context and command_context['diagnosis_tree']:
                        tree = command_context['diagnosis_tree']
                    
                    if tree:
                        await self.send_diagnosis_complete(websocket, tree, diagnosis_result, session_id)
                    
                    completion_summary = {
                        "diagnosis_success": diagnosis_result.get('success', False),
                        "has_response": 'response' in diagnosis_result,
                        "has_step_guide": 'step_by_step_guide' in diagnosis_result,
                        "tree_preserved": 'diagnosis_tree' in diagnosis_result,
                        "diagnosis_complete": True
                    }
                    await self.send_ws_result(
                        websocket, 
                        "Comprehensive diagnosis completed", 
                        MessageSource.ORCHESTRATOR, 
                        session_id=session_id,
                        details=completion_summary
                    )
                
                await self._log_exit("route_request", success=True, result_type="diagnosis")
                return diagnosis_result
            
            else:
                await self.send_orchestrator_stage(websocket, "No car_id provided, cannot diagnose", session_id=session_id)
                user_response = await self.user_interaction_agent.process(user_request, 'car_id is required for diagnosis.', session_id=session_id)
                await self._log_exit("route_request", success=False, reason="no_car_id")
                return {'response': user_response.get('user_message'), 'success': user_response.get('success', True)}
                
        except Exception as e:
            await self.logger.error(f"Error in route_request: {str(e)}")
            # Use standardized error method with tree state context
            current_tree = command_context.get('diagnosis_tree') if 'command_context' in locals() else None
            await self.send_error_with_tree_state(websocket, f"Error in orchestrator: {str(e)}", current_tree, session_id)
            await self._set_state(AgentState.ERROR)
            user_response = await self.user_interaction_agent.process(user_request, f'Error: {str(e)}', session_id=session_id)
            await self._log_exit("route_request", success=False, error=str(e))
            return {'response': user_response.get('user_message'), 'success': False}

    async def classify_message(self, user_message: str, session_context: str = "") -> dict:
        """
        Classify the user message to determine if it needs a simple response or full diagnostic response.
        This runs at the orchestrator level to optimize routing and reduce processing overhead.
        
        Returns:
            dict: Contains response_type ("simple" or "full_diagnostic"), confidence, and reasoning
        """
        try:
            formatted_prompt = self.classification_prompt.format(
                user_message=user_message,
                session_context=session_context or "No session context available"
            )
            
            response = await self.llm_service.generate_response(formatted_prompt)
            
            # Handle AIMessage objects from LangChain
            if hasattr(response, 'content'):
                response_text = response.content.strip().lower()
            else:
                response_text = str(response).strip().lower()
            
            # Simple keyword-based classification since LLM returns plain text
            if "simple" in response_text:
                classification = "simple"
                confidence = 0.8
                reasoning = "LLM classified as simple response"
            elif "full_diagnostic" in response_text or "diagnostic" in response_text:
                classification = "full_diagnostic"
                confidence = 0.8
                reasoning = "LLM classified as full diagnostic response"
            else:
                # Fallback classification based on keywords in the message
                simple_keywords = [
                    'what is', 'how often', 'difference between', 'explain', 'define',
                    'meaning of', 'purpose of', 'used for', 'maintenance schedule',
                    'oil change', 'tire pressure', 'general', 'basic', 'what are the signs',
                    'signs of', 'symptoms of', 'how to identify', 'what tools', 'what parts',
                    'how do i', 'what should i', 'educational', 'learn about'
                ]
                
                diagnostic_keywords = [
                    'my car', 'my engine', 'my brake', 'is making', 'started making',
                    'noise', 'problem', 'issue', 'broken', 'not working', 'failed',
                    'grinding', 'squeaking', 'vibration', 'leak', 'overheating',
                    'smell', 'smoke', 'warning light', 'getting worse', 'now also',
                    'also happening', 'started happening'
                ]
                
                user_lower = user_message.lower()
                
                # Check for educational question patterns first (these should be simple)
                educational_patterns = ['what are the signs', 'signs of', 'how to identify', 'what to look for']
                is_educational = any(pattern in user_lower for pattern in educational_patterns)
                
                simple_score = sum(1 for keyword in simple_keywords if keyword in user_lower)
                diagnostic_score = sum(1 for keyword in diagnostic_keywords if keyword in user_lower)
                
                # Educational questions are always simple, even if they contain diagnostic keywords
                if is_educational:
                    classification = "simple"
                    confidence = 0.7
                    reasoning = "Educational question pattern detected - simple response"
                elif simple_score > diagnostic_score:
                    classification = "simple"
                    confidence = 0.6
                    reasoning = "Keyword-based classification: simple question patterns detected"
                else:
                    classification = "full_diagnostic"
                    confidence = 0.6
                    reasoning = "Keyword-based classification: diagnostic patterns detected or default"
            
            result = {
                "response_type": classification,
                "confidence": confidence,
                "reasoning": reasoning
            }
            
            await self.logger.info(f"[ORCHESTRATOR] Message classification: {classification} (confidence: {confidence:.2f}) - {reasoning}")
            return result
            
        except Exception as e:
            await self.logger.error(f"[ORCHESTRATOR] Failed to classify message: {e}")
            # Default to full diagnostic for safety
            return {
                "response_type": "full_diagnostic",
                "confidence": 0.1,
                "reasoning": "Classification failed, defaulting to full diagnostic"
            }

    def _is_chat_request(self, user_request: str) -> bool:
        """Simple heuristic to determine if request is a chat vs diagnostic request"""
        keywords = ['symptom', 'diagnosis', 'extract']
        return not any(k in user_request.lower() for k in keywords)

    def request_more_info(self, info_type: str, from_agent: str = None):
        """Request additional information from agents or user"""
        # This could be extended to implement inter-agent communication
        return f'Additional info needed for {info_type}'

    @monitor_and_handle("OrchestratorAgent")
    async def process(self, user_request: str, user_id: str = None, context: dict = None) -> dict:
        """
        Standard entry point for all orchestrator interactions.
        Calls route_request and returns a standardized result dict.
        """
        await self._log_entry("process", user_request=user_request[:100])
        
        try:
            if self.state == AgentState.INACTIVE:
                await self.initialize()
            
            result = await self.route_request(user_request, user_id=user_id, context=context)
            
            process_result = {
                'result': result.get('response'),
                'success': result.get('success', True),
                'step_by_step_guide': result.get('step_by_step_guide'),
                'diagnosis_tree': result.get('diagnosis_tree')
            }
            
            await self._log_exit("process", success=process_result['success'])
            return process_result
            
        except Exception as e:
            await self.logger.error(f"Error in orchestrator process: {str(e)}")
            await self._set_state(AgentState.ERROR)
            user_response = await self.user_interaction_agent.process(user_request, f'Error: {str(e)}')
            
            await self._log_exit("process", success=False, error=str(e))
            return {
                'result': user_response.get('user_message'),
                'success': False
            }
    
    async def close(self) -> None:
        """
        Cleanup method for the agent.
        """
        await self.shutdown()
        if hasattr(self.user_interaction_agent, 'close'):
            self.user_interaction_agent.close()
