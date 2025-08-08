import json
from typing import Optional, Dict, Any
from app.agents.symptom_extraction_agent import SymptomExtractorAgent
from app.agents.diagnostic_agent import DiagnosisAgent
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.utils.diagnosis_tree_factory import get_diagnosis_tree
from app.agents.user_interaction_agent import UserInteractionAgent
from app.core.interfaces import IWebSocketManager
from app.agents.base_agent import BaseAgent, AgentCommand, AgentState
from app.utils.message_types import MessageSource
from app.utils.monitoring import monitor_and_handle

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
                user_request, 'car_id is required for symptom extraction.'
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
                user_request, f"Could you please provide more details about: {info_type}?"
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
                user_request, 'An error occurred during processing.'
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
        
        user_response = await self.user_interaction_agent.process(user_request, diagnosis_result)
        
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

class OrchestratorAgent(BaseAgent):
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
            
            await self.send_ws_stage(
                websocket, 
                "Orchestrator - Created new diagnosis tree", 
                MessageSource.ORCHESTRATOR, 
                session_id=session_id
            )
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
                        await self.logger.info(f"Successfully deserialized tree with {len(diagnosis_tree.children)} children")
                except Exception as e:
                    await self.logger.error(f"Failed to deserialize tree: {str(e)}")
                    # Create a new tree if deserialization failed
                    diagnosis_tree = get_diagnosis_tree(issue_name='root', likelyhood=1.0)
                    context['diagnosis_tree'] = diagnosis_tree
                    await self.logger.info("Created replacement diagnosis tree after deserialization failure")
            else:
                # It's a valid DiagnosisTreeNode
                await self.logger.info(f"Using existing diagnosis tree with {len(diagnosis_tree.children)} children")
                
                # Verify children are accessible
                if hasattr(diagnosis_tree, 'children'):
                    child_count = len(diagnosis_tree.children)
                    if child_count > 0:
                        child_names = [child.issue_name for child in diagnosis_tree.children]
                        await self.logger.info(f"Tree has {child_count} children: {child_names}")
                    else:
                        await self.logger.info("Tree has no children")
        
        await self._log_exit("_ensure_diagnosis_tree", children_count=len(diagnosis_tree.children) if diagnosis_tree else 0)
        return diagnosis_tree
    
    async def route_request(self, user_request: str, user_id: str = None, context: dict = None, websocket=None, session_id=None):
        """
        Main entry point: decides which command should handle the user request.
        """
        await self._log_entry("route_request", user_request=user_request[:100], user_id=user_id)
        
        # Log WebSocket connection status
        if websocket:
            await self.logger.info(f"WebSocket connection active for session {session_id}")
        else:
            await self.logger.info("No WebSocket connection provided")
            
        await self.send_ws_stage(websocket, "Orchestrator - Routing request", MessageSource.ORCHESTRATOR, session_id=session_id)
        
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
            await self.logger.info(f"Received existing tree with {len(original_tree.children)} children")
            # Log some info about the existing tree's structure
            if hasattr(original_tree, 'children') and original_tree.children:
                child_issues = [c.issue_name for c in original_tree.children]
                await self.logger.info(f"Existing tree child issues: {child_issues}")
        
        await self._ensure_diagnosis_tree(command_context)
        
        # After ensuring the tree exists, log its state
        current_tree = command_context.get('diagnosis_tree')
        await self.logger.info(f"Tree after ensure_diagnosis_tree has {len(current_tree.children) if current_tree else 0} children")
        
        car_id = command_context.get('car_id')
        
        # Route based on request type and context
        try:
            if command_context.get('is_initial_message'):
                await self.send_ws_stage(websocket, "Orchestrator - Processing initial message", MessageSource.ORCHESTRATOR, session_id=session_id)
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
                
                if needs_symptom_extraction:
                    await self.send_ws_stage(websocket, "Orchestrator - Extracting symptoms from follow-up message", MessageSource.ORCHESTRATOR, session_id=session_id)
                    symptom_result = await self.execute_command('symptom_extraction', command_context)
                    
                    # Update context with new tree
                    if 'diagnosis_tree' in symptom_result:
                        command_context['diagnosis_tree'] = symptom_result['diagnosis_tree']
                
                # Now run the diagnosis with the updated tree
                await self.send_ws_stage(websocket, "Orchestrator - Processing diagnosis request", MessageSource.ORCHESTRATOR, session_id=session_id)
                diagnosis_result = await self.execute_command('diagnosis', command_context)
                
                # Check if additional symptom extraction is needed based on diagnosis result
                if isinstance(diagnosis_result.get('diagnosis_result'), dict) and \
                   diagnosis_result['diagnosis_result'].get('need_symptom_extraction'):
                    
                    await self.send_ws_stage(websocket, "Orchestrator - Need additional symptom extraction after diagnosis", MessageSource.ORCHESTRATOR, session_id=session_id)
                    
                    # Execute additional symptom extraction
                    symptom_result = await self.execute_command('symptom_extraction', command_context)
                    
                    # Update context with new tree and re-run diagnosis
                    if 'diagnosis_tree' in symptom_result:
                        command_context['diagnosis_tree'] = symptom_result['diagnosis_tree']
                    
                    await self.send_ws_stage(websocket, "Orchestrator - Re-running diagnosis with updated tree", MessageSource.ORCHESTRATOR, session_id=session_id)
                    diagnosis_result = await self.execute_command('diagnosis', command_context)
                
                # Ensure the diagnosis tree is explicitly returned
                if 'diagnosis_tree' not in diagnosis_result and 'diagnosis_tree' in command_context:
                    diagnosis_result['diagnosis_tree'] = command_context['diagnosis_tree']
                    await self.logger.info("Explicitly adding diagnosis tree to result")
                
                await self.send_ws_stage(websocket, "Orchestrator - Done", MessageSource.ORCHESTRATOR, session_id=session_id)
                await self._log_exit("route_request", success=True, result_type="diagnosis")
                return diagnosis_result
            
            else:
                await self.send_ws_stage(websocket, "Orchestrator - No car_id provided, cannot diagnose", MessageSource.ORCHESTRATOR, session_id=session_id)
                user_response = await self.user_interaction_agent.process(user_request, 'car_id is required for diagnosis.')
                await self._log_exit("route_request", success=False, reason="no_car_id")
                return {'response': user_response.get('user_message'), 'success': user_response.get('success', True)}
                
        except Exception as e:
            await self.logger.error(f"Error in route_request: {str(e)}")
            await self.send_ws_error(websocket, f"Error in orchestrator: {str(e)}", MessageSource.ORCHESTRATOR, session_id=session_id)
            await self._set_state(AgentState.ERROR)
            user_response = await self.user_interaction_agent.process(user_request, f'Error: {str(e)}')
            await self._log_exit("route_request", success=False, error=str(e))
            return {'response': user_response.get('user_message'), 'success': False}

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
