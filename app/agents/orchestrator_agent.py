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
        
        # Debug logging
        print(f"DEBUG: SymptomExtractionCommand created agent with tree id: {id(diagnosis_tree) if diagnosis_tree else 'None'}")
        
        process_result = await agent.process(user_request, websocket=websocket, session_id=session_id)
        
        result = process_result.get('result')
        success = process_result.get('success', True)
        updated_tree = process_result.get('diagnosis_tree')
        
        # Debug logging
        print(f"DEBUG: SymptomExtractionCommand - result type: {type(result)}, updated_tree id: {id(updated_tree) if updated_tree else 'None'}")
        if updated_tree:
            print(f"DEBUG: SymptomExtractionCommand - updated tree children: {len(updated_tree.children)}")
        
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
        
        # Debug logging
        print(f"DEBUG: DiagnosisCommand created agent with tree id: {id(diagnosis_tree) if diagnosis_tree else 'None'}")
        if diagnosis_tree:
            print(f"DEBUG: DiagnosisCommand - tree children: {len(diagnosis_tree.children)}")
        
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
        # Debug logging
        initial_tree = context.get('diagnosis_tree')
        print(f"DEBUG: InitialProcessingCommand - initial tree id: {id(initial_tree) if initial_tree else 'None'}")
        if initial_tree:
            print(f"DEBUG: InitialProcessingCommand - initial tree children: {len(initial_tree.children)}")
        
        # Execute symptom extraction
        symptom_result = await self.symptom_command.execute(context)
        
        if 'diagnosis_tree' in symptom_result:
            context['diagnosis_tree'] = symptom_result['diagnosis_tree']
            updated_tree = symptom_result['diagnosis_tree']
            print(f"DEBUG: InitialProcessingCommand - updated tree from symptom extraction id: {id(updated_tree) if updated_tree else 'None'}")
            if updated_tree:
                print(f"DEBUG: InitialProcessingCommand - updated tree children: {len(updated_tree.children)}")
        
        # Execute diagnosis with updated tree
        diagnosis_result = await self.diagnosis_command.execute(context)
        
        final_tree = context.get('diagnosis_tree')
        print(f"DEBUG: InitialProcessingCommand - final tree id: {id(final_tree) if final_tree else 'None'}")
        if final_tree:
            print(f"DEBUG: InitialProcessingCommand - final tree children: {len(final_tree.children)}")
        
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
        diagnosis_tree = context.get('diagnosis_tree')
        car_id = context.get('car_id')
        
        if diagnosis_tree is None and car_id is not None:
            diagnosis_tree = get_diagnosis_tree(issue_name='root', likelyhood=1.0)
            context['diagnosis_tree'] = diagnosis_tree
            
            # Debug logging
            print(f"DEBUG: Orchestrator created new diagnosis tree with id: {id(diagnosis_tree)}")
            
            websocket = context.get('websocket')
            session_id = context.get('session_id')
            await self.send_ws_stage(
                websocket, 
                "Orchestrator - Created new diagnosis tree", 
                MessageSource.ORCHESTRATOR, 
                session_id=session_id
            )
        elif diagnosis_tree is not None:
            print(f"DEBUG: Orchestrator using existing diagnosis tree with id: {id(diagnosis_tree)}, children: {len(diagnosis_tree.children)}")
        
        return diagnosis_tree
    
    async def route_request(self, user_request: str, user_id: str = None, context: dict = None, websocket=None, session_id=None):
        """
        Main entry point: decides which command should handle the user request.
        """
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
        await self._ensure_diagnosis_tree(command_context)
        
        car_id = command_context.get('car_id')
        
        # Route based on request type and context
        try:
            if command_context.get('is_initial_message'):
                await self.send_ws_stage(websocket, "Orchestrator - Processing initial message", MessageSource.ORCHESTRATOR, session_id=session_id)
                return await self.execute_command('initial_processing', command_context)
            
            elif car_id is not None:
                await self.send_ws_stage(websocket, "Orchestrator - Processing diagnosis request", MessageSource.ORCHESTRATOR, session_id=session_id)
                diagnosis_result = await self.execute_command('diagnosis', command_context)
                
                # Check if symptom extraction is needed
                if isinstance(diagnosis_result.get('diagnosis_result'), dict) and \
                   diagnosis_result['diagnosis_result'].get('need_symptom_extraction'):
                    
                    await self.send_ws_stage(websocket, "Orchestrator - Need symptom extraction after diagnosis", MessageSource.ORCHESTRATOR, session_id=session_id)
                    
                    # Execute symptom extraction
                    symptom_result = await self.execute_command('symptom_extraction', command_context)
                    
                    # Update context with new tree and re-run diagnosis
                    if 'diagnosis_tree' in symptom_result:
                        command_context['diagnosis_tree'] = symptom_result['diagnosis_tree']
                    
                    await self.send_ws_stage(websocket, "Orchestrator - Re-running diagnosis with updated tree", MessageSource.ORCHESTRATOR, session_id=session_id)
                    diagnosis_result = await self.execute_command('diagnosis', command_context)
                
                await self.send_ws_stage(websocket, "Orchestrator - Done", MessageSource.ORCHESTRATOR, session_id=session_id)
                return diagnosis_result
            
            else:
                await self.send_ws_stage(websocket, "Orchestrator - No car_id provided, cannot diagnose", MessageSource.ORCHESTRATOR, session_id=session_id)
                user_response = await self.user_interaction_agent.process(user_request, 'car_id is required for diagnosis.')
                return {'response': user_response.get('user_message'), 'success': user_response.get('success', True)}
                
        except Exception as e:
            await self.logger.error(f"Error in route_request: {str(e)}")
            await self._set_state(AgentState.ERROR)
            user_response = await self.user_interaction_agent.process(user_request, f'Error: {str(e)}')
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
        try:
            if self.state == AgentState.INACTIVE:
                await self.initialize()
            
            result = await self.route_request(user_request, user_id=user_id, context=context)
            return {
                'result': result.get('response'),
                'success': result.get('success', True),
                'step_by_step_guide': result.get('step_by_step_guide'),
                'diagnosis_tree': result.get('diagnosis_tree')
            }
        except Exception as e:
            await self.logger.error(f"Error in orchestrator process: {str(e)}")
            await self._set_state(AgentState.ERROR)
            user_response = await self.user_interaction_agent.process(user_request, f'Error: {str(e)}')
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
