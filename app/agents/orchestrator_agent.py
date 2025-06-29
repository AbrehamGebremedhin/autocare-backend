import json
from app.agents.symptom_extraction_agent import SymptomExtractorAgent
from app.agents.diagnostic_agent import DiagnosisAgent
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.agents.user_interaction_agent import UserInteractionAgent
from app.core.interfaces import IWebSocketManager
from app.agents.base_agent import BaseAgent
from app.utils.message_types import MessageSource

class OrchestratorAgent(BaseAgent):
    def __init__(self, websocket_manager: IWebSocketManager = None, **kwargs):
        super().__init__(websocket_manager=websocket_manager, **kwargs)
        self.agents = {
            'symptom_extraction': SymptomExtractorAgent,  # Store the class, not the instance
        }
        self.user_interaction_agent = UserInteractionAgent()

    async def route_request(self, user_request: str, user_id: str = None, context: dict = None, websocket=None, session_id=None):
        await self.send_ws_stage(websocket, "Orchestrator - Routing request", MessageSource.ORCHESTRATOR, session_id=session_id)
        """
        Main entry point: decides which agent should handle the user request.
        """
        car_id = None
        diagnosis_tree = None
        if context:
            car_id = context.get('car_id')
            diagnosis_tree = context.get('diagnosis_tree')  # Always use the session's current tree
        # If this is the initial message, extract symptoms
        if context and context.get('is_initial_message'):
            await self.send_ws_stage(websocket, "Orchestrator - Initial message, extracting symptoms", MessageSource.ORCHESTRATOR, session_id=session_id)
            return await self._handle_with_agent('symptom_extraction', user_request, car_id, diagnosis_tree, websocket=websocket, session_id=session_id)
        # For all other messages, use the current session's diagnosis_tree
        if car_id is not None:
            await self.send_ws_stage(websocket, "Orchestrator - Running diagnostic agent", MessageSource.ORCHESTRATOR, session_id=session_id)
            diagnostic_agent = DiagnosisAgent(car_id, diagnosis_tree=diagnosis_tree)
            diagnosis_result = await diagnostic_agent.process(user_request, websocket=websocket, session_id=session_id)
            # If diagnostic agent requests symptom extraction, process with symptom extractor, then re-run diagnosis
            if isinstance(diagnosis_result, dict) and diagnosis_result.get('need_symptom_extraction'):
                await self.send_ws_stage(websocket, "Orchestrator - Need symptom extraction after diagnosis", MessageSource.ORCHESTRATOR, session_id=session_id)
                # Run symptom extraction
                symptom_result = await self._handle_with_agent('symptom_extraction', user_request, car_id, diagnosis_tree, websocket=websocket, session_id=session_id)
                # Get updated tree if available
                updated_tree = symptom_result['tree'] if isinstance(symptom_result, dict) and 'tree' in symptom_result else diagnosis_tree
                # Re-run diagnosis with updated tree
                diagnostic_agent = DiagnosisAgent(car_id, diagnosis_tree=updated_tree)
                await self.send_ws_stage(websocket, "Orchestrator - Re-running diagnosis with updated tree", MessageSource.ORCHESTRATOR, session_id=session_id)
                diagnosis_result = await diagnostic_agent.process(user_request, websocket=websocket, session_id=session_id)
            await self.send_ws_stage(websocket, "Orchestrator - Running user interaction agent", MessageSource.ORCHESTRATOR, session_id=session_id)
            user_response = await self.user_interaction_agent.process(user_request, diagnosis_result)
            await self.send_ws_stage(websocket, "Orchestrator - Done", MessageSource.ORCHESTRATOR, session_id=session_id)
            return {
                'response': user_response.get('user_message'),
                'success': user_response.get('success', True),
                'step_by_step_guide': diagnosis_result.get('step_by_step_guide')
            }
        else:
            await self.send_ws_stage(websocket, "Orchestrator - No car_id provided, cannot diagnose", MessageSource.ORCHESTRATOR, session_id=session_id)
            user_response = await self.user_interaction_agent.process(user_request, 'car_id is required for diagnosis.')
            return {'response': user_response.get('user_message'), 'success': user_response.get('success', True)}

    def _is_chat_request(self, user_request: str) -> bool:
        # Simple heuristic: treat as chat if not a specific agent keyword
        keywords = ['symptom', 'diagnosis', 'extract']
        return not any(k in user_request.lower() for k in keywords)

    async def _handle_with_agent(self, agent_key: str, request: str, car_id=None, diagnosis_tree=None, websocket=None, session_id=None):
        agent_class = self.agents.get(agent_key)
        if not agent_class:
            user_response = await self.user_interaction_agent.process(request, f'Agent {agent_key} not found.')
            return {'response': user_response.get('user_message'), 'success': user_response.get('success', True)}
        if car_id is not None:
            agent = agent_class(car_id, diagnosis_tree=diagnosis_tree)
        else:
            user_response = await self.user_interaction_agent.process(request, 'car_id is required for symptom extraction.')
            return {'response': user_response.get('user_message'), 'success': user_response.get('success', True)}
        process_result = await agent.process(request, websocket=websocket, session_id=session_id)
        result = process_result.get('result')
        success = process_result.get('success', True)
        if isinstance(result, dict) and result.get('need_more_info'):
            info_type = result.get('info_type', 'additional information')
            user_response = await self.user_interaction_agent.process(request, f"Could you please provide more details about: {info_type}?")
            return {
                'response': user_response.get('user_message'),
                'need_more_info': True,
                'info_type': info_type,
                'success': user_response.get('success', True)
            }
        if not success:
            user_response = await self.user_interaction_agent.process(request, 'An error occurred during processing.')
            return {'response': user_response.get('user_message'), 'success': user_response.get('success', True)}
        # After successful symptom extraction, run diagnostic agent
        if agent_key == 'symptom_extraction':
            updated_tree = process_result.get('tree')
            diagnostic_agent = DiagnosisAgent(car_id, diagnosis_tree=updated_tree)
            diagnosis_result = await diagnostic_agent.process(request)
            # Always pass the diagnosis result through UserInteractionAgent
            user_response = await self.user_interaction_agent.process(request, diagnosis_result)
            return {'response': user_response.get('user_message'), 'success': user_response.get('success', True)}
        # For all other agent results, pass through UserInteractionAgent
        user_response = await self.user_interaction_agent.process(request, result)
        return {'response': user_response.get('user_message'), 'success': user_response.get('success', True)}

    def request_more_info(self, info_type: str, from_agent: str = None):
        # Orchestrator can ask other agents or the user for more info
        # For now, just return a dummy value or extend as needed
        if from_agent and from_agent in self.agents:
            return self.agents[from_agent].provide_info(info_type)
        return f'Dummy info for {info_type}'

    async def process(self, user_request: str, user_id: str = None, context: dict = None) -> dict:
        """
        Standard entry point for all orchestrator interactions.
        Calls route_request and returns a standardized result dict.
        """
        try:
            result = await self.route_request(user_request, user_id=user_id, context=context)
            return {
                'result': result.get('response'),
                'success': result.get('success', True)
            }
        except Exception as e:
            user_response = await self.user_interaction_agent.process(user_request, f'Error: {str(e)}')
            return {
                'result': user_response.get('user_message'),
                'success': False
            }
