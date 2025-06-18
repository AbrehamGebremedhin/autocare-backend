from app.agents.symptom_extraction_agent import SymptomExtractorAgent
from app.agents.diagnostic_agent import DiagnosisAgent
from app.utils.diagnosis_tree import DiagnosisTreeNode

class OrchestratorAgent:
    def __init__(self):
        self.agents = {
            'symptom_extraction': SymptomExtractorAgent,  # Store the class, not the instance
        }

    async def route_request(self, user_request: str, user_id: str = None, context: dict = None):
        """
        Main entry point: decides which agent should handle the user request.
        """
        car_id = None
        diagnosis_tree = None
        if context:
            car_id = context.get('car_id')
            diagnosis_tree = context.get('diagnosis_tree')
        # If this is the initial message, extract symptoms
        if context and context.get('is_initial_message'):
            return await self._handle_with_agent('symptom_extraction', user_request, car_id, diagnosis_tree)
        if 'symptom' in user_request.lower():
            return await self._handle_with_agent('symptom_extraction', user_request, car_id, diagnosis_tree)
        # Add more routing logic here
        return {'response': 'No suitable agent found for this request.'}

    def _is_chat_request(self, user_request: str) -> bool:
        # Simple heuristic: treat as chat if not a specific agent keyword
        keywords = ['symptom', 'diagnosis', 'extract']
        return not any(k in user_request.lower() for k in keywords)

    async def _handle_with_agent(self, agent_key: str, request: str, car_id=None, diagnosis_tree=None):
        agent_class = self.agents.get(agent_key)
        if not agent_class:
            return {'response': f'Agent {agent_key} not found.'}
        if car_id is not None:
            agent = agent_class(car_id, diagnosis_tree=diagnosis_tree)
        else:
            return {'response': 'car_id is required for symptom extraction.'}
        # Always use the process function for agent interaction
        process_result = await agent.process(request)
        result = process_result.get('result')
        success = process_result.get('success', True)
        # If agent needs more info, orchestrator asks the user for clarification
        if isinstance(result, dict) and result.get('need_more_info'):
            info_type = result.get('info_type', 'additional information')
            return {
                'response': f"Could you please provide more details about: {info_type}?",
                'need_more_info': True,
                'info_type': info_type
            }
        if not success:
            return {'response': 'An error occurred during processing.'}
        # After successful symptom extraction, run diagnostic agent
        if agent_key == 'symptom_extraction':
            updated_tree = process_result.get('tree')
            diagnostic_agent = DiagnosisAgent(car_id, diagnosis_tree=updated_tree)
            diagnosis_result = await diagnostic_agent.process(request)
            return diagnosis_result
        return result

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
                'result': result,
                'success': True
            }
        except Exception as e:
            return {
                'result': {'response': f'Error: {str(e)}'},
                'success': False
            }
