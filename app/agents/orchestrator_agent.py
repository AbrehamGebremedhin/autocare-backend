from app.agents.symptom_extraction_agent import SymptomExtractorAgent

class OrchestratorAgent:
    def __init__(self):
        self.agents = {
            'symptom_extraction': SymptomExtractorAgent,  # Store the class, not the instance
        }

    async def route_request(self, user_request: str, user_id: str = None, context: dict = None):
        """
        Main entry point: decides which agent should handle the user request.
        """
        if 'symptom' in user_request.lower():
            car_id = None
            if context and 'car_id' in context:
                car_id = context['car_id']
            return self._handle_with_agent('symptom_extraction', user_request, car_id)
        # Add more routing logic here
        return {'response': 'No suitable agent found for this request.'}

    def _is_chat_request(self, user_request: str) -> bool:
        # Simple heuristic: treat as chat if not a specific agent keyword
        keywords = ['symptom', 'diagnosis', 'extract']
        return not any(k in user_request.lower() for k in keywords)

    def _handle_with_agent(self, agent_key: str, request: str, car_id=None):
        agent_class = self.agents.get(agent_key)
        if not agent_class:
            return {'response': f'Agent {agent_key} not found.'}
        if car_id is not None:
            agent = agent_class(car_id)
        else:
            return {'response': 'car_id is required for symptom extraction.'}
        response = agent.handle(request)
        # If agent needs more info, orchestrator can facilitate further interactions
        if response.get('need_more_info'):
            # Example: ask another agent or the user for more info
            more_info = self.request_more_info(response['info_type'])
            # Re-run the agent with additional info
            return agent.handle(request, more_info)
        return response

    def request_more_info(self, info_type: str, from_agent: str = None):
        # Orchestrator can ask other agents or the user for more info
        # For now, just return a dummy value or extend as needed
        if from_agent and from_agent in self.agents:
            return self.agents[from_agent].provide_info(info_type)
        return f'Dummy info for {info_type}'
