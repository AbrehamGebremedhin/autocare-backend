from app.agents.symptom_extraction_agent import SymptomExtractionAgent
from app.services.chat_service import ChatService

class OrchestratorAgent:
    def __init__(self):
        self.agents = {
            'symptom_extraction': SymptomExtractionAgent(),
            'chat': ChatService(),
            # Add other agents here
        }

    async def route_request(self, user_request: str, user_id: str = None, context: dict = None):
        """
        Main entry point: decides which agent should handle the user request.
        """
        # Example: route to chat agent for general chat or fallback
        if self._is_chat_request(user_request):
            return await self._handle_with_chat_agent(user_id, user_request, context)
        if 'symptom' in user_request.lower():
            return self._handle_with_agent('symptom_extraction', user_request)
        # Add more routing logic here
        return {'response': 'No suitable agent found for this request.'}

    def _is_chat_request(self, user_request: str) -> bool:
        # Simple heuristic: treat as chat if not a specific agent keyword
        keywords = ['symptom', 'diagnosis', 'extract']
        return not any(k in user_request.lower() for k in keywords)

    async def _handle_with_chat_agent(self, user_id: str, message: str, context: dict = None):
        chat_agent = self.agents.get('chat')
        if not chat_agent:
            return {'response': 'Chat agent not available.'}
        # ChatService expects async send_message
        return await chat_agent.send_message(user_id, message, context)

    def _handle_with_agent(self, agent_key: str, request: str):
        agent = self.agents.get(agent_key)
        if not agent:
            return {'response': f'Agent {agent_key} not found.'}
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

# Example usage (async):
# orchestrator = OrchestratorAgent()
# result = await orchestrator.route_request('How do I change my oil filter?', user_id='user123')
# print(result)