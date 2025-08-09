#!/usr/bin/env python3
"""
Comprehensive test to simulate the full diagnosis flow and identify the tree issue
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

class MockLLMService:
    def __init__(self):
        pass
    
    async def generate_response(self, prompt):
        # Mock a typical symptom extraction response
        if "symptom" in prompt.lower() or "issue" in prompt.lower():
            return """[
                {
                    "issue_name": "Engine Misfire",
                    "likelihood": 75,
                    "issue_type": "mechanical",
                    "issue_category": "engine",
                    "severity": "medium",
                    "issue_description": "Engine cylinder not firing properly"
                },
                {
                    "issue_name": "Faulty Spark Plugs",
                    "likelihood": 60,
                    "issue_type": "mechanical", 
                    "issue_category": "ignition",
                    "severity": "low",
                    "issue_description": "Worn or fouled spark plugs"
                }
            ]"""
        # Mock tree manager response
        return "root"
    
    def get_llm(self):
        return self

    async def ainvoke(self, prompt):
        return await self.generate_response(prompt)

    def invoke(self, prompt):
        return self.generate_response(prompt)

class MockCarCRUD:
    async def get_car_by_id(self, car_id):
        return {
            'id': car_id,
            'make': 'Toyota',
            'model': 'Camry', 
            'year': '2020',
            'car_guide_links': []
        }
        
    async def get_owner_manual_text(self, car_id):
        return "Owner manual text for car"

class MockEmbeddingService:
    def __init__(self):
        self.search_engine_service = self
    
    async def embed_text(self, text):
        return [0.1] * 10
    
    async def embed_texts(self, texts):
        return [[0.1] * 10 for _ in texts]
    
    async def embed_and_vector_search(self, car_id, query, top_k=1):
        return [{"chunk": "Manual text chunk"}]

class MockSearchEngineService:
    async def embed_and_vector_search(self, car_id, query, top_k=1):
        return [{"chunk": "Manual text chunk"}]

class MockScraperService:
    def __init__(self, headless=True):
        pass
    
    async def perform_action(self, urls, limit=None):
        return []

async def test_full_symptom_extraction_flow():
    """Test the full symptom extraction flow with mocked dependencies"""
    
    print("=== Testing Full Symptom Extraction Flow ===")
    
    # Import after setting up mocks
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from app.utils.diagnosis_tree import DiagnosisTreeNode
    from app.agents.tree_manager_agent import TreeManagerAgent
    from app.agents.symptom_extraction_agent import SymptomExtractorAgent
    
    # Create initial tree
    diagnosis_tree = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
    print(f"Initial tree: {diagnosis_tree.issue_name}, children: {len(diagnosis_tree.children)}, id: {id(diagnosis_tree)}")
    
    # Create tree manager
    tree_manager = TreeManagerAgent(diagnosis_tree, llm_service=MockLLMService())
    print(f"TreeManager root id: {id(tree_manager.root)}, same as tree: {tree_manager.root is diagnosis_tree}")
    
    # Create symptom extraction agent with the tree and tree manager
    symptom_agent = SymptomExtractorAgent(
        car_id="test-car-123",
        diagnosis_tree=diagnosis_tree,
        car_make="Toyota",
        car_model="Camry", 
        car_year="2020",
        tree_manager_agent=tree_manager,
        llm_service=MockLLMService(),
        embedding_service=MockEmbeddingService(),
        search_engine_service=MockSearchEngineService(),
        scraper_service=MockScraperService()
    )
    
    # Override the car_crud with mock
    symptom_agent.car_crud = MockCarCRUD()
    
    print(f"SymptomAgent tree id: {id(symptom_agent.diagnosis_tree)}")
    print(f"SymptomAgent tree_manager root id: {id(symptom_agent.tree_manager_agent.root)}")
    print(f"All references same: {symptom_agent.diagnosis_tree is diagnosis_tree is tree_manager.root}")
    
    # Test direct symptom addition to verify tree manager works
    print("\n--- Testing direct tree manager symptom addition ---")
    await tree_manager.add_symptom("Test Direct Symptom", 0.9, data={"test": True})
    print(f"After direct addition - tree children: {len(diagnosis_tree.children)}")
    
    # Now test the handle method directly with mock symptoms
    print("\n--- Testing symptom agent handle method ---")
    
    # Simulate the parsed_result that would come from LLM
    mock_parsed_result = [
        {
            'issue_name': 'Engine Misfire',
            'likelihood': 75,
            'issue_type': 'mechanical',
            'issue_category': 'engine',
            'severity': 'medium'
        },
        {
            'issue_name': 'Faulty Spark Plugs',
            'likelihood': 60,
            'issue_type': 'mechanical', 
            'issue_category': 'ignition',
            'severity': 'low'
        }
    ]
    
    # Manually test the symptom addition logic
    tree_before = len(diagnosis_tree.children)
    print(f"Tree children before symptom addition: {tree_before}")
    
    if symptom_agent.tree_manager_agent is not None and isinstance(mock_parsed_result, list):
        for issue in mock_parsed_result:
            issue_name = issue.get('issue_name', 'Unknown Issue')
            likelihood = issue.get('likelihood', 0) / 100.0  # Convert to 0-1 float
            print(f"Adding symptom: {issue_name} with likelihood {likelihood}")
            await symptom_agent.tree_manager_agent.add_symptom(
                symptom=issue_name,
                likelyhood=likelihood,
                data=issue
            )
        
        # Optionally prune and sort after adding
        symptom_agent.tree_manager_agent.prune_tree()
        symptom_agent.tree_manager_agent.sort_tree()
    
    tree_after = len(diagnosis_tree.children)
    print(f"Tree children after symptom addition: {tree_after}")
    print(f"Children added: {tree_after - tree_before}")
    
    # Print final tree state
    final_tree_state = tree_manager.get_tree_state()
    print(f"\nFinal tree state:\n{final_tree_state}")
    
    # Test tree serialization
    tree_dict = diagnosis_tree.to_dict()
    print(f"\nSerialized tree:\n{json.dumps(tree_dict, indent=2)}")
    
    return diagnosis_tree

if __name__ == "__main__":
    asyncio.run(test_full_symptom_extraction_flow())
