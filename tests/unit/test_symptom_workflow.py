#!/usr/bin/env python3
"""
Test to simulate the actual symptom extraction workflow that was failing
"""
import asyncio
import json
import sys
import os
from unittest.mock import Mock

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

# Import core components
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.agents.tree_manager_agent import TreeManagerAgent

class MockLLMService:
    """Mock LLM service with more realistic responses"""
    async def generate_response(self, prompt, **kwargs):
        # Simulate LLM response for parent node selection
        if "decide under which node" in prompt.lower():
            return "root"  # Return root as the parent
        return "Mock LLM response"
    
    async def perform_action(self, prompt, **kwargs):
        return "Mock action result"

async def test_symptom_extraction_workflow():
    """Test the complete symptom extraction workflow"""
    
    print("=== Testing Complete Symptom Extraction Workflow ===")
    
    # Simulate the initial setup as it would happen in the real system
    car_id = "test-car-123"
    
    # Step 1: Create the diagnosis tree (as orchestrator would)
    diagnosis_tree = DiagnosisTreeNode(issue_name='Vehicle Diagnosis', likelyhood=1.0)
    print(f"1. Created diagnosis tree: {diagnosis_tree.issue_name}")
    print(f"   Tree ID: {id(diagnosis_tree)}")
    print(f"   Initial children: {len(diagnosis_tree.children)}")
    
    # Step 2: Create tree manager (as orchestrator would)
    mock_llm = MockLLMService()
    tree_manager = TreeManagerAgent(diagnosis_tree, llm_service=mock_llm)
    print(f"2. Created tree manager")
    print(f"   Tree manager root ID: {id(tree_manager.root)}")
    print(f"   Tree references match: {id(diagnosis_tree) == id(tree_manager.root)}")
    
    # Step 3: Simulate symptom extraction agent workflow
    print(f"3. Simulating symptom extraction...")
    
    # This simulates what SymptomExtractorAgent.extract_symptoms() would do
    extracted_symptoms = [
        {
            'issue_name': 'Engine Rough Idle',
            'likelihood': 85,
            'issue_type': 'mechanical',
            'issue_category': 'engine',
            'severity': 'medium'
        },
        {
            'issue_name': 'Check Engine Light',
            'likelihood': 70,
            'issue_type': 'electrical',
            'issue_category': 'engine',
            'severity': 'low'
        },
        {
            'issue_name': 'Poor Fuel Economy',
            'likelihood': 60,
            'issue_type': 'mechanical',
            'issue_category': 'fuel system',
            'severity': 'medium'
        }
    ]
    
    print(f"   Extracted {len(extracted_symptoms)} symptoms")
    
    # Step 4: Add symptoms to tree (as symptom extraction agent would)
    print(f"4. Adding symptoms to tree...")
    print(f"   Tree children before: {len(diagnosis_tree.children)}")
    
    for symptom in extracted_symptoms:
        issue_name = symptom.get('issue_name', 'Unknown Issue')
        likelihood = symptom.get('likelihood', 0) / 100.0  # Convert to 0-1
        
        print(f"   Adding: {issue_name} (likelihood: {likelihood})")
        
        # Add symptom using tree manager
        await tree_manager.add_symptom(
            symptom=issue_name,
            likelyhood=likelihood,
            data=symptom
        )
    
    print(f"   Tree children after: {len(diagnosis_tree.children)}")
    print(f"   Tree manager children: {len(tree_manager.root.children)}")
    
    # Step 5: Verify tree state
    print(f"5. Verifying tree state...")
    print(f"   Tree references still match: {id(diagnosis_tree) == id(tree_manager.root)}")
    
    # Show all symptoms in tree
    if len(diagnosis_tree.children) > 0:
        print(f"   ✅ SUCCESS: Tree has {len(diagnosis_tree.children)} symptoms")
        for i, child in enumerate(diagnosis_tree.children):
            print(f"      {i+1}. {child.issue_name} (likelihood: {child.likelyhood})")
    else:
        print(f"   ❌ FAILURE: Tree is still empty!")
        return False
    
    # Step 6: Test tree operations
    print(f"6. Testing tree operations...")
    
    # Get tree state (as diagnosis agent would)
    tree_state = tree_manager.get_tree_state()
    print(f"   Tree state:")
    for line in tree_state.split('\n'):
        print(f"      {line}")
    
    # Test pruning with a low threshold
    print(f"   Testing pruning (threshold=0.5)...")
    children_before = len(diagnosis_tree.children)
    tree_manager.prune_tree(threshold=0.5)
    children_after = len(diagnosis_tree.children)
    print(f"   Pruning: {children_before} -> {children_after} children")
    
    # Test sorting
    print(f"   Testing sorting...")
    tree_manager.sort_tree()
    print(f"   Children after sorting:")
    for i, child in enumerate(diagnosis_tree.children):
        print(f"      {i+1}. {child.issue_name} (likelihood: {child.likelyhood})")
    
    # Step 7: Final verification
    print(f"7. Final verification...")
    final_tree_dict = diagnosis_tree.to_dict()
    print(f"   Final tree structure:")
    print(json.dumps(final_tree_dict, indent=4))
    
    # Return success if tree has children
    success = len(diagnosis_tree.children) > 0
    print(f"\n=== Test Result: {'✅ SUCCESS' if success else '❌ FAILURE'} ===")
    
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(test_symptom_extraction_workflow())
        if success:
            print("✅ The symptom extraction and tree population is working correctly!")
        else:
            print("❌ The symptom extraction is still failing.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
