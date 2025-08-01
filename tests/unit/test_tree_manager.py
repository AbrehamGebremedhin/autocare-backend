#!/usr/bin/env python3
"""
Test TreeManagerAgent functionality without external dependencies
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
    """Mock LLM service to avoid dependency issues"""
    async def generate_response(self, prompt, **kwargs):
        return "Mock LLM response"
    
    async def perform_action(self, prompt, **kwargs):
        return "Mock action result"

async def test_tree_manager():
    """Test TreeManagerAgent functionality"""
    
    print("=== Testing TreeManagerAgent ===")
    
    # Create a root tree
    root = DiagnosisTreeNode(issue_name='Root Issue', likelyhood=1.0)
    print(f"Created root: {root.issue_name}")
    
    # Create tree manager with mock LLM service
    mock_llm = MockLLMService()
    tree_manager = TreeManagerAgent(root, llm_service=mock_llm)
    
    print(f"Tree manager created with tree: {tree_manager.root.issue_name}")
    print(f"Initial tree children: {len(tree_manager.root.children)}")
    
    # Test adding symptoms
    print("\n=== Testing Add Symptom ===")
    
    # Add first symptom
    await tree_manager.add_symptom(
        symptom="Engine Knocking",
        likelyhood=0.80,
        data={"category": "engine", "severity": "high"}
    )
    
    print(f"After adding 'Engine Knocking': {len(tree_manager.root.children)} children")
    
    # Add second symptom
    await tree_manager.add_symptom(
        symptom="Rough Idle", 
        likelyhood=0.65,
        data={"category": "engine", "severity": "medium"}
    )
    
    print(f"After adding 'Rough Idle': {len(tree_manager.root.children)} children")
    
    # Add third symptom
    await tree_manager.add_symptom(
        symptom="Low Fuel Pressure",
        likelyhood=0.45,
        data={"category": "fuel", "severity": "medium"}
    )
    
    print(f"After adding 'Low Fuel Pressure': {len(tree_manager.root.children)} children")
    
    # Show all children
    print("\nCurrent tree children:")
    for i, child in enumerate(tree_manager.root.children):
        print(f"  {i+1}. {child.issue_name} (likelihood: {child.likelyhood})")
        if child.data:
            print(f"     Data: {child.data}")
    
    # Test tree state
    print("\n=== Testing Tree State ===")
    tree_state = tree_manager.get_tree_state()
    print("Tree state:")
    print(tree_state)
    
    # Test pruning
    print("\n=== Testing Pruning ===")
    print(f"Before pruning: {len(tree_manager.root.children)} children")
    tree_manager.prune_tree(threshold=0.5)  # Should remove "Low Fuel Pressure" (0.45)
    print(f"After pruning (threshold=0.5): {len(tree_manager.root.children)} children")
    
    print("Remaining children after pruning:")
    for i, child in enumerate(tree_manager.root.children):
        print(f"  {i+1}. {child.issue_name} (likelihood: {child.likelyhood})")
    
    # Test sorting
    print("\n=== Testing Sorting ===")
    
    # Add another symptom to test sorting
    await tree_manager.add_symptom(
        symptom="High RPM",
        likelyhood=0.90,
        data={"category": "engine", "severity": "high"}
    )
    
    print("Before sorting:")
    for i, child in enumerate(tree_manager.root.children):
        print(f"  {i+1}. {child.issue_name} (likelihood: {child.likelyhood})")
    
    tree_manager.sort_tree()
    
    print("After sorting (should be in descending order of likelihood):")
    for i, child in enumerate(tree_manager.root.children):
        print(f"  {i+1}. {child.issue_name} (likelihood: {child.likelyhood})")
    
    # Test final tree serialization
    print("\n=== Final Tree Serialization ===")
    final_tree_dict = tree_manager.root.to_dict()
    print("Final tree structure:")
    print(json.dumps(final_tree_dict, indent=2))
    
    print("\n=== TreeManagerAgent Test Complete ===")
    return tree_manager

if __name__ == "__main__":
    try:
        result = asyncio.run(test_tree_manager())
        print(f"Test completed successfully. Final tree has {len(result.root.children)} children")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
