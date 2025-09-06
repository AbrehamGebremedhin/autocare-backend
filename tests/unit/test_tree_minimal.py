#!/usr/bin/env python3
"""
Minimal test for diagnosis tree functionality without external dependencies
"""
import asyncio
import json
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

# Import only the core tree functionality
from app.utils.diagnosis_tree import DiagnosisTreeNode

async def test_minimal_tree():
    """Test basic tree functionality without agents"""
    
    # Create a root tree
    root = DiagnosisTreeNode(issue_name='Root Issue', likelyhood=1.0)
    
    # Add some child nodes
    child1 = DiagnosisTreeNode(issue_name='Engine Misfire', likelyhood=0.75)
    child2 = DiagnosisTreeNode(issue_name='Fuel System Problem', likelyhood=0.60)
    
    root.add_child(child1)
    root.add_child(child2)
    
    assert len(root.children) == 2
    
    # Test tree dictionary representation
    tree_dict = root.to_dict()
    assert 'issue_name' in tree_dict
    assert 'children' in tree_dict
    assert len(tree_dict['children']) == 2
    
    # Test pruning
    initial_children_count = len(root.children)
    
    # Add a low likelihood child that should be pruned
    low_likelihood_child = DiagnosisTreeNode(issue_name='Low Probability Issue', likelyhood=0.05)
    root.add_child(low_likelihood_child)
    print(f"After adding low likelihood child: {len(root.children)} children")
    
    # Prune with threshold 0.1 (should remove the 0.05 likelihood child)
    pruned_nodes = root.prune(threshold=0.1)
    print(f"After pruning (threshold=0.1): {len(root.children)} children")
    print(f"Pruned nodes: {[node.issue_name for node in pruned_nodes]}")
    
    # Validate remaining children
    assert len(root.children) == 2  # Should have 2 children after pruning
    for child in root.children:
        assert child.likelyhood >= 0.1  # All remaining should be above threshold
    
    return root

if __name__ == "__main__":
    try:
        result = asyncio.run(test_minimal_tree())
        assert len(result.children) >= 0  # Basic validation
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
