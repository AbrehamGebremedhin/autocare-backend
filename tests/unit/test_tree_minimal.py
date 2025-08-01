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
    
    print("=== Testing Basic DiagnosisTreeNode ===")
    
    # Create a root tree
    root = DiagnosisTreeNode(issue_name='Root Issue', likelyhood=1.0)
    print(f"Created root: {root.issue_name}, likelihood: {root.likelyhood}")
    print(f"Initial children count: {len(root.children)}")
    
    # Add some child nodes
    child1 = DiagnosisTreeNode(issue_name='Engine Misfire', likelyhood=0.75)
    child2 = DiagnosisTreeNode(issue_name='Fuel System Problem', likelyhood=0.60)
    
    root.add_child(child1)
    root.add_child(child2)
    
    print(f"After adding children: {len(root.children)}")
    for i, child in enumerate(root.children):
        print(f"  Child {i+1}: {child.issue_name} (likelihood: {child.likelyhood})")
    
    # Test tree dictionary representation
    print("\n=== Testing Tree Serialization ===")
    tree_dict = root.to_dict()
    print("Tree as dictionary:")
    print(json.dumps(tree_dict, indent=2))
    
    # Test pruning
    print("\n=== Testing Tree Pruning ===")
    print(f"Before pruning: {len(root.children)} children")
    
    # Add a low likelihood child that should be pruned
    low_likelihood_child = DiagnosisTreeNode(issue_name='Low Probability Issue', likelyhood=0.05)
    root.add_child(low_likelihood_child)
    print(f"After adding low likelihood child: {len(root.children)} children")
    
    # Prune with threshold 0.1 (should remove the 0.05 likelihood child)
    pruned_nodes = root.prune(threshold=0.1)
    print(f"After pruning (threshold=0.1): {len(root.children)} children")
    print(f"Pruned nodes: {[node.issue_name for node in pruned_nodes]}")
    
    # Show remaining children
    for i, child in enumerate(root.children):
        print(f"  Remaining child {i+1}: {child.issue_name} (likelihood: {child.likelyhood})")
    
    print("\n=== Test Complete ===")
    return root

if __name__ == "__main__":
    try:
        result = asyncio.run(test_minimal_tree())
        print(f"Final tree has {len(result.children)} children")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
