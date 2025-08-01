#!/usr/bin/env python3
"""
Simple test to isolate the tree issue without dependencies
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.diagnosis_tree import DiagnosisTreeNode

def test_basic_tree_operations():
    """Test basic tree operations to ensure they work correctly"""
    
    print("=== Basic Tree Operations Test ===")
    
    # Create root tree
    root = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
    print(f"Root created: {root.issue_name}, children: {len(root.children)}")
    
    # Add children directly
    child1 = DiagnosisTreeNode(issue_name='Engine Issues', likelyhood=0.8, data={'category': 'engine'})
    child2 = DiagnosisTreeNode(issue_name='Electrical Issues', likelyhood=0.6, data={'category': 'electrical'})
    
    root.add_child(child1)
    root.add_child(child2)
    
    print(f"After adding children: {len(root.children)} children")
    for i, child in enumerate(root.children):
        print(f"  Child {i+1}: {child.issue_name} (likelihood: {child.likelyhood})")
    
    # Test tree serialization
    tree_dict = root.to_dict()
    print(f"Tree serialization works: {len(tree_dict['children'])} children in dict")
    
    # Test object reference sharing
    print("\n=== Object Reference Test ===")
    tree_ref1 = root
    tree_ref2 = root
    
    # Add child via first reference
    child3 = DiagnosisTreeNode(issue_name='Fuel System', likelyhood=0.7)
    tree_ref1.add_child(child3)
    
    print(f"tree_ref1 children: {len(tree_ref1.children)}")
    print(f"tree_ref2 children: {len(tree_ref2.children)}")
    print(f"root children: {len(root.children)}")
    print(f"References are same object: {tree_ref1 is tree_ref2 is root}")
    
    return root

if __name__ == "__main__":
    test_basic_tree_operations()
