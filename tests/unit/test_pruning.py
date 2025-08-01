#!/usr/bin/env python3
"""
Simple test to verify the diagnosis tree pruning issue
"""
from app.utils.diagnosis_tree import DiagnosisTreeNode

def test_tree_pruning():
    """Test if the pruning is removing symptoms incorrectly"""
    
    # Create a root diagnosis tree
    diagnosis_tree = DiagnosisTreeNode(issue_name='root', likelyhood=1.0)
    print(f"Initial tree: {diagnosis_tree.issue_name}, children: {len(diagnosis_tree.children)}")
    
    # Add some test symptoms directly to the tree
    symptoms = [
        {"name": "Engine Knock", "likelihood": 0.85},  # 85%
        {"name": "Engine Misfire", "likelihood": 0.75},  # 75%
        {"name": "Fuel System Problem", "likelihood": 0.60},  # 60%
        {"name": "Low Battery", "likelihood": 0.25},  # 25% - This should be pruned with default threshold
        {"name": "Minor Oil Leak", "likelihood": 0.15},  # 15% - This should definitely be pruned
    ]
    
    for symptom in symptoms:
        child = DiagnosisTreeNode(
            issue_name=symptom["name"], 
            likelyhood=symptom["likelihood"],
            data=symptom
        )
        diagnosis_tree.add_child(child)
        print(f"Added: {symptom['name']} with likelihood {symptom['likelihood']}")
    
    print(f"\nBefore pruning - children: {len(diagnosis_tree.children)}")
    for child in diagnosis_tree.children:
        print(f"  - {child.issue_name}: {child.likelyhood}")
    
    # Test pruning with default threshold (0.3)
    print(f"\nPruning with default threshold (0.3)...")
    diagnosis_tree.prune(0.3)
    
    print(f"\nAfter pruning with 0.3 threshold - children: {len(diagnosis_tree.children)}")
    for child in diagnosis_tree.children:
        print(f"  - {child.issue_name}: {child.likelyhood}")
    
    # Test with a lower threshold
    print(f"\nAdding symptoms back...")
    for symptom in symptoms:
        child = DiagnosisTreeNode(
            issue_name=symptom["name"], 
            likelyhood=symptom["likelihood"],
            data=symptom
        )
        diagnosis_tree.add_child(child)
    
    print(f"\nPruning with threshold 0.2...")
    diagnosis_tree.prune(0.2)
    
    print(f"\nAfter pruning with 0.2 threshold - children: {len(diagnosis_tree.children)}")
    for child in diagnosis_tree.children:
        print(f"  - {child.issue_name}: {child.likelyhood}")

if __name__ == "__main__":
    test_tree_pruning()
