#!/usr/bin/env python3
"""
Test hierarchical tree structure functionality - adding children to existing children
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
    """Mock LLM service with more intelligent responses for hierarchical testing"""
    
    def __init__(self):
        self.call_count = 0
    
    async def generate_response(self, prompt, **kwargs):
        self.call_count += 1
        
        # Simulate intelligent parent selection based on prompt content
        if "decide under which node" in prompt.lower():
            # For specific symptoms, return appropriate parent nodes
            if "timing belt" in prompt.lower() or "valve" in prompt.lower() or "piston" in prompt.lower():
                return "Engine Issues"  # Return engine parent for engine-specific symptoms
            elif "battery" in prompt.lower() or "alternator" in prompt.lower() or "wiring" in prompt.lower():
                return "Electrical Issues"  # Return electrical parent for electrical symptoms
            elif "belt tension" in prompt.lower() or "belt material" in prompt.lower():
                return "Timing Belt Issue"  # Return more specific parent for very specific symptoms
            else:
                return "root"  # Default to root for general symptoms
        
        return f"Mock LLM response #{self.call_count}"
    
    async def perform_action(self, prompt, **kwargs):
        return "Mock action result"

def count_total_nodes(node):
    """Count total number of nodes in the tree"""
    count = 1  # Count current node
    for child in node.children:
        count += count_total_nodes(child)
    return count

def get_tree_depth(node, current_depth=0):
    """Get the maximum depth of the tree"""
    if not node.children:
        return current_depth
    return max(get_tree_depth(child, current_depth + 1) for child in node.children)

def count_nodes_with_children(node):
    """Count nodes that have children"""
    count = 1 if node.children else 0
    for child in node.children:
        count += count_nodes_with_children(child)
    return count

def print_tree_hierarchy(node, depth=0):
    """Print tree structure with proper indentation"""
    indent = "  " * depth
    print(f"{indent}- {node.issue_name} (likelihood: {node.likelyhood:.2f})")
    if node.data:
        print(f"{indent}  Data: {node.data}")
    for child in node.children:
        print_tree_hierarchy(child, depth + 1)

async def test_hierarchical_tree_structure():
    """Test creating and managing hierarchical tree structures"""
    
    print("=== Testing Hierarchical Diagnosis Tree ===")
    
    # Step 1: Create root tree and manager
    diagnosis_tree = DiagnosisTreeNode(issue_name='Vehicle Diagnosis', likelyhood=1.0)
    mock_llm = MockLLMService()
    tree_manager = TreeManagerAgent(diagnosis_tree, llm_service=mock_llm)
    
    print(f"1. Created root tree: {diagnosis_tree.issue_name}")
    print(f"   Initial tree depth: {get_tree_depth(diagnosis_tree)}")
    print(f"   Initial node count: {count_total_nodes(diagnosis_tree)}")
    
    # Step 2: Add primary categories using tree manager
    print(f"\n2. Adding primary symptom categories...")
    
    primary_categories = [
        {"name": "Engine Issues", "likelihood": 0.85, "data": {"category": "engine", "type": "primary"}},
        {"name": "Electrical Issues", "likelihood": 0.70, "data": {"category": "electrical", "type": "primary"}},
        {"name": "Transmission Issues", "likelihood": 0.60, "data": {"category": "transmission", "type": "primary"}},
        {"name": "Brake Issues", "likelihood": 0.45, "data": {"category": "brakes", "type": "primary"}}
    ]
    
    for category in primary_categories:
        await tree_manager.add_symptom(
            symptom=category["name"],
            likelyhood=category["likelihood"],
            data=category["data"]
        )
        print(f"   Added primary category: {category['name']}")
    
    print(f"   Primary categories added. Tree now has {len(diagnosis_tree.children)} children")
    print(f"   Tree depth: {get_tree_depth(diagnosis_tree)}, Total nodes: {count_total_nodes(diagnosis_tree)}")
    
    # Step 3: Add secondary symptoms using tree manager (should find appropriate parents)
    print(f"\n3. Adding secondary symptoms (should be nested under appropriate parents)...")
    
    secondary_symptoms = [
        {"name": "Timing Belt Issue", "likelihood": 0.75, "data": {"sub_category": "timing", "urgency": "high"}},
        {"name": "Valve Problems", "likelihood": 0.65, "data": {"sub_category": "valves", "urgency": "medium"}},
        {"name": "Piston Ring Wear", "likelihood": 0.55, "data": {"sub_category": "pistons", "urgency": "low"}},
        {"name": "Battery Voltage Low", "likelihood": 0.80, "data": {"component": "battery", "voltage": "low"}},
        {"name": "Alternator Malfunction", "likelihood": 0.60, "data": {"component": "alternator", "output": "insufficient"}},
        {"name": "Wiring Harness Damage", "likelihood": 0.40, "data": {"component": "wiring", "condition": "damaged"}}
    ]
    
    for symptom in secondary_symptoms:
        await tree_manager.add_symptom(
            symptom=symptom["name"],
            likelyhood=symptom["likelihood"],
            data=symptom["data"]
        )
        print(f"   Added secondary symptom: {symptom['name']}")
    
    print(f"   Secondary symptoms added. Total nodes: {count_total_nodes(diagnosis_tree)}")
    print(f"   Tree depth: {get_tree_depth(diagnosis_tree)}")
    
    # Step 4: Add tertiary symptoms (grandchildren)
    print(f"\n4. Adding tertiary symptoms (should create 3-level hierarchy)...")
    
    tertiary_symptoms = [
        {"name": "Belt Tension Too Loose", "likelihood": 0.70, "data": {"detail": "tension", "severity": "moderate"}},
        {"name": "Belt Material Degradation", "likelihood": 0.50, "data": {"detail": "material", "severity": "high"}}
    ]
    
    for symptom in tertiary_symptoms:
        await tree_manager.add_symptom(
            symptom=symptom["name"],
            likelyhood=symptom["likelihood"],
            data=symptom["data"]
        )
        print(f"   Added tertiary symptom: {symptom['name']}")
    
    print(f"   Tertiary symptoms added. Total nodes: {count_total_nodes(diagnosis_tree)}")
    print(f"   Final tree depth: {get_tree_depth(diagnosis_tree)}")
    
    # Step 5: Display the complete hierarchical structure
    print(f"\n5. Complete hierarchical tree structure:")
    print_tree_hierarchy(diagnosis_tree)
    
    # Step 6: Test tree manager operations on hierarchical structure
    print(f"\n6. Testing tree manager operations on hierarchical structure...")
    
    # Tree state representation
    print(f"Tree state representation:")
    tree_state = tree_manager.get_tree_state()
    print(tree_state)
    
    # Test pruning with hierarchy
    print(f"\nTesting hierarchical pruning...")
    nodes_before = count_total_nodes(diagnosis_tree)
    print(f"Before pruning: {nodes_before} total nodes")
    
    tree_manager.prune_tree(threshold=0.5)  # Remove nodes with likelihood < 50%
    
    nodes_after = count_total_nodes(diagnosis_tree)
    print(f"After pruning (threshold=0.5): {nodes_after} total nodes")
    print(f"Pruned {nodes_before - nodes_after} nodes")
    
    # Test sorting with hierarchy
    print(f"\nTesting hierarchical sorting...")
    tree_manager.sort_tree()
    print("Tree sorted by likelihood (descending)")
    
    # Step 7: Final analysis
    print(f"\n7. Final tree analysis:")
    print(f"   Total nodes: {count_total_nodes(diagnosis_tree)}")
    print(f"   Maximum depth: {get_tree_depth(diagnosis_tree)}")
    print(f"   Nodes with children: {count_nodes_with_children(diagnosis_tree)}")
    print(f"   Leaf nodes: {count_total_nodes(diagnosis_tree) - count_nodes_with_children(diagnosis_tree)}")
    
    # Step 8: Final tree structure and serialization
    print(f"\n8. Final hierarchical tree structure after operations:")
    print_tree_hierarchy(diagnosis_tree)
    
    print(f"\nFinal tree serialization:")
    final_dict = diagnosis_tree.to_dict()
    print(json.dumps(final_dict, indent=2))
    
    return diagnosis_tree, tree_manager

async def test_manual_hierarchical_addition():
    """Test manually adding children to specific nodes"""
    
    print(f"\n=== Testing Manual Hierarchical Addition ===")
    
    # Create a simple tree
    root = DiagnosisTreeNode(issue_name='Car Problems', likelyhood=1.0)
    
    # Add primary nodes manually
    engine_node = DiagnosisTreeNode(issue_name='Engine Problems', likelyhood=0.80, data={"category": "engine"})
    electrical_node = DiagnosisTreeNode(issue_name='Electrical Problems', likelyhood=0.60, data={"category": "electrical"})
    
    root.add_child(engine_node)
    root.add_child(electrical_node)
    
    print(f"Created base tree with {len(root.children)} primary categories")
    
    # Add secondary nodes manually to specific parents
    timing_issue = DiagnosisTreeNode(issue_name='Timing Chain Issue', likelyhood=0.70, data={"type": "timing"})
    valve_issue = DiagnosisTreeNode(issue_name='Valve Seal Leak', likelyhood=0.50, data={"type": "valves"})
    
    engine_node.add_child(timing_issue)
    engine_node.add_child(valve_issue)
    
    battery_issue = DiagnosisTreeNode(issue_name='Dead Battery', likelyhood=0.85, data={"component": "battery"})
    alternator_issue = DiagnosisTreeNode(issue_name='Faulty Alternator', likelyhood=0.45, data={"component": "alternator"})
    
    electrical_node.add_child(battery_issue)
    electrical_node.add_child(alternator_issue)
    
    print(f"Added secondary symptoms:")
    print(f"  Engine problems: {len(engine_node.children)} children")
    print(f"  Electrical problems: {len(electrical_node.children)} children")
    
    # Add tertiary nodes (grandchildren)
    chain_stretch = DiagnosisTreeNode(issue_name='Chain Stretch', likelyhood=0.60, data={"detail": "wear"})
    chain_tensioner = DiagnosisTreeNode(issue_name='Tensioner Failure', likelyhood=0.40, data={"detail": "mechanical"})
    
    timing_issue.add_child(chain_stretch)
    timing_issue.add_child(chain_tensioner)
    
    print(f"Added tertiary symptoms under 'Timing Chain Issue': {len(timing_issue.children)} children")
    
    # Display the manually created hierarchy
    print(f"\nManually created hierarchical structure:")
    print_tree_hierarchy(root)
    
    # Test tree operations
    print(f"\nTree statistics:")
    print(f"  Total nodes: {count_total_nodes(root)}")
    print(f"  Maximum depth: {get_tree_depth(root)}")
    print(f"  Nodes with children: {count_nodes_with_children(root)}")
    
    # Test pruning on manual hierarchy
    print(f"\nTesting pruning on manual hierarchy (threshold=0.5)...")
    nodes_before = count_total_nodes(root)
    pruned = root.prune(threshold=0.5)
    nodes_after = count_total_nodes(root)
    
    print(f"Pruned {len(pruned)} nodes: {[node.issue_name for node in pruned]}")
    print(f"Tree nodes: {nodes_before} -> {nodes_after}")
    
    print(f"\nFinal manual hierarchy after pruning:")
    print_tree_hierarchy(root)
    
    return root

async def main():
    """Main test function"""
    print("🔍 Testing Hierarchical Tree Functionality\n")
    
    # Test 1: Tree manager with intelligent parent selection
    tree, manager = await test_hierarchical_tree_structure()
    
    # Test 2: Manual hierarchical tree construction
    manual_tree = await test_manual_hierarchical_addition()
    
    print(f"\n✅ All hierarchical tree tests completed successfully!")
    print(f"   TreeManager tree: {count_total_nodes(tree)} nodes, depth {get_tree_depth(tree)}")
    print(f"   Manual tree: {count_total_nodes(manual_tree)} nodes, depth {get_tree_depth(manual_tree)}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
