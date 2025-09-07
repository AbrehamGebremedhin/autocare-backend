"""
Tree data formatting utilities for standardized WebSocket communication
"""
from typing import Dict, Any, Optional, List
from app.utils.diagnosis_tree import DiagnosisTreeNode


class TreeDataFormatter:
    """Utility class for formatting diagnosis tree data for WebSocket communication"""
    
    @staticmethod
    def format_tree_data(tree: DiagnosisTreeNode, stage: str = "tree_update", include_metadata: bool = True) -> Dict[str, Any]:
        """
        Format diagnosis tree data into a standardized structure for WebSocket communication
        
        Args:
            tree: The DiagnosisTreeNode to format (or dict to convert)
            stage: The current stage of tree processing
            include_metadata: Whether to include metadata about the tree
            
        Returns:
            Standardized tree data dictionary
        """
        # Handle None trees
        if not tree:
            return {
                "tree_structure": None,
                "metadata": {
                    "total_nodes": 0,
                    "total_symptoms": 0,
                    "root_issue": None,
                    "stage": stage,
                    "is_empty": True
                } if include_metadata else {}
            }
        
        # Handle dict trees (convert to DiagnosisTreeNode first)
        if isinstance(tree, dict):
            try:
                from app.schemas.Chat_Session import ChatSession
                tree = ChatSession.deserialize_diagnosis_tree(tree)
            except Exception as e:
                # If conversion fails, return basic structure
                return {
                    "tree_structure": tree,
                    "metadata": {
                        "total_nodes": 1,
                        "total_symptoms": len(tree.get('children', [])),
                        "root_issue": tree.get('issue_name', 'unknown'),
                        "stage": stage,
                        "is_empty": len(tree.get('children', [])) == 0,
                        "conversion_error": str(e)
                    } if include_metadata else {}
                }
        
        # Verify tree has children attribute
        if not hasattr(tree, 'children'):
            return {
                "tree_structure": {"issue_name": getattr(tree, 'issue_name', 'unknown'), "children": []},
                "metadata": {
                    "total_nodes": 1,
                    "total_symptoms": 0,
                    "root_issue": getattr(tree, 'issue_name', 'unknown'),
                    "stage": stage,
                    "is_empty": True,
                    "missing_children_attr": True
                } if include_metadata else {}
            }
        
        # Format the tree structure
        tree_structure = TreeDataFormatter._format_node_recursive(tree)
        
        # Calculate metadata
        metadata = {}
        if include_metadata:
            metadata = {
                "total_nodes": TreeDataFormatter._count_nodes(tree),
                "total_symptoms": len(tree.children),
                "root_issue": tree.issue_name,
                "stage": stage,
                "is_empty": len(tree.children) == 0,
                "symptom_categories": TreeDataFormatter._get_symptom_categories(tree),
                "likelihood_distribution": TreeDataFormatter._get_likelihood_distribution(tree)
            }
        
        return {
            "tree_structure": tree_structure,
            "metadata": metadata
        }
    
    @staticmethod
    def format_tree_summary(tree: DiagnosisTreeNode) -> Dict[str, Any]:
        """
        Format a condensed summary of the tree for quick updates
        
        Args:
            tree: The DiagnosisTreeNode to summarize
            
        Returns:
            Tree summary dictionary
        """
        if not tree:
            return {
                "total_symptoms": 0,
                "high_likelihood_count": 0,
                "medium_likelihood_count": 0,
                "low_likelihood_count": 0,
                "top_symptoms": []
            }
        
        children = getattr(tree, 'children', [])
        high_likelihood = [child for child in children if getattr(child, 'likelyhood', 0) > 0.7]
        medium_likelihood = [child for child in children if 0.3 <= getattr(child, 'likelyhood', 0) <= 0.7]
        low_likelihood = [child for child in children if getattr(child, 'likelyhood', 0) < 0.3]
        
        # Get top 3 symptoms by likelihood
        sorted_symptoms = sorted(children, key=lambda x: getattr(x, 'likelyhood', 0), reverse=True)
        top_symptoms = [
            {
                "issue_name": symptom.issue_name,
                "likelihood": round(getattr(symptom, 'likelyhood', 0) * 100, 1),
                "category": symptom.data.get("issue_category") if symptom.data else "Unknown"
            }
            for symptom in sorted_symptoms[:3]
        ]
        
        return {
            "total_symptoms": len(children),
            "high_likelihood_count": len(high_likelihood),
            "medium_likelihood_count": len(medium_likelihood),
            "low_likelihood_count": len(low_likelihood),
            "top_symptoms": top_symptoms
        }
    
    @staticmethod
    def _format_node_recursive(node: DiagnosisTreeNode, depth: int = 0) -> Dict[str, Any]:
        """Recursively format a node and its children"""
        children = getattr(node, 'children', [])
        
        formatted_node = {
            "issue_name": node.issue_name,
            "likelihood": round(getattr(node, 'likelyhood', 0) * 100, 1),
            "depth": depth,
            "data": getattr(node, 'data', None),
            "children_count": len(children),
            "children": []
        }
        
        # Sort children by likelihood (highest first)
        sorted_children = sorted(children, key=lambda x: getattr(x, 'likelyhood', 0), reverse=True)
        
        for child in sorted_children:
            formatted_child = TreeDataFormatter._format_node_recursive(child, depth + 1)
            formatted_node["children"].append(formatted_child)
        
        return formatted_node
    
    @staticmethod
    def _count_nodes(node: DiagnosisTreeNode) -> int:
        """Count total nodes in the tree"""
        count = 1  # Count this node
        children = getattr(node, 'children', [])
        for child in children:
            count += TreeDataFormatter._count_nodes(child)
        return count
    
    @staticmethod
    def _get_symptom_categories(tree: DiagnosisTreeNode) -> List[str]:
        """Get unique categories from all symptoms"""
        categories = set()
        children = getattr(tree, 'children', [])
        
        for child in children:
            if hasattr(child, 'data') and child.data and isinstance(child.data, dict):
                category = child.data.get('issue_category') or child.data.get('category')
                if category:
                    categories.add(category)
        
        return sorted(list(categories))
    
    @staticmethod
    def _get_likelihood_distribution(tree: DiagnosisTreeNode) -> Dict[str, int]:
        """Get distribution of symptoms by likelihood ranges"""
        children = getattr(tree, 'children', [])
        distribution = {
            "high": 0,      # > 70%
            "medium": 0,    # 30-70%
            "low": 0        # < 30%
        }
        
        for child in children:
            likelihood = getattr(child, 'likelyhood', 0)
            if likelihood > 0.7:
                distribution["high"] += 1
            elif likelihood >= 0.3:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
