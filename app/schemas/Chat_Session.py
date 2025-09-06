from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.utils.diagnosis_tree import DiagnosisTreeNode

class ChatSession(BaseModel):
    id: Optional[str] = Field(default=None, primary_key=True, unique=True)
    user_id: str
    title: Optional[str] = Field(default="New Chat Session", max_length=100)  # Short title for the session
    messages: List[Dict[str, Any]] = []  # Each message can be a dict with role, content, timestamp, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    context: Optional[Dict[str, Any]] = None
    diagnosis_tree: Any  # Store the DiagnosisTreeNode or its serializable representation

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True

    @staticmethod
    def serialize_diagnosis_tree(tree: DiagnosisTreeNode) -> dict:
        """Recursively serialize a DiagnosisTreeNode to a dict."""
        return {
            'issue_name': tree.issue_name,
            'likelyhood': tree.likelyhood,
            'data': tree.data,
            'children': [ChatSession.serialize_diagnosis_tree(child) for child in tree.children]
        }

    @staticmethod
    def deserialize_diagnosis_tree(data: dict) -> DiagnosisTreeNode:
        """Recursively deserialize a dict to a DiagnosisTreeNode."""
        node = DiagnosisTreeNode(
            issue_name=data['issue_name'],
            likelyhood=data['likelyhood'],
            data=data.get('data')
        )
        for child_data in data.get('children', []):
            child_node = ChatSession.deserialize_diagnosis_tree(child_data)
            node.add_child(child_node)
        return node

    @staticmethod
    def generate_session_title(diagnosis_tree: DiagnosisTreeNode, messages: List[Dict[str, Any]] = None) -> str:
        """
        Generate a short descriptive title for the session based on the diagnosis tree and messages.
        """
        if not diagnosis_tree or not hasattr(diagnosis_tree, 'children'):
            return "New Chat Session"
        
        # If no symptoms have been identified yet, use initial user message if available
        if not diagnosis_tree.children:
            if messages and len(messages) > 0:
                # Look for the first user message
                first_user_msg = next((msg for msg in messages if msg.get('role') == 'user'), None)
                if first_user_msg and first_user_msg.get('content'):
                    # Extract key words from the first message and create a short title
                    content = first_user_msg['content'][:50]  # First 50 chars
                    if len(first_user_msg['content']) > 50:
                        content += "..."
                    return content
            return "New Chat Session"
        
        # Find the highest likelihood symptoms/issues from the diagnosis tree
        high_likelihood_issues = []
        medium_likelihood_issues = []
        
        def extract_symptoms(node, depth=0):
            """Extract symptoms with likelihood above certain thresholds"""
            if node.issue_name != 'root' and depth <= 2:  # Don't go too deep
                likelihood = getattr(node, 'likelyhood', 0.0)
                if likelihood > 0.7:
                    high_likelihood_issues.append(node.issue_name)
                elif likelihood > 0.4:
                    medium_likelihood_issues.append(node.issue_name)
            
            # Process children
            for child in getattr(node, 'children', []):
                extract_symptoms(child, depth + 1)
        
        extract_symptoms(diagnosis_tree)
        
        # Build title based on symptoms found
        if high_likelihood_issues:
            # Use the most likely issues
            main_issues = high_likelihood_issues[:2]  # Take top 2
            title = " & ".join(main_issues)
        elif medium_likelihood_issues:
            # Use medium likelihood issues if no high likelihood ones
            main_issues = medium_likelihood_issues[:2]  # Take top 2
            title = " & ".join(main_issues)
        else:
            # Fallback to any child issues
            all_children = [child.issue_name for child in diagnosis_tree.children[:2]]
            title = " & ".join(all_children) if all_children else "Car Diagnosis"
        
        # Ensure title is not too long
        if len(title) > 50:
            title = title[:47] + "..."
        
        return title if title else "Car Diagnosis"
