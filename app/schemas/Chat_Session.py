from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.utils.diagnosis_tree import DiagnosisTreeNode

class ChatSession(BaseModel):
    id: Optional[str] = Field(default=None, primary_key=True, unique=True)
    user_id: str
    messages: List[Dict[str, Any]] = []  # Each message can be a dict with role, content, timestamp, etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    context: Optional[Dict[str, Any]] = None
    diagnosis_tree: Any  # Store the DiagnosisTreeNode or its serializable representation

    class Config:
        orm_mode = True
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
