"""
Orchestrator-specific WebSocket communication extensions
"""
from typing import Dict, Any, Optional
from app.utils.tree_formatter import TreeDataFormatter
from app.utils.diagnosis_tree import DiagnosisTreeNode
from app.utils.message_types import MessageSource
from app.agents.base_agent import BaseAgent


class OrchestratorWebSocketMixin:
    """
    Mixin class providing standardized WebSocket methods specifically for the orchestrator agent
    """
    
    async def send_orchestrator_stage(self, websocket, content: str, session_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Send orchestrator-specific stage update"""
        orchestrator_details = {
            "agent": "orchestrator",
            "classification": "stage_update"
        }
        if details:
            orchestrator_details.update(details)
        
        await self.send_ws_stage(websocket, content, MessageSource.ORCHESTRATOR, session_id, orchestrator_details)
    
    async def send_classification_result(self, websocket, classification: Dict[str, Any], session_id: Optional[str] = None) -> None:
        """Send message classification result"""
        content = f"Message classified as: {classification.get('response_type', 'unknown')}"
        details = {
            "classification": classification,
            "agent": "orchestrator",
            "stage": "message_classification"
        }
        await self.send_ws_info(websocket, content, MessageSource.ORCHESTRATOR, session_id, details)
    
    async def send_tree_initialization(self, websocket, tree: DiagnosisTreeNode, session_id: Optional[str] = None) -> None:
        """Send tree initialization data"""
        tree_data = TreeDataFormatter.format_tree_data(tree, stage="tree_initialization")
        content = f"Diagnosis tree initialized with {tree_data['metadata']['total_symptoms']} symptoms"
        
        await self.send_ws_tree_data(
            websocket, 
            content, 
            MessageSource.ORCHESTRATOR, 
            tree_data, 
            session_id, 
            stage="tree_initialization"
        )
    
    async def send_tree_update(self, websocket, tree: DiagnosisTreeNode, stage: str, session_id: Optional[str] = None, additional_details: Optional[Dict[str, Any]] = None) -> None:
        """Send standardized tree update"""
        tree_data = TreeDataFormatter.format_tree_data(tree, stage=stage)
        tree_summary = TreeDataFormatter.format_tree_summary(tree)
        
        content = f"Tree updated: {tree_summary['total_symptoms']} symptoms ({tree_summary['high_likelihood_count']} high priority)"
        
        details = {
            "tree_summary": tree_summary,
            "agent": "orchestrator",
            "update_type": stage
        }
        if additional_details:
            details.update(additional_details)
        
        await self.send_ws_tree_data(
            websocket, 
            content, 
            MessageSource.ORCHESTRATOR, 
            tree_data, 
            session_id, 
            stage=stage, 
            details=details
        )
    
    async def send_symptom_extraction_complete(self, websocket, tree: DiagnosisTreeNode, new_symptoms: int, session_id: Optional[str] = None) -> None:
        """Send symptom extraction completion with tree data"""
        tree_data = TreeDataFormatter.format_tree_data(tree, stage="symptom_extraction_complete")
        content = f"Symptom extraction complete: {new_symptoms} new symptoms added"
        
        details = {
            "new_symptoms_count": new_symptoms,
            "agent": "orchestrator",
            "stage": "symptom_extraction_complete"
        }
        
        await self.send_ws_tree_data(
            websocket, 
            content, 
            MessageSource.ORCHESTRATOR, 
            tree_data, 
            session_id, 
            stage="symptom_extraction_complete", 
            details=details
        )
    
    async def send_diagnosis_preparation(self, websocket, tree: DiagnosisTreeNode, session_id: Optional[str] = None) -> None:
        """Send pre-diagnosis tree state"""
        tree_data = TreeDataFormatter.format_tree_data(tree, stage="diagnosis_preparation")
        tree_summary = TreeDataFormatter.format_tree_summary(tree)
        
        content = f"Preparing diagnosis with {tree_summary['total_symptoms']} symptoms"
        
        details = {
            "tree_summary": tree_summary,
            "agent": "orchestrator",
            "stage": "diagnosis_preparation",
            "analysis_context": {
                "symptoms_available": tree_summary['total_symptoms'] > 0,
                "high_priority_symptoms": tree_summary['high_likelihood_count'],
                "ready_for_diagnosis": tree_summary['total_symptoms'] > 0
            }
        }
        
        await self.send_ws_tree_data(
            websocket, 
            content, 
            MessageSource.ORCHESTRATOR, 
            tree_data, 
            session_id, 
            stage="diagnosis_preparation", 
            details=details
        )
    
    async def send_diagnosis_complete(self, websocket, tree: DiagnosisTreeNode, diagnosis_result: Dict[str, Any], session_id: Optional[str] = None) -> None:
        """Send diagnosis completion with final tree state"""
        tree_data = TreeDataFormatter.format_tree_data(tree, stage="diagnosis_complete")
        tree_summary = TreeDataFormatter.format_tree_summary(tree)
        
        content = f"Diagnosis complete with {tree_summary['total_symptoms']} analyzed symptoms"
        
        details = {
            "tree_summary": tree_summary,
            "diagnosis_success": diagnosis_result.get("success", False),
            "agent": "orchestrator",
            "stage": "diagnosis_complete",
            "final_analysis": {
                "total_symptoms_analyzed": tree_summary['total_symptoms'],
                "top_symptoms": tree_summary['top_symptoms'],
                "likelihood_distribution": tree_data['metadata']['likelihood_distribution']
            }
        }
        
        await self.send_ws_tree_data(
            websocket, 
            content, 
            MessageSource.ORCHESTRATOR, 
            tree_data, 
            session_id, 
            stage="diagnosis_complete", 
            details=details
        )
    
    async def send_error_with_tree_state(self, websocket, error_message: str, tree: Optional[DiagnosisTreeNode], session_id: Optional[str] = None) -> None:
        """Send error message with current tree state for context"""
        details = {
            "agent": "orchestrator",
            "error_context": "processing_error"
        }
        
        if tree:
            tree_summary = TreeDataFormatter.format_tree_summary(tree)
            details["tree_state"] = {
                "symptoms_count": tree_summary['total_symptoms'],
                "tree_available": True
            }
        else:
            details["tree_state"] = {
                "symptoms_count": 0,
                "tree_available": False
            }
        
        await self.send_ws_error(websocket, error_message, MessageSource.ORCHESTRATOR, session_id, details)
