from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from app.utils.diagnosis_tree import DiagnosisTreeNode
from typing import Any, Optional
from app.services.llm_service import LLMService
import asyncio
from app.agents.base_agent import BaseAgent
from app.core.interfaces import IWebSocketManager
from app.utils.message_types import MessageSource

class TreeManagerAgent(BaseAgent):
    """
    Manages the diagnosis tree structure and symptom assignment.
    """
    def __init__(
        self,
        root: DiagnosisTreeNode,
        llm_service: Optional[LLMService] = None,
        websocket_manager: Optional[IWebSocketManager] = None,
        **kwargs
    ):
        """
        Initialize the TreeManagerAgent with dependency injection for testability.
        """
        super().__init__(websocket_manager=websocket_manager, **kwargs)
        self.root = root
        self.llm_service = llm_service or LLMService()
        self.prompt = PromptTemplate(
            input_variables=["symptom", "tree_state"],
            template="""
                Given the current diagnosis tree:
                {tree_state}
                And a new symptom: {symptom}
                Decide under which node (by issue_name) this symptom should be added as a child. Reply with the exact issue_name or 'root' if it should be a direct child of the root.
            """
        )

    def get_tree_state(self) -> str:
        """
        Get the current state of the diagnosis tree as a string.
        """
        def node_repr(node, depth=0):
            s = f"{'  '*depth}- {node.issue_name} (likelyhood={node.likelyhood:.2f})"
            for child in node.children:
                s += "\n" + node_repr(child, depth+1)
            return s
        return node_repr(self.root)

    async def decide_parent_for_symptom(self, symptom: str) -> DiagnosisTreeNode:
        """
        Decide the best parent node for a symptom using a fast heuristic approach.
        Falls back to LLM only for complex decisions to improve performance.
        """
        # Fast heuristic: if tree is empty or has fewer than 3 children, add to root
        if len(self.root.children) < 3:
            await self.logger.info(f"TreeManager: Using root as parent for '{symptom}' (tree has < 3 children)")
            return self.root
        
        # Fast heuristic: find highest likelihood parent that might be related
        best_parent = self.root
        highest_likelihood = 0.0
        
        # Check if any existing symptoms might be related (simple keyword matching)
        symptom_lower = symptom.lower()
        for child in self.root.children:
            child_name_lower = child.issue_name.lower()
            
            # Simple keyword matching for common automotive terms
            common_keywords = ['sensor', 'engine', 'electrical', 'brake', 'transmission', 'fuel', 'ignition', 'speed', 'cluster', 'wiring']
            shared_keywords = [kw for kw in common_keywords if kw in symptom_lower and kw in child_name_lower]
            
            if shared_keywords and child.likelyhood > highest_likelihood:
                best_parent = child
                highest_likelihood = child.likelyhood
                await self.logger.info(f"TreeManager: Found related parent '{child.issue_name}' for '{symptom}' (shared keywords: {shared_keywords})")
        
        # If no good heuristic match and tree is complex, use LLM (but only occasionally)
        if best_parent == self.root and len(self.root.children) > 5:
            tree_state = self.get_tree_state()
            await self.logger.info(f"TreeManager: Current tree state when deciding parent for '{symptom}':\n{tree_state}")
            
            prompt = self.prompt.format(symptom=symptom, tree_state=tree_state)
            response = await self.llm_service.generate_response(prompt)
            
            # Handle AIMessage or other objects with content attribute (from LangChain)
            if hasattr(response, 'content'):
                parent_name = response.content.strip()
            else:
                parent_name = str(response).strip()
            
            await self.logger.info(f"TreeManager: LLM suggested parent '{parent_name}' for symptom '{symptom}'")
            
            if parent_name.lower() == 'root':
                await self.logger.info(f"TreeManager: Using root as parent for '{symptom}'")
                return self.root
            
            node = self.root.find(parent_name)
            if node:
                await self.logger.info(f"TreeManager: Found parent node '{parent_name}' for symptom '{symptom}'")
                return node
            else:
                await self.logger.info(f"TreeManager: Could not find parent node '{parent_name}', using heuristic result for symptom '{symptom}'")
                return best_parent
        
        await self.logger.info(f"TreeManager: Using heuristic parent '{best_parent.issue_name}' for symptom '{symptom}'")
        return best_parent

    async def add_symptom(self, symptom: str, likelyhood: float, data: Any = None, websocket=None, session_id=None):
        """
        Add a new symptom to the diagnosis tree under the decided parent node.
        """
        await self.logger.info(f"TreeManager: Starting to add symptom '{symptom}' with likelihood {likelyhood}")
        
        if websocket:
            await self.send_ws_stage(websocket, f"Adding symptom '{symptom}' to diagnosis tree", MessageSource.TREE_MANAGER, session_id=session_id)
        
        # Debug logging
        initial_children_count = len(self.root.children)
        await self.logger.info(f"TreeManager: Adding symptom '{symptom}' with likelihood {likelyhood}. Current tree children: {initial_children_count}")
        
        if websocket:
            tree_state_summary = {
                "current_children": initial_children_count,
                "symptom_being_added": symptom,
                "likelihood": round(likelyhood * 100, 1)
            }
            await self.send_ws_stage(
                websocket, 
                f"Analyzing tree structure for symptom placement", 
                MessageSource.TREE_MANAGER, 
                session_id=session_id,
                details={"tree_analysis": tree_state_summary}
            )
        
        parent_node = await self.decide_parent_for_symptom(symptom)
        await self.logger.info(f"TreeManager: Selected parent node: {parent_node.issue_name}")
        
        if websocket:
            await self.send_ws_stage(
                websocket, 
                f"Placing symptom under: {parent_node.issue_name}", 
                MessageSource.TREE_MANAGER, 
                session_id=session_id
            )
        
        new_node = DiagnosisTreeNode(issue_name=symptom, likelyhood=likelyhood, data=data)
        parent_node.add_child(new_node)
        
        final_children_count = len(self.root.children)
        await self.logger.info(f"TreeManager: Added symptom '{symptom}'. Tree children count: {initial_children_count} -> {final_children_count}")
        
        if websocket:
            tree_update_summary = {
                "symptom_added": symptom,
                "parent_node": parent_node.issue_name,
                "likelihood": round(likelyhood * 100, 1),
                "new_tree_size": final_children_count,
                "tree_growth": final_children_count - initial_children_count
            }
            await self.send_ws_result(
                websocket, 
                f"Symptom '{symptom}' successfully added to tree", 
                MessageSource.TREE_MANAGER, 
                session_id=session_id, 
                details=tree_update_summary
            )
        
        await self.logger.info(f"TreeManager: Successfully completed adding symptom '{symptom}'")
        return new_node

    def prune_tree(self, threshold: float = 0.3):
        """
        Prune the tree by removing nodes that have a likelihood below the given threshold.
        """
        # Note: This is a sync method called from async context, so we cannot use async logger
        # The caller should log this operation instead
        initial_count = len(self.root.children)
        self.root.prune(threshold)
        final_count = len(self.root.children)
        # Store pruning stats for caller to log
        self._last_prune_stats = {
            "threshold": threshold,
            "initial_count": initial_count,
            "final_count": final_count
        }

    def sort_tree(self):
        """
        Sort the children of each node in the tree by likelihood in descending order.
        """
        for node in self.root.traverse():
            node.sort_children_by_likelyhood()

    async def process(self, *args, **kwargs) -> None:
        """
        Dummy process method to satisfy BaseAgent's abstract method requirement.
        TreeManagerAgent does not use a process entry point.
        """
        pass

    def close(self) -> None:
        """
        Optional cleanup method for the agent.
        """
        pass
