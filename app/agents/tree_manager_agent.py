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
        Decide the parent node for a new symptom based on the current tree state.
        """
        tree_state = self.get_tree_state()
        await self.logger.info(f"TreeManager: Current tree state when deciding parent for '{symptom}':\n{tree_state}")
        
        prompt = self.prompt.format(symptom=symptom, tree_state=tree_state)
        response = await self.llm_service.generate_response(prompt)
        parent_name = response.strip()
        
        await self.logger.info(f"TreeManager: LLM suggested parent '{parent_name}' for symptom '{symptom}'")
        
        if parent_name.lower() == 'root':
            await self.logger.info(f"TreeManager: Using root as parent for '{symptom}'")
            return self.root
        
        node = self.root.find(parent_name)
        if node:
            await self.logger.info(f"TreeManager: Found parent node '{parent_name}' for symptom '{symptom}'")
            return node
        else:
            await self.logger.info(f"TreeManager: Could not find parent node '{parent_name}', defaulting to root for symptom '{symptom}'")
            return self.root

    async def add_symptom(self, symptom: str, likelyhood: float, data: Any = None, websocket=None, session_id=None):
        """
        Add a new symptom to the diagnosis tree under the decided parent node.
        """
        await self.logger.info(f"TreeManager: Starting to add symptom '{symptom}' with likelihood {likelyhood}")
        
        if websocket:
            await self.send_ws_stage(websocket, f"Adding symptom '{symptom}' to tree", MessageSource.ORCHESTRATOR, session_id=session_id)
        
        # Debug logging
        initial_children_count = len(self.root.children)
        await self.logger.info(f"TreeManager: Adding symptom '{symptom}' with likelihood {likelyhood}. Current tree children: {initial_children_count}")
        
        parent_node = await self.decide_parent_for_symptom(symptom)
        await self.logger.info(f"TreeManager: Selected parent node: {parent_node.issue_name}")
        
        new_node = DiagnosisTreeNode(issue_name=symptom, likelyhood=likelyhood, data=data)
        parent_node.add_child(new_node)
        
        final_children_count = len(self.root.children)
        await self.logger.info(f"TreeManager: Added symptom '{symptom}'. Tree children count: {initial_children_count} -> {final_children_count}")
        
        if websocket:
            await self.send_ws_result(websocket, f"Symptom '{symptom}' added", MessageSource.ORCHESTRATOR, session_id=session_id, details={"symptom": symptom, "likelyhood": likelyhood})
        
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
