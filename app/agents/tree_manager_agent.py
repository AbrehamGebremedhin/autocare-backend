from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from app.utils.diagnosis_tree import DiagnosisTreeNode
from typing import Any
from app.services.llm_service import LLMService

class TreeManagerAgent:
    def __init__(self, root: DiagnosisTreeNode, llm_service: LLMService = None):
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
        def node_repr(node, depth=0):
            s = f"{'  '*depth}- {node.issue_name} (likelyhood={node.likelyhood:.2f})"
            for child in node.children:
                s += "\n" + node_repr(child, depth+1)
            return s
        return node_repr(self.root)

    def decide_parent_for_symptom(self, symptom: str) -> DiagnosisTreeNode:
        import asyncio
        tree_state = self.get_tree_state()
        prompt = self.prompt.format(symptom=symptom, tree_state=tree_state)
        response = asyncio.run(self.llm_service.generate_response(prompt))
        parent_name = response.strip()
        if parent_name.lower() == 'root':
            return self.root
        node = self.root.find(parent_name)
        return node if node else self.root

    def add_symptom(self, symptom: str, likelyhood: float, data: Any = None):
        parent_node = self.decide_parent_for_symptom(symptom)
        new_node = DiagnosisTreeNode(issue_name=symptom, likelyhood=likelyhood, data=data)
        parent_node.add_child(new_node)
        return new_node

    def prune_tree(self, threshold: float = 0.3):
        self.root.prune(threshold)

    def sort_tree(self):
        for node in self.root.traverse():
            node.sort_children_by_likelyhood()
