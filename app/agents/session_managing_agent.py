from app.utils.diagnosis_tree import AbstractTreeNode
from typing import Any, Dict, Optional, List
import asyncio
from langchain_core.language_models.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.llm import LLMChain

class DiagnosisTreeAgent:
    """
    Unified LangChain agent for session-wide diagnosis tree management using LLMs.
    - Expands any node (or all nodes recursively) using the LLM
    - Accepts lists of TreeNodes for batch update/prune
    - Prunes, sorts, updates, and resets the tree
    - Thread-safe for async use
    """
    def __init__(self, llm: BaseLanguageModel, prompt: PromptTemplate, root_issue_name: str = "root", root_likelyhood: float = 1.0):
        self.lock = asyncio.Lock()
        self.root = self._create_root(root_issue_name, root_likelyhood)
        self.node_map: Dict[str, AbstractTreeNode] = {self.root.issue_name: self.root}
        self.llm = llm
        self.prompt = prompt
        self.output_parser = JsonOutputParser()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _create_root(self, issue_name: str, likelyhood: float) -> AbstractTreeNode:
        class RootNode(AbstractTreeNode):
            def process(self):
                pass
        return RootNode(issue_name, likelyhood)

    async def get_tree(self) -> AbstractTreeNode:
        async with self.lock:
            return self.root

    async def expand_node_with_llm(self, node_name: str, context: str, symptom_text: str) -> List[AbstractTreeNode]:
        async with self.lock:
            parent = self.node_map.get(node_name)
            if not parent:
                raise ValueError(f"Node '{node_name}' not found.")
            llm_input = {
                "parent_issue": node_name,
                "context": context,
                "symptom_text": symptom_text
            }
            issues = self.chain.invoke(llm_input)
            try:
                parsed_issues = self.output_parser.parse(issues)
            except Exception:
                parsed_issues = []
            new_children = []
            for issue in parsed_issues:
                name = issue.get('issue_name')
                likelyhood = issue.get('likelihood', 0)
                data = issue.get('data', None)
                node = self.node_map.get(name)
                if node:
                    node.likelyhood = likelyhood
                    node.update_data(data)
                    if node.parent != parent:
                        if node.parent:
                            node.parent.remove_child(node)
                        parent.add_child(node)
                else:
                    class IssueNode(AbstractTreeNode):
                        def process(self):
                            pass
                    new_node = IssueNode(name, likelyhood, data)
                    parent.add_child(new_node)
                    self.node_map[name] = new_node
                    new_children.append(new_node)
            parent.sort_children_by_likelyhood()
            return [self.node_map[issue.get('issue_name')] for issue in parsed_issues if issue.get('issue_name') in self.node_map]

    async def expand_all_nodes_with_llm(self, context: str, symptom_text: str):
        async with self.lock:
            async def expand_node(node: AbstractTreeNode):
                llm_input = {
                    "parent_issue": node.issue_name,
                    "context": context,
                    "symptom_text": symptom_text
                }
                issues = self.chain.invoke(llm_input)
                try:
                    parsed_issues = self.output_parser.parse(issues)
                except Exception:
                    parsed_issues = []
                for issue in parsed_issues:
                    name = issue.get('issue_name')
                    likelyhood = issue.get('likelihood', 0)
                    data = issue.get('data', None)
                    child = self.node_map.get(name)
                    if child:
                        child.likelyhood = likelyhood
                        child.update_data(data)
                        if child.parent != node:
                            if child.parent:
                                child.parent.remove_child(child)
                            node.add_child(child)
                    else:
                        class IssueNode(AbstractTreeNode):
                            def process(self):
                                pass
                        new_node = IssueNode(name, likelyhood, data)
                        node.add_child(new_node)
                        self.node_map[name] = new_node
                node.sort_children_by_likelyhood()
                for child in node.children:
                    await expand_node(child)
            await expand_node(self.root)

    async def update_tree_from_nodes(self, nodes: List[AbstractTreeNode], prune_threshold: float = 0.3):
        async with self.lock:
            for node in nodes:
                existing = self.node_map.get(node.issue_name)
                if existing:
                    existing.likelyhood = node.likelyhood
                    existing.update_data(node.data)
                else:
                    parent = None
                    if node.parent and node.parent.issue_name in self.node_map:
                        parent = self.node_map[node.parent.issue_name]
                    elif node.parent:
                        parent = next((n for n in nodes if n.issue_name == node.parent.issue_name), None)
                    if parent:
                        parent.add_child(node)
                        node.parent = parent
                    else:
                        self.root.add_child(node)
                        node.parent = self.root
                    self.node_map[node.issue_name] = node
            self.root.prune(prune_threshold)
            self.node_map = {n.issue_name: n for n in self.root.traverse()}

    async def update_issue(self, issue_name: str, data: Any):
        async with self.lock:
            node = self.node_map.get(issue_name)
            if not node:
                raise ValueError(f"Issue '{issue_name}' not found.")
            node.update_data(data)

    async def prune_tree(self, threshold: float = 0.3):
        async with self.lock:
            self.root.prune(threshold)
            self.node_map = {n.issue_name: n for n in self.root.traverse()}

    async def find_issue(self, issue_name: str) -> Optional[AbstractTreeNode]:
        async with self.lock:
            return self.node_map.get(issue_name)

    async def sort_children(self, issue_name: str, reverse: bool = True):
        async with self.lock:
            node = self.node_map.get(issue_name)
            if node:
                node.sort_children_by_likelyhood(reverse=reverse)

    async def reset(self):
        async with self.lock:
            self.root = self._create_root(self.root.issue_name, self.root.likelyhood)
            self.node_map = {self.root.issue_name: self.root}
