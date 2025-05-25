import asyncio
from typing import Any, List, Optional, Dict
from langchain_core.language_models.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.llm import LLMChain
from app.utils.logger import Logger

class DiagnosisTreeAgent:
    """
    Unified LangChain agent for managing a diagnosis tree with optional LLM-based expansion.
    - Maintains a tree of issues, each with a name, likelihood, and associated data.
    - Supports batch updates: incorporate new issues and update existing ones, inferring parent-child relationships.
    - Prunes low-likelihood issues and sorts children by likelihood.
    - Can expand any node (or the entire tree) using an LLMChain based on context and symptom text.
    - Thread-safe operations using an asyncio lock for concurrent usage.
    """
    def __init__(self, llm: BaseLanguageModel, prompt: PromptTemplate,
                 root_issue_name: str = "root", root_likelihood: float = 1.0,
                 root: Optional['DiagnosisTreeAgent.TreeNode'] = None,
                 logger: Optional[Logger] = None):
        self.lock = asyncio.Lock()
        if root is not None:
            self.root = root
            # Build node_map from the provided root's subtree
            self.node_map: Dict[str, DiagnosisTreeAgent.TreeNode] = {n.issue_name: n for n in root.traverse()}
        else:
            # Create the root node of the tree
            self.root = self._create_root(root_issue_name, root_likelihood)
            self.node_map: Dict[str, DiagnosisTreeAgent.TreeNode] = {self.root.issue_name: self.root}
        # LLM chain for expansions
        self.llm = llm
        self.prompt = prompt
        self.output_parser = JsonOutputParser()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.logger = logger or Logger()

    class TreeNode:
        """
        Internal class representing a node in the diagnosis tree.
        """
        def __init__(self, issue_name: str, likelihood: float, data: Any = None,
                     parent: Optional['DiagnosisTreeAgent.TreeNode'] = None):
            self.issue_name = issue_name
            self.likelihood = likelihood
            self.data = data
            self.children: List['DiagnosisTreeAgent.TreeNode'] = []
            self.parent: Optional['DiagnosisTreeAgent.TreeNode'] = parent

        def add_child(self, child: 'DiagnosisTreeAgent.TreeNode'):
            """
            Add a child node to this node.
            """
            child.parent = self
            self.children.append(child)

        def remove_child(self, child: 'DiagnosisTreeAgent.TreeNode'):
            """
            Remove a child node from this node.
            """
            self.children.remove(child)
            child.parent = None

        def find(self, issue_name: str) -> Optional['DiagnosisTreeAgent.TreeNode']:
            """
            Find a node by issue name in the subtree rooted at this node.
            """
            if self.issue_name == issue_name:
                return self
            for child in self.children:
                result = child.find(issue_name)
                if result:
                    return result
            return None

        def traverse(self) -> List['DiagnosisTreeAgent.TreeNode']:
            """
            Traverse the tree (pre-order) and return all nodes in a list.
            """
            nodes = [self]
            for child in self.children:
                nodes.extend(child.traverse())
            return nodes

        def update_data(self, data: Any):
            """
            Update the data associated with this node.
            """
            self.data = data

        def prune(self, threshold: float = 0.3):
            """
            Prune children whose likelihood is below the threshold (e.g., 0.3 for 30%).
            """
            pruned_children = []
            for child in self.children:
                if child.likelihood < threshold:
                    pruned_children.append(child)
                else:
                    child.prune(threshold)
            for child in pruned_children:
                self.children.remove(child)

        def sort_children_by_likelihood(self, reverse: bool = True):
            """
            Sort children nodes by their likelihood (descending by default).
            """
            self.children.sort(key=lambda x: x.likelihood, reverse=reverse)

        def process(self):
            """
            Placeholder method; override with processing logic if needed.
            """
            pass

    def _create_root(self, issue_name: str, likelihood: float) -> 'DiagnosisTreeAgent.TreeNode':
        """
        Create a root node (subclass of TreeNode) with the given name and likelihood.
        """
        root = DiagnosisTreeAgent.TreeNode(issue_name, likelihood)
        return root

    async def get_tree(self) -> TreeNode:
        """
        Get the root of the diagnosis tree.
        """
        async with self.lock:
            return self.root

    async def find_issue(self, issue_name: str) -> Optional[TreeNode]:
        """
        Find and return the node with the given issue_name, or None if not found.
        """
        async with self.lock:
            return self.node_map.get(issue_name)

    async def update_tree_from_nodes(self, nodes: List[TreeNode], prune_threshold: float = 0.3):
        """
        Batch update the tree with a list of nodes:
        - If a node exists, update its data and likelihood.
        - Otherwise, attach the new node to the tree (using parent reference if available, or as child of root).
        - After insertion, prune low-likelihood nodes and sort children by likelihood.
        """
        async with self.lock:
            for node in nodes:
                existing = self.node_map.get(node.issue_name)
                if existing:
                    # Update existing node's likelihood and data
                    existing.likelihood = node.likelihood
                    existing.update_data(node.data)
                    # If a new parent is provided and is different, re-attach node
                    if node.parent:
                        parent_name = node.parent.issue_name
                        parent = self.node_map.get(parent_name)
                        if parent and parent != existing.parent:
                            # Remove from old parent, add to new parent
                            if existing.parent:
                                existing.parent.remove_child(existing)
                            parent.add_child(existing)
                            existing.parent = parent
                else:
                    # Node does not exist; determine parent
                    parent = None
                    if node.parent:
                        # Check if parent already in current tree
                        parent = self.node_map.get(node.parent.issue_name)
                        if not parent:
                            # Parent is also new in this batch? Find the actual node object for parent
                            parent = next((n for n in nodes if n.issue_name == node.parent.issue_name), None)
                    if parent:
                        parent.add_child(node)
                        node.parent = parent
                    else:
                        # No parent specified or not found; attach to root
                        self.root.add_child(node)
                        node.parent = self.root
                    # Add this new node to the map
                    self.node_map[node.issue_name] = node
            # Prune low-likelihood nodes from the tree
            self.root.prune(prune_threshold)
            # Sort children by likelihood at each node
            for n in self.root.traverse():
                n.sort_children_by_likelihood()
            # Rebuild the node map to ensure it's consistent with the pruned tree
            self.node_map = {n.issue_name: n for n in self.root.traverse()}
        await self.logger.info(f"Tree updated from nodes. Total nodes: {len(self.node_map)}")

    async def update_issue(self, issue_name: str, data: Any):
        """
        Update the data of a single issue in the tree.
        """
        async with self.lock:
            node = self.node_map.get(issue_name)
            if not node:
                await self.logger.error(f"Issue '{issue_name}' not found.")
                raise ValueError(f"Issue '{issue_name}' not found.")
            node.update_data(data)
        await self.logger.info(f"Issue '{issue_name}' updated.")

    async def prune_tree(self, threshold: float = 0.3):
        """
        Prune the tree by removing all nodes (and subtrees) below the likelihood threshold.
        """
        async with self.lock:
            self.root.prune(threshold)
            self.node_map = {n.issue_name: n for n in self.root.traverse()}
        await self.logger.info(f"Tree pruned with threshold {threshold}.")

    async def sort_children(self, issue_name: str, reverse: bool = True):
        """
        Sort the children of the given node by likelihood.
        """
        async with self.lock:
            node = self.node_map.get(issue_name)
            if node:
                node.sort_children_by_likelihood(reverse=reverse)

    async def reset(self):
        """
        Reset the entire tree to just the root node, clearing all other issues.
        """
        async with self.lock:
            root_name = self.root.issue_name
            root_likelihood = self.root.likelihood
            self.root = self._create_root(root_name, root_likelihood)
            self.node_map = {self.root.issue_name: self.root}
        await self.logger.info("Diagnosis tree reset to root.")

    async def expand_node_with_llm(self, node_name: str, context: str, symptom_text: str) -> List[TreeNode]:
        """
        Expand a single node using the LLM chain:
        - Generates potential child issues based on context and symptom text.
        - Updates existing children or adds new ones, updating likelihood and data.
        - Sorts new children by likelihood before returning them.
        """
        async with self.lock:
            parent = self.node_map.get(node_name)
            if not parent:
                await self.logger.error(f"Node '{node_name}' not found.")
                raise ValueError(f"Node '{node_name}' not found.")
            llm_input = {
                "parent_issue": node_name,
                "context": context,
                "symptom_text": symptom_text
            }
            # Invoke LLM chain and parse JSON output
            output = self.chain.invoke(llm_input)
            try:
                parsed_issues = self.output_parser.parse(output)
            except Exception as e:
                await self.logger.error(f"LLM output parsing failed: {e}")
                parsed_issues = []
            new_children = []
            for issue in parsed_issues:
                name = issue.get("issue_name")
                likelihood = issue.get("likelihood", 0)
                data = issue.get("data", None)
                node = self.node_map.get(name)
                if node:
                    # Update existing child's likelihood and data
                    node.likelihood = likelihood
                    node.update_data(data)
                    # If the parent has changed, re-attach node to the new parent
                    if node.parent != parent:
                        if node.parent:
                            node.parent.remove_child(node)
                        parent.add_child(node)
                        node.parent = parent
                else:
                    # Create a new child node and attach it to the parent
                    new_node = DiagnosisTreeAgent.TreeNode(name, likelihood, data)
                    parent.add_child(new_node)
                    new_node.parent = parent
                    self.node_map[name] = new_node
                    new_children.append(new_node)
            # Sort the parent's children by likelihood (descending)
            parent.sort_children_by_likelihood()
        await self.logger.info(f"Node '{node_name}' expanded with LLM. Children count: {len(parent.children)}")
        # Return the list of (new or updated) child nodes for the given parent
        return [self.node_map.get(issue.get("issue_name")) for issue in parsed_issues if issue.get("issue_name") in self.node_map]

    async def expand_all_with_llm(self, context: str, symptom_text: str):
        """
        Recursively expand all nodes in the tree using the LLM chain.
        Each node will generate new child issues, which are then added to the tree.
        """
        async with self.lock:
            async def expand_node(node: DiagnosisTreeAgent.TreeNode):
                llm_input = {
                    "parent_issue": node.issue_name,
                    "context": context,
                    "symptom_text": symptom_text
                }
                output = self.chain.invoke(llm_input)
                try:
                    parsed_issues = self.output_parser.parse(output)
                except Exception as e:
                    await self.logger.error(f"LLM output parsing failed at node '{node.issue_name}': {e}")
                    parsed_issues = []
                for issue in parsed_issues:
                    name = issue.get("issue_name")
                    likelihood = issue.get("likelihood", 0)
                    data = issue.get("data", None)
                    child = self.node_map.get(name)
                    if child:
                        # Update existing child's likelihood and data
                        child.likelihood = likelihood
                        child.update_data(data)
                        if child.parent != node:
                            if child.parent:
                                child.parent.remove_child(child)
                            node.add_child(child)
                            child.parent = node
                    else:
                        # Add new child node to current node
                        new_node = DiagnosisTreeAgent.TreeNode(name, likelihood, data)
                        node.add_child(new_node)
                        new_node.parent = node
                        self.node_map[name] = new_node
                # Sort the current node's children
                node.sort_children_by_likelihood()
                # Recursively expand each child
                for child in list(node.children):
                    await expand_node(child)
            # Start expansion from the root
            await expand_node(self.root)
        await self.logger.info("All nodes expanded with LLM.")
