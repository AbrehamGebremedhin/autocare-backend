from typing import Any, List, Optional
from abc import ABC, abstractmethod

class AbstractTreeNode(ABC):
    def __init__(self, issue_name: str, likelyhood: float, data: Any = None, parent: Optional['AbstractTreeNode'] = None):
        self.issue_name = issue_name
        self.likelyhood = likelyhood
        self.data = data
        self.children: List['AbstractTreeNode'] = []
        self.parent: Optional['AbstractTreeNode'] = parent

    def add_child(self, child: 'AbstractTreeNode'):
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: 'AbstractTreeNode'):
        self.children.remove(child)
        child.parent = None

    def find(self, issue_name: str) -> Optional['AbstractTreeNode']:
        if self.issue_name == issue_name:
            return self
        for child in self.children:
            result = child.find(issue_name)
            if result:
                return result
        return None

    def traverse(self):
        yield self
        for child in self.children:
            yield from child.traverse()

    def update_data(self, data: Any):
        """Update the data attribute of the node."""
        self.data = data

    def prune(self, threshold: float = 0.3):
        """Prune children whose likelyhood is below the threshold (as a fraction, e.g., 0.3 for 30%)."""
        pruned_children = []
        for child in self.children:
            if child.likelyhood < threshold:
                pruned_children.append(child)
            else:
                child.prune(threshold)
        for child in pruned_children:
            self.children.remove(child)

    def sort_children_by_likelyhood(self, reverse: bool = True):
        """Sort children nodes by their likelyhood attribute."""
        self.children.sort(key=lambda child: child.likelyhood, reverse=reverse)

    @abstractmethod
    def process(self):
        pass

