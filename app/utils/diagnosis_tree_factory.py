from app.utils.diagnosis_tree import DiagnosisTreeNode

def get_diagnosis_tree(*, issue_name='root', likelyhood=1.0):
    """
    Factory function for creating a new DiagnosisTreeNode instance.
    In the future, this can be extended to provide per-session singleton trees.
    """
    return DiagnosisTreeNode(issue_name=issue_name, likelyhood=likelyhood)
