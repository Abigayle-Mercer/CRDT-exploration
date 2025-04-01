import uuid
from typing import Optional, Dict, List, Tuple
from collections import defaultdict


class CRDTNode:
    def __init__(self, char: str, parent_id: Optional[str], replica_id: str, counter: int):
        self.id = f"{replica_id}:{counter}"  # Unique ID (replica-specific counter)
        self.char = char
        self.parent_id = parent_id
        self.deleted = False
        self.replica_id = replica_id
        self.counter = counter

    def __repr__(self):
        return f"{self.char}({self.id}{' DEL' if self.deleted else ''})"


class CRDTDocument:
    def __init__(self, replica_id: str):
        self.replica_id = replica_id
        self.clock = 0  # Logical clock for this replica
        self.nodes: Dict[str, CRDTNode] = {}
        self.order: Dict[Optional[str], List[str]] = defaultdict(list)

    def insert(self, char: str, after_id: Optional[str]) -> str:
        self.clock += 1
        node = CRDTNode(char, after_id, self.replica_id, self.clock)
        self.nodes[node.id] = node
        self.order[after_id].append(node.id)
        return node.id

    def delete(self, node_id: str):
        if node_id in self.nodes:
            self.nodes[node_id].deleted = True

    def render(self) -> str:
        result = []

        def dfs(node_id):
            for child_id in sorted(self.order.get(node_id, [])):
                node = self.nodes[child_id]
                if not node.deleted:
                    result.append(node.char)
                dfs(child_id)

        dfs(None)
        return ''.join(result)

    def get_all_operations(self) -> List[Tuple[str, CRDTNode]]:
        return [(node_id, self.nodes[node_id]) for node_id in self.nodes]

    def merge(self, remote_ops: List[Tuple[str, CRDTNode]]):
        for node_id, remote_node in remote_ops:
            if node_id not in self.nodes:
                self.nodes[node_id] = remote_node
                self.order[remote_node.parent_id].append(node_id)
            else:
                # If we already have this node, update tombstone status
                if remote_node.deleted:
                    self.nodes[node_id].deleted = True
