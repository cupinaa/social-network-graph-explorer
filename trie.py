from dataclasses import dataclass, field


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    user_ids: set[int] = field(default_factory=set)


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str, user_id: int) -> None:
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.user_ids.add(user_id)

    def find_by_prefix(self, prefix: str) -> set[int]:
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return set()
            node = node.children[char]

        return self._collect_user_ids(node)

    def _collect_user_ids(self, node: TrieNode) -> set[int]:
        result: set[int] = set()
        stack = [node]

        while stack:
            current = stack.pop()
            result.update(current.user_ids)
            for child in current.children.values():
                stack.append(child)

        return result
