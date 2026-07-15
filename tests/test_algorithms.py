import tempfile
import unittest
from pathlib import Path

from social_graph import SocialGraph
from text_utils import jaccard_similarity, levenshtein_distance
from trie import Trie


class TextAlgorithmTests(unittest.TestCase):
    def test_jaccard_similarity(self) -> None:
        self.assertEqual(jaccard_similarity({"graph", "python"}, {"graph"}), 0.5)

    def test_levenshtein_distance(self) -> None:
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)


class TrieTests(unittest.TestCase):
    def test_prefix_search_is_case_insensitive(self) -> None:
        trie = Trie()
        trie.insert("GraphFan", 7)
        self.assertEqual(trie.find_by_prefix("GRAPH"), {7})


class SocialGraphTests(unittest.TestCase):
    def test_load_search_and_block_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "users.txt").write_text(
                "1|alice|python graphs\n2|bob|python data\n3|carol|music\n",
                encoding="utf-8",
            )
            (folder / "connections.txt").write_text("1|2\n2|3\n", encoding="utf-8")
            (folder / "blocked.txt").write_text("1|3\n", encoding="utf-8")

            graph, report = SocialGraph.build_from_text_files(folder)

            self.assertEqual(report.users_loaded, 3)
            self.assertEqual(graph.number_of_edges(), 2)
            self.assertEqual(graph.search_bio("python")[0][0].username, "bob")
            self.assertFalse(graph.add_follow("alice", "carol")[0])
            self.assertAlmostEqual(sum(graph.pagerank.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
