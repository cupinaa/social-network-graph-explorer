from __future__ import annotations

from collections import defaultdict, deque
import heapq
from pathlib import Path
import pickle
import time

from models import Interaction, LoadReport, User
from text_utils import jaccard_similarity, levenshtein_distance, tokenize
from trie import Trie


class SocialGraph:
    def __init__(self) -> None:
        self.users_by_id: dict[int, User] = {}
        self.users_by_username: dict[str, User] = {}

        self.outgoing: dict[int, set[int]] = defaultdict(set)
        self.incoming: dict[int, set[int]] = defaultdict(set)
        self.out_degree: dict[int, int] = defaultdict(int)

        self.blocked_out: dict[int, set[int]] = defaultdict(set)
        self.blocked_in: dict[int, set[int]] = defaultdict(set)

        self.inverted_index: dict[str, dict[int, int]] = defaultdict(dict)
        self.username_trie = Trie()

        self.pagerank: dict[int, float] = {}
        self.last_pagerank_iterations = 0
        self.last_pagerank_time = 0.0

        self.followed_history: dict[int, list[Interaction]] = defaultdict(list)
        self.follower_history: dict[int, list[Interaction]] = defaultdict(list)
        self._interaction_counter = 0

    @classmethod
    def build_from_text_files(cls, folder: str | Path) -> tuple["SocialGraph", LoadReport]:
        graph = cls()
        folder_path = Path(folder)
        report = LoadReport(folder=str(folder_path))
        start = time.perf_counter()

        users_path = folder_path / "users.txt"
        connections_path = folder_path / "connections.txt"
        blocked_path = folder_path / "blocked.txt"

        graph._load_users(users_path, report)
        graph._load_connections(connections_path, report)
        graph._load_blocked(blocked_path, report)

        pr_start = time.perf_counter()
        graph.compute_pagerank(use_warm_start=False)
        report.pagerank_time = time.perf_counter() - pr_start
        report.pagerank_iterations = graph.last_pagerank_iterations
        report.load_time = time.perf_counter() - start
        return graph, report

    @classmethod
    def prepare_serialized_data(cls, folder: str | Path) -> Path:
        folder_path = Path(folder)
        graph, report = cls.build_from_text_files(folder_path)
        serialized_path = folder_path / "social_graph.pkl"

        with serialized_path.open("wb") as file:
            pickle.dump((graph, report), file, protocol=pickle.HIGHEST_PROTOCOL)

        return serialized_path

    @classmethod
    def load_from_folder(cls, folder: str | Path) -> tuple["SocialGraph", LoadReport]:
        folder_path = Path(folder)
        serialized_path = folder_path / "social_graph.pkl"
        start = time.perf_counter()

        if not serialized_path.exists():
            raise FileNotFoundError(
                f"Missing {serialized_path}. Run python prepare_data.py first."
            )

        with serialized_path.open("rb") as file:
            graph, report = pickle.load(file)

        report.folder = str(folder_path)
        report.load_time = time.perf_counter() - start
        return graph, report

    def is_ready(self) -> bool:
        return len(self.users_by_id) > 0

    def number_of_edges(self) -> int:
        return sum(len(neighbours) for neighbours in self.outgoing.values())

    def _load_users(self, path: Path, report: LoadReport) -> None:
        with path.open("r", encoding="utf-8-sig", errors="replace") as file:
            for line in file:
                self._load_user_line(line, report)

    def _load_user_line(self, line: str, report: LoadReport) -> None:
        line = line.strip()
        if not line:
            return

        raw_id, username, bio = line.split("|", 2)
        user_id = int(raw_id.strip())

        username = username.strip()
        if not username:
            return

        username_key = username.lower()
        if user_id in self.users_by_id:
            return
        if username_key in self.users_by_username:
            return

        words = tokenize(bio)
        user = User(
            user_id=user_id,
            username=username,
            bio=bio.strip(),
            bio_words=set(words),
        )
        user.bio_word_counts.update(words)

        self.users_by_id[user_id] = user
        self.users_by_username[username_key] = user
        self.outgoing[user_id]
        self.incoming[user_id]
        self.out_degree[user_id] = 0
        self.username_trie.insert(username, user_id)

        for word, count in user.bio_word_counts.items():
            self.inverted_index[word][user_id] = count

        report.users_loaded += 1

    def _load_connections(self, path: Path, report: LoadReport) -> None:
        with path.open("r", encoding="utf-8-sig", errors="replace") as file:
            for line in file:
                self._load_connection_line(line, report)

    def _load_connection_line(self, line: str, report: LoadReport) -> None:
        line = line.strip()
        if not line:
            return

        raw_from_id, raw_to_id = line.split("|")
        from_id = int(raw_from_id.strip())
        to_id = int(raw_to_id.strip())

        if from_id not in self.users_by_id or to_id not in self.users_by_id:
            return
        
        if from_id == to_id:
            return

        if self._add_follow_edge(from_id, to_id):
            report.connections_loaded += 1

    def _load_blocked(self, path: Path, report: LoadReport) -> None:
        with path.open("r", encoding="utf-8-sig", errors="replace") as file:
            for line in file:
                self._load_blocked_line(line, report)

    def _load_blocked_line(self, line: str, report: LoadReport) -> None:
        line = line.strip()
        if not line:
            return

        raw_blocker_id, raw_blocked_id = line.split("|")
        blocker_id = int(raw_blocker_id.strip())
        blocked_id = int(raw_blocked_id.strip())

        if blocker_id not in self.users_by_id or blocked_id not in self.users_by_id:
            return
        if blocker_id == blocked_id:
            return

        if blocked_id not in self.blocked_out[blocker_id]:
            self.blocked_out[blocker_id].add(blocked_id)
            self.blocked_in[blocked_id].add(blocker_id)
            report.blocked_loaded += 1

    def _add_follow_edge(self, from_id: int, to_id: int) -> bool:
        if to_id in self.outgoing[from_id]:
            return False

        self.outgoing[from_id].add(to_id)
        self.incoming[to_id].add(from_id)
        self.out_degree[from_id] = len(self.outgoing[from_id])
        return True

    def resolve_user(self, value: str) -> User | None:
        value = value.strip()
        if not value:
            return None

        user = self.users_by_username.get(value.lower())
        if user is not None:
            return user

        if value.isdigit():
            return self.users_by_id.get(int(value))

        return None

    def has_block_between(self, first_id: int, second_id: int) -> bool:
        return (
            second_id in self.blocked_out.get(first_id, set())
            or first_id in self.blocked_out.get(second_id, set())
        )

    def compute_pagerank(
        self,
        damping: float = 0.85,
        epsilon: float = 1e-6,
        max_iterations: int = 200,
        use_warm_start: bool = True,
    ) -> dict[int, float]:
        start = time.perf_counter()
        user_ids = list(self.users_by_id.keys())
        n = len(user_ids)

        if n == 0:
            self.pagerank = {}
            self.last_pagerank_iterations = 0
            self.last_pagerank_time = time.perf_counter() - start
            return self.pagerank

        if use_warm_start and set(self.pagerank.keys()) == set(user_ids):
            ranks = dict(self.pagerank)
            total = sum(ranks.values())
            if total > 0:
                ranks = {user_id: value / total for user_id, value in ranks.items()}
        else:
            ranks = {user_id: 1.0 / n for user_id in user_ids}

        base_score = (1.0 - damping) / n
        iterations = 0

        for iterations in range(1, max_iterations + 1):
            new_ranks = {user_id: base_score for user_id in user_ids}

            dangling_mass = sum(ranks[user_id] for user_id in user_ids if not self.outgoing[user_id])
            dangling_share = damping * dangling_mass / n
            if dangling_share:
                for user_id in user_ids:
                    new_ranks[user_id] += dangling_share

            for from_id in user_ids:
                neighbours = self.outgoing[from_id]
                if not neighbours:
                    continue
                share = damping * ranks[from_id] / len(neighbours)
                for to_id in neighbours:
                    new_ranks[to_id] += share

            diff = sum(abs(new_ranks[user_id] - ranks[user_id]) for user_id in user_ids)
            ranks = new_ranks
            if diff < epsilon:
                break

        self.pagerank = ranks
        self.last_pagerank_iterations = iterations
        self.last_pagerank_time = time.perf_counter() - start
        return self.pagerank

    def compute_personalized_pagerank(
        self,
        source_id: int,
        damping: float = 0.85,
        epsilon: float = 1e-6,
        max_iterations: int = 200,
    ) -> dict[int, float]:
        user_ids = list(self.users_by_id.keys())
        if source_id not in self.users_by_id:
            return {}

        ranks = {user_id: 0.0 for user_id in user_ids}
        ranks[source_id] = 1.0

        for _ in range(max_iterations):
            new_ranks = {user_id: 0.0 for user_id in user_ids}
            new_ranks[source_id] = 1.0 - damping

            dangling_mass = sum(ranks[user_id] for user_id in user_ids if not self.outgoing[user_id])
            if dangling_mass:
                new_ranks[source_id] += damping * dangling_mass

            for from_id in user_ids:
                neighbours = self.outgoing[from_id]
                if not neighbours:
                    continue
                share = damping * ranks[from_id] / len(neighbours)
                for to_id in neighbours:
                    new_ranks[to_id] += share

            diff = sum(abs(new_ranks[user_id] - ranks[user_id]) for user_id in user_ids)
            ranks = new_ranks
            if diff < epsilon:
                break

        return ranks

    def top_users(self, limit: int = 10) -> list[tuple[User, float]]:
        users = list(self.users_by_id.values())
        top = heapq.nlargest(limit, users, key=lambda user: self.pagerank.get(user.user_id, 0.0))
        top.sort(key=lambda user: (-self.pagerank.get(user.user_id, 0.0), user.username.lower()))
        return [(user, self.pagerank.get(user.user_id, 0.0)) for user in top]

    def search_username(self, query: str, limit: int = 10) -> list[tuple[User, float, float]]:
        query_key = query.strip().lower().rstrip("*")
        if not query_key:
            return []

        scored = []
        for username_key, user in self.users_by_username.items():
            if username_key == query_key:
                relevance = 3.0
            elif username_key.startswith(query_key):
                relevance = 2.0
            elif query_key in username_key:
                relevance = 1.0
            else:
                continue
            scored.append((user, relevance, self.pagerank.get(user.user_id, 0.0)))

        top = heapq.nlargest(limit, scored, key=lambda item: (item[1], item[2]))
        top.sort(key=lambda item: (-item[1], -item[2], item[0].username.lower()))
        return top

    def search_bio(self, query: str, limit: int = 10) -> list[tuple[User, float, float]]:
        query_words = set(tokenize(query))
        if not query_words:
            return []

        matched_words: dict[int, set[str]] = defaultdict(set)
        term_frequency: dict[int, int] = defaultdict(int)

        for word in query_words:
            for user_id, count in self.inverted_index.get(word, {}).items():
                matched_words[user_id].add(word)
                term_frequency[user_id] += count

        scored = []
        for user_id, words in matched_words.items():
            relevance = len(words) / len(query_words) + 0.01 * term_frequency[user_id]
            scored.append((self.users_by_id[user_id], relevance, self.pagerank.get(user_id, 0.0)))

        top = heapq.nlargest(limit, scored, key=lambda item: (item[1], item[2]))
        top.sort(key=lambda item: (-item[1], -item[2], item[0].username.lower()))
        return top

    def autocomplete(self, prefix: str, limit: int = 10) -> list[tuple[User, float]]:
        clean_prefix = prefix.strip().lower().rstrip("*")
        if not clean_prefix:
            return []

        user_ids = self.username_trie.find_by_prefix(clean_prefix)
        users = [self.users_by_id[user_id] for user_id in user_ids]
        top = heapq.nlargest(limit, users, key=lambda user: self.pagerank.get(user.user_id, 0.0))
        top.sort(key=lambda user: (-self.pagerank.get(user.user_id, 0.0), user.username.lower()))
        return [(user, self.pagerank.get(user.user_id, 0.0)) for user in top]

    def did_you_mean(self, username: str, limit: int = 5) -> list[tuple[User, int, float]]:
        username_key = username.strip().lower()
        if not username_key or username_key in self.users_by_username:
            return []

        scored = []
        for user in self.users_by_id.values():
            distance = levenshtein_distance(username_key, user.username.lower())
            scored.append((user, distance, self.pagerank.get(user.user_id, 0.0)))

        scored.sort(key=lambda item: (item[1], -item[2], item[0].username.lower()))
        return scored[:limit]

    def add_user(self, user_id: int, username: str, bio: str) -> tuple[bool, str]:
        username = username.strip()
        username_key = username.lower()

        if user_id in self.users_by_id:
            return False, "A user with that ID already exists."
        if not username:
            return False, "The username cannot be empty."
        if username_key in self.users_by_username:
            return False, "A user with that username already exists."

        words = tokenize(bio)
        user = User(
            user_id=user_id,
            username=username,
            bio=bio.strip(),
            bio_words=set(words),
        )
        user.bio_word_counts.update(words)

        self.users_by_id[user_id] = user
        self.users_by_username[username_key] = user
        self.outgoing[user_id]
        self.incoming[user_id]
        self.out_degree[user_id] = 0
        self.username_trie.insert(username, user_id)

        for word, count in user.bio_word_counts.items():
            self.inverted_index[word][user_id] = count

        self.pagerank[user_id] = 1.0 / len(self.users_by_id)
        self.compute_pagerank(use_warm_start=True)
        return True, (
            f"Added user {username} (ID={user_id}). "
            f"PageRank was recomputed in {self.last_pagerank_iterations} iterations."
        )

    def add_follow(self, from_value: str, to_value: str) -> tuple[bool, str]:
        from_user = self.resolve_user(from_value)
        to_user = self.resolve_user(to_value)

        if from_user is None:
            return False, "The follower does not exist."
        if to_user is None:
            return False, "The user to follow does not exist."
        if from_user.user_id == to_user.user_id:
            return False, "A user cannot follow themselves."
        if to_user.user_id in self.outgoing[from_user.user_id]:
            return False, "That follow relationship already exists."
        if self.has_block_between(from_user.user_id, to_user.user_id):
            return False, "The relationship is not allowed because one user has blocked the other."

        self._add_follow_edge(from_user.user_id, to_user.user_id)

        self._interaction_counter += 1
        interaction = Interaction(
            order=self._interaction_counter,
            from_id=from_user.user_id,
            to_id=to_user.user_id,
        )
        self.followed_history[from_user.user_id].append(interaction)
        self.follower_history[to_user.user_id].append(interaction)

        self.compute_pagerank(use_warm_start=True)
        return True, (
            f"Added relationship: {from_user.username} follows {to_user.username}. "
            f"PageRank was recomputed in {self.last_pagerank_iterations} iterations."
        )

    def get_history(self, user_value: str) -> tuple[User | None, list[Interaction], list[Interaction]]:
        user = self.resolve_user(user_value)
        if user is None:
            return None, [], []

        followed = list(self.followed_history.get(user.user_id, []))
        followers = list(self.follower_history.get(user.user_id, []))
        followed.sort(key=lambda interaction: interaction.order)
        followers.sort(key=lambda interaction: interaction.order)
        return user, followed, followers

    def bfs_levels(self, user_value: str, max_depth: int) -> tuple[User | None, dict[int, list[User]]]:
        start_user = self.resolve_user(user_value)
        if start_user is None or max_depth < 1:
            return start_user, {}

        visited = {start_user.user_id}
        queue = deque([(start_user.user_id, 0)])
        levels: dict[int, list[User]] = defaultdict(list)

        while queue:
            current_id, depth = queue.popleft()
            if depth == max_depth:
                continue

            for neighbour_id in self.outgoing[current_id]:
                if neighbour_id in visited:
                    continue
                visited.add(neighbour_id)
                next_depth = depth + 1
                levels[next_depth].append(self.users_by_id[neighbour_id])
                queue.append((neighbour_id, next_depth))

        for users in levels.values():
            users.sort(key=lambda user: user.username.lower())
        return start_user, dict(levels)

    def hybrid_recommendations(
        self,
        user_value: str,
        alpha: float,
        limit: int = 10,
    ) -> tuple[User | None, list[tuple[User, float, float, float]]]:
        source_user = self.resolve_user(user_value)
        if source_user is None:
            return None, []

        alpha = max(0.0, min(1.0, alpha))
        ppr = self.compute_personalized_pagerank(source_user.user_id)
        already_following = self.outgoing[source_user.user_id]
        scored = []

        for candidate in self.users_by_id.values():
            if candidate.user_id == source_user.user_id:
                continue
            if candidate.user_id in already_following:
                continue
            if self.has_block_between(source_user.user_id, candidate.user_id):
                continue

            content_similarity = jaccard_similarity(source_user.bio_words, candidate.bio_words)
            ppr_score = ppr.get(candidate.user_id, 0.0)
            combined_score = alpha * ppr_score + (1.0 - alpha) * content_similarity

            if combined_score > 0:
                scored.append((candidate, combined_score, ppr_score, content_similarity))

        top = heapq.nlargest(limit, scored, key=lambda item: (item[1], self.pagerank.get(item[0].user_id, 0.0)))
        top.sort(
            key=lambda item: (
                -item[1],
                -self.pagerank.get(item[0].user_id, 0.0),
                item[0].username.lower(),
            )
        )
        return source_user, top

    def interaction_to_text(self, interaction: Interaction) -> str:
        from_user = self.users_by_id.get(interaction.from_id)
        to_user = self.users_by_id.get(interaction.to_id)
        from_name = from_user.username if from_user else str(interaction.from_id)
        to_name = to_user.username if to_user else str(interaction.to_id)
        return f"{interaction.order}. {from_name} -> {to_name}"

def find_dataset_folders(base_folder: str | Path, max_depth: int = 3) -> list[Path]:
    base_path = Path(base_folder)
    result = []

    if (base_path / "users.txt").exists():
        result.append(base_path)

    for path in base_path.rglob("users.txt"):
        try:
            relative_depth = len(path.relative_to(base_path).parts) - 1
        except ValueError:
            continue
        if relative_depth <= max_depth:
            result.append(path.parent)

    unique = sorted(set(result), key=lambda item: str(item).lower())
    return unique
