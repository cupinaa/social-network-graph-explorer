from dataclasses import dataclass, field
from collections import Counter


@dataclass
class User:
    user_id: int
    username: str
    bio: str
    bio_words: set[str] = field(default_factory=set)
    bio_word_counts: Counter = field(default_factory=Counter)


@dataclass
class Interaction:
    order: int
    from_id: int
    to_id: int


@dataclass
class LoadReport:
    folder: str
    users_loaded: int = 0
    connections_loaded: int = 0
    blocked_loaded: int = 0
    load_time: float = 0.0
    pagerank_time: float = 0.0
    pagerank_iterations: int = 0
