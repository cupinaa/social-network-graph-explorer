from __future__ import annotations

from pathlib import Path
import pickle
import sys
import time

from models import LoadReport, User
from social_graph import SocialGraph, find_dataset_folders


def print_load_report(report: LoadReport) -> None:
    print("\n=== Load report ===")
    print(f"Dataset: {Path(report.folder).name}")
    print(f"Users: {report.users_loaded}")
    print(f"Follow relationships: {report.connections_loaded}")
    print(f"Blocks: {report.blocked_loaded}")
    print(f"Load time: {report.load_time:.4f} s")
    print(f"PageRank: {report.pagerank_iterations} iterations, {report.pagerank_time:.4f} s")


def load_graph(folder: Path) -> tuple[SocialGraph, LoadReport]:
    try:
        graph, report = SocialGraph.load_from_folder(folder)
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as error:
        print(f"Load error: {error}")
        print("Run 'python prepare_data.py', then start the application again.")
        raise SystemExit(1)
    print_load_report(report)
    return graph, report


def ensure_data(graph: SocialGraph) -> bool:
    if graph.is_ready():
        return True
    print("No data is loaded. Check the directory containing users.txt, connections.txt, and blocked.txt.")
    return False


def read_int(prompt: str, default: int, minimum: int = 1) -> int:
    raw_value = input(prompt).strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        print(f"The input is not an integer. Using the default value {default}.")
        return default
    if value < minimum:
        print(f"The value must be at least {minimum}. Using {default}.")
        return default
    return value


def read_float(prompt: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    raw_value = input(prompt).strip().replace(",", ".")
    if not raw_value:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        print(f"The input is not a number. Using the default value {default}.")
        return default
    if value < minimum or value > maximum:
        print(f"The value must be between {minimum} and {maximum}. Using {default}.")
        return default
    return value


def print_user_results(rows: list[tuple[User, float, float]], score_name: str) -> None:
    if not rows:
        print("No results.")
        return

    for position, (user, score, pagerank) in enumerate(rows, start=1):
        print(f"{position}. {user.username} (id={user.user_id}, {score_name}={score:.6f}, PageRank={pagerank:.8f})")
        print(f"   Bio: {user.bio}")


def option_top_users(graph: SocialGraph) -> None:
    if not ensure_data(graph):
        return
    limit = read_int("How many users should be shown? [10]: ", 10)
    start = time.perf_counter()
    rows = graph.top_users(limit)
    elapsed = time.perf_counter() - start

    if not rows:
        print("No users found.")
        return
    for position, (user, pagerank) in enumerate(rows, start=1):
        print(f"{position}. {user.username} (id={user.user_id}, PageRank={pagerank:.8f})")
    print(f"Operation completed in {elapsed:.4f} s.")


def option_search_username(graph: SocialGraph) -> None:
    if not ensure_data(graph):
        return
    query = input("Enter part of a username: ")
    limit = read_int("How many results should be shown? [10]: ", 10)
    start = time.perf_counter()
    rows = graph.search_username(query, limit)
    elapsed = time.perf_counter() - start
    print_user_results(rows, "relevance")
    print(f"Operation completed in {elapsed:.4f} s.")

    if not rows:
        suggestions = graph.did_you_mean(query, 5)
        if suggestions:
            print("Did you mean:")
            for user, distance, pagerank in suggestions:
                print(f"- {user.username} (distance={distance}, PageRank={pagerank:.8f})")


def option_search_bio(graph: SocialGraph) -> None:
    if not ensure_data(graph):
        return
    query = input("Enter words to search for in biographies: ")
    limit = read_int("How many results should be shown? [10]: ", 10)
    start = time.perf_counter()
    rows = graph.search_bio(query, limit)
    elapsed = time.perf_counter() - start
    print_user_results(rows, "relevance")
    print(f"Operation completed in {elapsed:.4f} s.")


def option_add_follow(graph: SocialGraph) -> None:
    if not ensure_data(graph):
        return
    from_value = input("Follower (username or ID): ")
    to_value = input("User to follow (username or ID): ")
    start = time.perf_counter()
    success, message = graph.add_follow(from_value, to_value)
    elapsed = time.perf_counter() - start
    print(message)
    if success:
        print(f"PageRank recomputation took {graph.last_pagerank_time:.4f} s.")
    print(f"Operation completed in {elapsed:.4f} s.")


def option_add_user(graph: SocialGraph) -> None:
    raw_id = input("New user ID: ").strip()
    try:
        user_id = int(raw_id)
    except ValueError:
        print("The ID must be an integer.")
        return

    username = input("New username: ")
    bio = input("New user biography: ")
    start = time.perf_counter()
    success, message = graph.add_user(user_id, username, bio)
    elapsed = time.perf_counter() - start
    print(message)
    if success:
        print(f"PageRank recomputation took {graph.last_pagerank_time:.4f} s.")
    print(f"Operation completed in {elapsed:.4f} s.")


def option_history(graph: SocialGraph) -> None:
    if not ensure_data(graph):
        return
    value = input("Enter a username or user ID: ")
    user, followed, followers = graph.get_history(value)
    if user is None:
        print("The user does not exist.")
        return

    print(f"\nHistory for {user.username}:")
    print("Users they followed:")
    if followed:
        for interaction in followed:
            print("- " + graph.interaction_to_text(interaction))
    else:
        print("- No interactions were added during this session.")

    print("Users who followed them:")
    if followers:
        for interaction in followers:
            print("- " + graph.interaction_to_text(interaction))
    else:
        print("- No interactions were added during this session.")


def option_autocomplete(graph: SocialGraph) -> None:
    if not ensure_data(graph):
        return
    prefix = input("Enter a username prefix (for example, mar*): ")
    limit = read_int("How many suggestions should be shown? [10]: ", 10)
    start = time.perf_counter()
    rows = graph.autocomplete(prefix, limit)
    elapsed = time.perf_counter() - start

    if not rows:
        print("No autocomplete suggestions.")
    else:
        for position, (user, pagerank) in enumerate(rows, start=1):
            print(f"{position}. {user.username} (id={user.user_id}, PageRank={pagerank:.8f})")
    print(f"Operation completed in {elapsed:.4f} s.")


def option_recommendations(graph: SocialGraph) -> None:
    if not ensure_data(graph):
        return
    value = input("Recommendations for which user? (username or ID): ")
    alpha = read_float("Enter alpha between 0 and 1 [0.5]: ", 0.5)
    limit = read_int("How many recommendations should be shown? [10]: ", 10)
    start = time.perf_counter()
    user, rows = graph.hybrid_recommendations(value, alpha, limit)
    elapsed = time.perf_counter() - start

    if user is None:
        print("The user does not exist.")
        return
    if not rows:
        print("No recommendations are available.")
    else:
        print(f"Recommendations for {user.username}:")
        for position, (candidate, score, ppr, similarity) in enumerate(rows, start=1):
            print(
                f"{position}. {candidate.username} "
                f"(score={score:.8f}, PPR={ppr:.8f}, similarity={similarity:.6f}, "
                f"PageRank={graph.pagerank.get(candidate.user_id, 0.0):.8f})"
            )
            print(f"   Bio: {candidate.bio}")
    print(f"Operation completed in {elapsed:.4f} s.")


def option_bfs(graph: SocialGraph) -> None:
    if not ensure_data(graph):
        return
    value = input("Starting user (username or ID): ")
    depth = read_int("Maximum depth? [3]: ", 3)
    start = time.perf_counter()
    user, levels = graph.bfs_levels(value, depth)
    elapsed = time.perf_counter() - start

    if user is None:
        print("The user does not exist.")
        return
    if not levels:
        print("There are no connections to show, or the depth is less than 1.")
    else:
        print(f"BFS levels from {user.username}'s perspective:")
        for level in range(1, depth + 1):
            users = levels.get(level, [])
            print(f"Level {level}:")
            if users:
                print(", ".join(user.username for user in users))
            else:
                print("-")
    print(f"Operation completed in {elapsed:.4f} s.")


def option_did_you_mean(graph: SocialGraph) -> None:
    if not ensure_data(graph):
        return
    username = input("Enter a misspelled or incomplete username: ")
    limit = read_int("How many suggestions should be shown? [5]: ", 5)
    start = time.perf_counter()
    rows = graph.did_you_mean(username, limit)
    elapsed = time.perf_counter() - start

    if not rows:
        print("No suggestions, or the username already exists.")
    else:
        for position, (user, distance, pagerank) in enumerate(rows, start=1):
            print(f"{position}. {user.username} (distance={distance}, PageRank={pagerank:.8f})")
    print(f"Operation completed in {elapsed:.4f} s.")


def option_stats(graph: SocialGraph, report: LoadReport) -> None:
    print_load_report(report)
    if graph.is_ready():
        print("\n=== Current graph state ===")
        print(f"Users: {len(graph.users_by_id)}")
        print(f"Follow relationships: {graph.number_of_edges()}")
        print(f"Terms in the inverted index: {len(graph.inverted_index)}")
        print(f"Latest PageRank: {graph.last_pagerank_iterations} iterations, {graph.last_pagerank_time:.4f} s")


def option_reload() -> tuple[SocialGraph, LoadReport]:
    return load_graph(choose_dataset_folder())


def print_menu() -> None:
    print("\n=== Social Network Explorer ===")
    print("\n--- Search ---")
    print("1. Search by username")
    print("2. Search biography terms")
    print("3. Autocomplete usernames")
    print("4. Did-you-mean suggestions")
    print("\n--- Ranking and recommendations ---")
    print("5. Top influential users")
    print("6. Hybrid recommendations")
    print("\n--- Graph and interactions ---")
    print("7. Explore the network with BFS")
    print("8. Add a follow relationship")
    print("9. Add a user")
    print("10. Interaction history")
    print("\n--- Data ---")
    print("11. Statistics and timings")
    print("12. Change dataset")
    print("0. Exit")


def menu_loop(graph: SocialGraph, report: LoadReport) -> None:
    while True:
        print_menu()
        choice = input("\nChoice: ").strip()

        if choice == "1":
            option_search_username(graph)
        elif choice == "2":
            option_search_bio(graph)
        elif choice == "3":
            option_autocomplete(graph)
        elif choice == "4":
            option_did_you_mean(graph)
        elif choice == "5":
            option_top_users(graph)
        elif choice == "6":
            option_recommendations(graph)
        elif choice == "7":
            option_bfs(graph)
        elif choice == "8":
            option_add_follow(graph)
        elif choice == "9":
            option_add_user(graph)
        elif choice == "10":
            option_history(graph)
        elif choice == "11":
            option_stats(graph, report)
        elif choice == "12":
            graph, report = option_reload()
        elif choice == "0":
            print("Goodbye.")
            break
        else:
            print("Unknown option. Try again.")


def choose_dataset_folder() -> Path:
    current = Path.cwd()
    candidates = find_dataset_folders(current)
    folders_by_name = {candidate.name.lower(): candidate for candidate in candidates}

    available = [name for name in ("small", "medium", "full") if name in folders_by_name]
    if not available:
        print("No small, medium, or full dataset was found in the dataset directory.")
        return current

    if len(candidates) == 1:
        print(f"Using the only dataset found: {candidates[0].name}")
        return candidates[0]

    print("\n=== Dataset selection ===")
    print("1. small")
    print("2. medium")
    print("3. full")

    choices = {
        "1": "small",
        "2": "medium",
        "3": "full",
    }

    while True:
        choice = input("Choice [1]: ").strip().lower() or "1"
        selected_name = choices.get(choice)
        if selected_name in folders_by_name:
            print(f"Selected the {selected_name} dataset.")
            return folders_by_name[selected_name]
        print("Unknown choice. Enter 1, 2, or 3.")


def choose_initial_folder() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])

    return choose_dataset_folder()


def main() -> None:
    folder = choose_initial_folder()
    graph, report = load_graph(folder)
    menu_loop(graph, report)


if __name__ == "__main__":
    main()
