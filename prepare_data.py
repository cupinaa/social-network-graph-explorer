from pathlib import Path
import time

from social_graph import SocialGraph, find_dataset_folders


def main() -> None:
    dataset_folder = Path.cwd() / "dataset"
    folders = find_dataset_folders(dataset_folder)

    if not folders:
        print("No datasets were found in the dataset directory.")
        return

    for folder in folders:
        start = time.perf_counter()
        serialized_path = SocialGraph.prepare_serialized_data(folder)
        elapsed = time.perf_counter() - start
        print(f"Prepared {folder.name}: {serialized_path} ({elapsed:.4f} s)")


if __name__ == "__main__":
    main()
