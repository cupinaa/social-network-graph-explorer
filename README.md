# Social Network Graph Explorer

A dependency-free Python command-line application for searching, ranking, and exploring directed social-network graphs.

The project is intended for students and developers interested in practical graph algorithms and indexing data structures. It loads pipe-delimited user, relationship, and block data; builds efficient in-memory indexes; and exposes the results through an interactive terminal menu.

![Social Network Graph Explorer preview](docs/assets/social-preview.png)

## Key features

- Iterative PageRank with dangling-node handling and warm starts
- Personalized PageRank combined with Jaccard biography similarity for recommendations
- Username and biography search backed by hash maps and an inverted index
- Trie-based, case-insensitive username autocomplete
- Levenshtein-distance spelling suggestions
- Breadth-first graph traversal by level
- Block-aware follow relationships and session interaction history
- A synthetic English demo dataset with no personal information

## Technology and architecture

The application uses Python 3.10 or newer and only the standard library. `SocialGraph` owns adjacency sets, reverse adjacency sets, block indexes, PageRank state, a biography inverted index, and a username trie. The CLI delegates graph operations to that domain class, while small utility modules contain text algorithms and data models.

## Getting started

### Prerequisites

- Python 3.10+

No third-party packages are required.

### Prepare the data

Serialized graph files are generated locally and intentionally excluded from version control:

```bash
python prepare_data.py
```

Run this command again whenever an input text file changes.

### Run the application

```bash
python main.py
```

Choose the included dataset interactively, or pass it directly:

```bash
python main.py dataset/small
```

### Run the tests

```bash
python -m unittest discover -s tests -v
```

## Input format

Each dataset directory contains three pipe-delimited files:

```text
users.txt:       id|username|bio
connections.txt: from_id|to_id
blocked.txt:     blocker_id|blocked_id
```

Malformed references, duplicate users and relationships, self-links, and unknown user IDs are ignored during loading.

## Example workflows

With the `small` dataset, try username search for `al`, biography search for `python graph`, autocomplete for `gr*`, or a did-you-mean query for `alce`. Menu option 11 displays graph size and the latest PageRank timings.

## Project structure

```text
main.py          Interactive command-line interface
social_graph.py  Graph storage, loading, ranking, search, and recommendations
models.py        User, interaction, and load-report data classes
trie.py          Prefix-search data structure
text_utils.py    Tokenization, Jaccard similarity, and Levenshtein distance
prepare_data.py  Local serialized-data generator
dataset/         Pipe-delimited sample datasets
tests/           Standard-library unit tests
```

## Known limitations

- Changes made in the CLI are held in memory and are not written back to the source files.
- Interaction history includes only relationships added during the current session.
- Personalized PageRank and spelling suggestions scan the loaded graph, so performance depends on dataset size.
- Pickle files must only be generated from and loaded within a trusted local checkout.

## Data and license

The included dataset is synthetic and exists only to demonstrate the application. The project is available under the [MIT License](LICENSE).

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [SECURITY.md](SECURITY.md) for responsible vulnerability reporting.
