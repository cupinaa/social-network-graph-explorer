import re


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def jaccard_similarity(first: set[str], second: set[str]) -> float:
    union = first | second
    if not union:
        return 0.0

    intersection = first & second
    return len(intersection) / len(union)


def levenshtein_distance(first: str, second: str) -> int:
    first = first.lower()
    second = second.lower()

    if first == second:
        return 0
    if not first:
        return len(second)
    if not second:
        return len(first)

    previous = list(range(len(second) + 1))

    for i, char_first in enumerate(first, start=1):
        current = [i]
        for j, char_second in enumerate(second, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1]
            if char_first != char_second:
                replace_cost += 1
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    return previous[-1]
