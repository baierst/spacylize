"""Dataset analysis helpers used by the validator.

Provides duplicate detection, diversity metrics, label quality checks, the topic
check and the per-document issues shown in the report.
"""

import re
from collections import Counter, defaultdict
from itertools import pairwise

from spacy.tokens import Doc


def diversity(docs: list[Doc]) -> dict:
    """Compute lexical diversity metrics.

    Args:
        docs: List of spaCy Doc objects.

    Returns:
        dict: distinct-1/2 ratios and the most repeated 3-word openings.
    """
    tokens = [[t.lower_ for t in doc if not (t.is_punct or t.is_space)] for doc in docs]
    unigrams = [t for toks in tokens for t in toks]
    bigrams = [b for toks in tokens for b in pairwise(toks)]

    def distinct(items: list) -> float:
        return round(len(set(items)) / len(items), 3) if items else 0

    openings = Counter(" ".join(toks[:3]) for toks in tokens if len(toks) >= 3)
    return {
        "distinct_1": distinct(unigrams),
        "distinct_2": distinct(bigrams),
        "top_openings": openings.most_common(5),
    }


def label_quality(docs: list[Doc], task: str) -> dict:
    """Check label quality.

    For NER: unique entity values per label and entity texts used with several
    labels. For textcat: documents with no or several positive categories.

    Args:
        docs: List of spaCy Doc objects.
        task: 'ner' or 'textcat'.

    Returns:
        dict: Label quality statistics.
    """
    if task == "textcat":
        positives = [sum(score > 0.5 for score in doc.cats.values()) for doc in docs]
        return {
            "no_category": positives.count(0),
            "multiple_categories": sum(p > 1 for p in positives),
        }

    values = defaultdict(Counter)
    labels_by_text = defaultdict(set)
    for doc in docs:
        for ent in doc.ents:
            text = ent.text.lower()
            values[ent.label_][text] += 1
            labels_by_text[text].add(ent.label_)

    return {
        "by_label": {
            label: {
                "total": sum(counts.values()),
                "unique": len(counts),
                "values": counts.most_common(),
            }
            for label, counts in sorted(values.items())
        },
        "conflicts": [
            {"text": text, "labels": sorted(labels)}
            for text, labels in sorted(labels_by_text.items())
            if len(labels) > 1
        ],
    }


def check_labels(docs: list[Doc], task: str, labels: list[str]) -> dict:
    """Compare the labels in the dataset with the configured labels.

    Args:
        docs: List of spaCy Doc objects.
        task: 'ner' or 'textcat'.
        labels: List of configured labels.

    Returns:
        dict: 'counts' as the number of uses per configured label, 'missing' as
        configured labels that are never used, and 'unknown' as labels in the data
        that are not configured. A textcat label counts as used when it is a
        positive category (score > 0.5).
    """
    counts = Counter()
    for doc in docs:
        if task == "textcat":
            counts.update(label for label, score in doc.cats.items() if score > 0.5)
        else:
            counts.update(ent.label_ for ent in doc.ents)

    return {
        "counts": {label: counts[label] for label in labels},
        "missing": [label for label in labels if counts[label] == 0],
        "unknown": sorted(set(counts) - set(labels)),
    }


def off_topic_ids(docs: list[Doc]) -> set[int]:
    """Indices of documents with a topic score below 0.5, as stored by generate.

    Args:
        docs: List of spaCy Doc objects.

    Returns:
        set: Document indices. Empty if no document has a topic score.
    """
    return {
        i for i, doc in enumerate(docs) if doc.user_data.get("topic_score", 1) < 0.5
    }


def duplicate_ids(texts: list[str]) -> set[int]:
    """Indices of documents that repeat an earlier one.

    Texts are compared after lower-casing and removing punctuation and extra
    whitespace, so copies that differ only in these still count.

    Args:
        texts: List of document texts.

    Returns:
        set: All but the first document of each group of duplicates.
    """
    seen, ids = set(), set()
    for i, text in enumerate(texts):
        key = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
        if key in seen:
            ids.add(i)
        seen.add(key)
    return ids


def find_issues(
    docs: list[Doc], task: str, labels: list[str] | None = None
) -> dict[int, list[str]]:
    """Run the validator checks on each document.

    Args:
        docs: List of spaCy Doc objects.
        task: 'ner' or 'textcat'.
        labels: Optional configured labels, to find unknown labels.

    Returns:
        dict: Document index -> list of issues, only for documents with issues.
    """
    if not docs:
        return {}
    issues = defaultdict(list)
    for i in duplicate_ids([doc.text for doc in docs]):
        issues[i].append("duplicate")
    for i in off_topic_ids(docs):
        issues[i].append("off-topic")
    conflicts = {c["text"] for c in label_quality(docs, task).get("conflicts", [])}
    for i, doc in enumerate(docs):
        if any(e.text.lower() in conflicts for e in doc.ents):
            issues[i].append("label conflict")
        if labels is not None:
            used = {e.label_ for e in doc.ents} | {
                c for c, s in doc.cats.items() if s > 0.5
            }
            if used - set(labels):
                issues[i].append("unknown label")
        if "[" in doc.text:
            issues[i].append("leftover markup")
    return dict(sorted(issues.items()))
