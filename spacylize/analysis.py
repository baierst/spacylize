"""Dataset analysis helpers used by the validator.

Provides document embeddings, a 2D projection for the embedding map, duplicate
detection, diversity metrics and label quality checks.
"""

from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE


def embed(texts):
    """Embed texts as L2-normalised TF-IDF vectors.

    Args:
        texts: List of document texts.

    Returns:
        Sparse matrix with one row per text.
    """
    return TfidfVectorizer(sublinear_tf=True).fit_transform(texts)


def project_2d(X, seed=42):
    """Project document vectors to 2D for the embedding map.

    Uses truncated SVD, followed by t-SNE for datasets with at least 10 documents.

    Args:
        X: Document vectors (one row per document).
        seed: Random seed for reproducible layouts.

    Returns:
        numpy.ndarray: Array of shape (n_docs, 2).
    """
    n = X.shape[0]
    k = min(50, n - 1, X.shape[1] - 1)
    if k < 2:
        return np.zeros((n, 2))

    reduced = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)
    if n < 10:
        return reduced[:, :2]

    tsne = TSNE(
        n_components=2, perplexity=min(30, (n - 1) / 3), init="pca", random_state=seed
    )
    return tsne.fit_transform(reduced)


def find_duplicates(texts, X, threshold=0.9):
    """Find exact and near-duplicate documents.

    Args:
        texts: List of document texts.
        X: L2-normalised document vectors.
        threshold: Minimum cosine similarity for a near-duplicate pair.

    Returns:
        dict: 'exact' as groups of document indices, 'near' as
        [index_a, index_b, similarity] pairs that are not exact duplicates.
    """
    groups = defaultdict(list)
    for i, text in enumerate(texts):
        groups[" ".join(text.lower().split())].append(i)
    exact = [group for group in groups.values() if len(group) > 1]
    exact_pairs = {(a, b) for group in exact for a in group for b in group if a < b}

    sims = (X @ X.T).toarray()
    near = [
        [int(a), int(b), round(float(sims[a, b]), 3)]
        for a, b in np.argwhere(np.triu(sims, k=1) >= threshold)
        if (a, b) not in exact_pairs
    ]
    return {"exact": exact, "near": near}


def diversity(docs):
    """Compute lexical diversity metrics.

    Args:
        docs: List of spaCy Doc objects.

    Returns:
        dict: distinct-1/2 ratios, the most repeated 3-word openings and the
        most frequent terms.
    """
    tokens = [[t.lower_ for t in doc if not (t.is_punct or t.is_space)] for doc in docs]
    unigrams = [t for toks in tokens for t in toks]
    bigrams = [b for toks in tokens for b in zip(toks, toks[1:])]

    def distinct(items):
        return round(len(set(items)) / len(items), 3) if items else 0

    openings = Counter(" ".join(toks[:3]) for toks in tokens if len(toks) >= 3)
    terms = Counter(
        t.lower_
        for doc in docs
        for t in doc
        if t.is_alpha and not t.is_stop and len(t) > 2
    )
    return {
        "distinct_1": distinct(unigrams),
        "distinct_2": distinct(bigrams),
        "top_openings": openings.most_common(10),
        "top_terms": terms.most_common(20),
    }


def label_quality(docs, task):
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
                "top_values": counts.most_common(5),
            }
            for label, counts in sorted(values.items())
        },
        "conflicts": [
            {"text": text, "labels": sorted(labels)}
            for text, labels in sorted(labels_by_text.items())
            if len(labels) > 1
        ],
    }


def check_labels(docs, task, labels):
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
