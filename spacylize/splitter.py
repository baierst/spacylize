"""Dataset splitting module.

Splits a SpaCy dataset into any number of named sets, e.g. train/dev or
train/dev/test, using given fractions.
"""

import math
import random
from pathlib import Path

import spacy
from spacy.tokens import DocBin


def split_dataset(dataset_path, splits, output_folder, seed=42):
    """Shuffle a dataset and split it into named sets.

    Each set is written to ``<output_folder>/<name>.spacy``.

    Args:
        dataset_path: Path to the SpaCy dataset file (.spacy).
        splits: Mapping of set name to fraction, e.g.
            ``{"train": 0.8, "dev": 0.1, "test": 0.1}``. The fractions must
            be positive and sum to 1.
        output_folder: Folder where the sets will be written.
        seed: Random seed for a reproducible shuffle.

    Returns:
        dict: Number of documents per set name.

    Raises:
        ValueError: If no sets are given, a fraction is not positive or the
            fractions do not sum to 1.
    """
    if not splits:
        raise ValueError("At least one split is required.")
    if any(fraction <= 0 for fraction in splits.values()):
        raise ValueError("Split fractions must be greater than 0.")
    if not math.isclose(sum(splits.values()), 1.0, abs_tol=1e-6):
        raise ValueError(
            f"Split fractions must sum to 1, got {sum(splits.values()):g}."
        )

    docs = list(DocBin().from_disk(dataset_path).get_docs(spacy.blank("en").vocab))
    random.Random(seed).shuffle(docs)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    counts = {}
    start = 0
    cumulative = 0.0
    names = list(splits)
    for name in names:
        cumulative += splits[name]
        # Cumulative boundaries assign every document exactly once; the last
        # set always ends at the last document, whatever the rounding.
        end = len(docs) if name == names[-1] else round(cumulative * len(docs))
        part = docs[start:end]
        DocBin(docs=part, store_user_data=True).to_disk(output_folder / f"{name}.spacy")
        counts[name] = len(part)
        start = end
    return counts
