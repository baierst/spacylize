"""Dataset splitting module.

Splits a SpaCy dataset into any number of named sets, e.g. train/dev or
train/dev/test, using given fractions.
"""

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import spacy
from spacy.tokens import DocBin


@dataclass
class DataSplitter:
    """Shuffle a dataset and split it into named sets.

    Each set is written to ``<output_folder>/<name>.spacy``.

    Attributes:
        dataset_path: Path to the SpaCy dataset file (.spacy).
        splits: Mapping of set name to fraction, e.g.
            ``{"train": 0.8, "dev": 0.1, "test": 0.1}``. The fractions must
            be positive and sum to 1.
        output_folder: Folder where the sets will be written.
        seed: Random seed for a reproducible shuffle.
        counts: Number of documents per set, filled by :meth:`run`.

    Raises:
        ValueError: On creation, if no sets are given, a fraction is not
            positive or the fractions do not sum to 1.
    """

    dataset_path: Path
    splits: dict[str, float]
    output_folder: Path
    seed: int = 42
    counts: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if not self.splits:
            raise ValueError("At least one split is required.")
        if any(fraction <= 0 for fraction in self.splits.values()):
            raise ValueError("Split fractions must be greater than 0.")
        total = sum(self.splits.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Split fractions must sum to 1, got {total:g}.")

    def run(self):
        """Split the dataset and write the sets.

        Returns:
            dict: Path of the written file per set name.
        """
        vocab = spacy.blank("en").vocab
        docs = list(DocBin().from_disk(self.dataset_path).get_docs(vocab))
        random.Random(self.seed).shuffle(docs)

        output_folder = Path(self.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        paths = {}
        start = 0
        cumulative = 0.0
        names = list(self.splits)
        for name in names:
            cumulative += self.splits[name]
            # Cumulative boundaries assign every document exactly once; the
            # last set always ends at the last document, whatever the rounding.
            end = len(docs) if name == names[-1] else round(cumulative * len(docs))
            part = docs[start:end]
            paths[name] = output_folder / f"{name}.spacy"
            DocBin(docs=part, store_user_data=True).to_disk(paths[name])
            self.counts[name] = len(part)
            start = end
        return paths
