from pathlib import Path

import pytest
import spacy
from spacy.tokens import DocBin

from spacylize.splitter import DataSplitter


def create_docbin(tmp_path: Path, n_docs=10):
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    for i in range(n_docs):
        doc_bin.add(nlp.make_doc(f"Document {i}"))
    path = tmp_path / "data.spacy"
    doc_bin.to_disk(path)
    return path


def read_texts(path: Path):
    return [
        doc.text for doc in DocBin().from_disk(path).get_docs(spacy.blank("en").vocab)
    ]


def test_three_way_split(tmp_path: Path):
    dataset = create_docbin(tmp_path)
    out = tmp_path / "splits"

    splitter = DataSplitter(dataset, {"train": 0.8, "dev": 0.1, "test": 0.1}, out)
    paths = splitter.run()

    assert paths == {name: out / f"{name}.spacy" for name in ("train", "dev", "test")}
    assert splitter.counts == {"train": 8, "dev": 1, "test": 1}
    texts = [t for path in paths.values() for t in read_texts(path)]
    assert sorted(texts) == sorted(read_texts(dataset))


def test_two_way_split_is_reproducible(tmp_path: Path):
    dataset = create_docbin(tmp_path)
    splits = {"train": 0.7, "dev": 0.3}

    a = DataSplitter(dataset, splits, tmp_path / "a", seed=1).run()
    b = DataSplitter(dataset, splits, tmp_path / "b", seed=1).run()

    assert read_texts(a["train"]) == read_texts(b["train"])
    assert len(read_texts(a["dev"])) == 3


def test_all_documents_assigned_with_rounding(tmp_path: Path):
    dataset = create_docbin(tmp_path, n_docs=7)

    splitter = DataSplitter(
        dataset, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, tmp_path / "s"
    )
    splitter.run()

    assert sum(splitter.counts.values()) == 7


@pytest.mark.parametrize(
    "splits", [{}, {"train": 0.8, "dev": 0.1}, {"train": 1.2, "dev": -0.2}]
)
def test_invalid_splits_fail_on_creation(tmp_path: Path, splits):
    with pytest.raises(ValueError):
        DataSplitter(tmp_path / "data.spacy", splits, tmp_path / "s")
