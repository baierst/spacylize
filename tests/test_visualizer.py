import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import spacy
from spacy.tokens import DocBin, Doc

from spacylize.visualizer import DataVisualizer


def test_init_raises_if_input_missing(tmp_path):
    missing_file = tmp_path / "missing.spacy"

    with pytest.raises(FileNotFoundError, match="Input file not found"):
        DataVisualizer(
            input_path=missing_file,
            task="ner",
        )


def test_init_raises_for_unsupported_task(tmp_path):
    file_path = tmp_path / "data.spacy"
    file_path.write_bytes(b"dummy")

    with pytest.raises(ValueError, match="only 'ner' task is supported"):
        DataVisualizer(
            input_path=file_path,
            task="textcat",
        )


def create_docbin(path: Path, n_docs: int = 3):
    nlp = spacy.blank("en")
    docs = []

    for i in range(n_docs):
        doc = nlp.make_doc(f"Entity {i}")
        doc.ents = []  # no entities needed for visualization
        docs.append(doc)

    doc_bin = DocBin(docs=docs)
    doc_bin.to_disk(path)

def test_load_docbin_limits_samples(tmp_path):
    data_path = tmp_path / "data.spacy"
    create_docbin(data_path, n_docs=10)

    visualizer = DataVisualizer(
        input_path=data_path,
        task="ner",
        n_samples=4,
    )

    docs = visualizer._load_docbin()

    assert len(docs) == 4
    assert all(isinstance(doc, Doc) for doc in docs)

def test_run_calls_displacy_serve(tmp_path):
    data_path = tmp_path / "data.spacy"
    create_docbin(data_path, n_docs=2)

    visualizer = DataVisualizer(
        input_path=data_path,
        task="ner",
        port=5010,
    )

    with patch("spacylize.visualizer.displacy.serve") as mock_serve:
        visualizer.run()

        mock_serve.assert_called_once()
        args, kwargs = mock_serve.call_args

        assert kwargs["style"] == "ent"
        assert kwargs["port"] == 5010
        assert isinstance(args[0], list)
