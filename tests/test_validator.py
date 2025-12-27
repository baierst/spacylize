import json
from pathlib import Path
import pytest
import spacy
from spacy.tokens import DocBin, Doc

from spacylize.validator import DataValidator  # Replace with actual import path


def create_dummy_docbin(tmp_path: Path, n_docs=5):
    """Create a dummy .spacy dataset with NER annotations for testing."""
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    for i in range(n_docs):
        text = f"Document {i} with entity TEST{i}"
        doc = nlp.make_doc(text)

        # Find character indices of "TEST{i}"
        start = text.find(f"TEST{i}")
        end = start + len(f"TEST{i}")

        span = doc.char_span(start, end, label="TEST")
        if span is not None:
            doc.ents = (span,)
        else:
            doc.ents = ()  # fallback if alignment fails

        doc_bin.add(doc)

    file_path = tmp_path / "dummy.spacy"
    doc_bin.to_disk(file_path)
    return file_path


def test_validator_creates_reports(tmp_path: Path):
    dataset_path = create_dummy_docbin(tmp_path)
    output_folder = tmp_path / "reports"

    validator = DataValidator(dataset_path=dataset_path, output_folder=output_folder)
    validator.run()

    # Check files exist
    dataset_name = dataset_path.stem
    json_file = output_folder / f"{dataset_name}_report.json"
    png_file = output_folder / f"{dataset_name}_report.png"

    assert json_file.exists(), "JSON report was not created"
    assert png_file.exists(), "PNG report was not created"

    # Check JSON structure
    with json_file.open() as f:
        report = json.load(f)

    assert "dataset" in report
    assert "documents" in report
    assert "entities" in report
    assert report["dataset"]["num_documents"] == 5
    assert report["entities"]["by_label"]["TEST"] == 5


def test_validator_handles_empty_dataset(tmp_path: Path):
    # Create an empty DocBin
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    empty_file = tmp_path / "empty.spacy"
    doc_bin.to_disk(empty_file)

    output_folder = tmp_path / "reports"
    validator = DataValidator(dataset_path=empty_file, output_folder=output_folder)
    validator.run()

    # JSON and PNG should still exist
    dataset_name = empty_file.stem
    json_file = output_folder / f"{dataset_name}_report.json"
    png_file = output_folder / f"{dataset_name}_report.png"

    assert json_file.exists()
    assert png_file.exists()

    with json_file.open() as f:
        report = json.load(f)

    assert report["dataset"]["num_documents"] == 0
    assert report["dataset"]["num_entities"] == 0
    assert report["entities"]["by_label"] == {}
