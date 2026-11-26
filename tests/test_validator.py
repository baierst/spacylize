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

    # Empty dataset should raise ValueError since task cannot be auto-detected
    with pytest.raises(ValueError, match="Dataset is empty, cannot detect task type"):
        validator = DataValidator(dataset_path=empty_file, output_folder=output_folder)


def create_textcat_docbin(tmp_path: Path, texts):
    """Create a .spacy dataset with one positive category per document."""
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    for i, text in enumerate(texts):
        doc = nlp.make_doc(text)
        doc.cats = {"A": float(i % 2 == 0), "B": float(i % 2 == 1)}
        doc_bin.add(doc)

    file_path = tmp_path / "textcat.spacy"
    doc_bin.to_disk(file_path)
    return file_path


def test_validator_writes_dashboard_and_quality(tmp_path: Path):
    dataset_path = create_dummy_docbin(tmp_path, n_docs=12)
    output_folder = tmp_path / "reports"

    DataValidator(dataset_path=dataset_path, output_folder=output_folder).run()

    assert (output_folder / "dummy_report.html").exists()
    report = json.loads((output_folder / "dummy_report.json").read_text())
    assert {"duplicates", "diversity", "labels"} <= report["quality"].keys()


def test_validator_textcat_dashboard_escapes_html(tmp_path: Path):
    texts = [f"Product description number {i}." for i in range(10)]
    texts.append("<script>alert('x')</script> injected text")
    dataset_path = create_textcat_docbin(tmp_path, texts)
    output_folder = tmp_path / "reports"

    DataValidator(dataset_path=dataset_path, output_folder=output_folder).run()

    html = (output_folder / "textcat_report.html").read_text()
    assert "<script>alert" not in html
    report = json.loads((output_folder / "textcat_report.json").read_text())
    assert report["quality"]["labels"] == {"no_category": 0, "multiple_categories": 0}
