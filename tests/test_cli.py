import pytest
from typer.testing import CliRunner
from pathlib import Path
import yaml
import spacy
from spacy.tokens import DocBin

from spacylize.cli import app  # Change this if your main app file is named differently

runner = CliRunner()


def create_temp_yaml(data: dict, tmp_path: Path) -> Path:
    """Creates a temporary YAML file for testing."""
    temp_file = tmp_path / "temp.yaml"
    with open(temp_file, "w") as f:
        yaml.dump(data, f)
    return temp_file


def create_temp_spacy_file(docs: list, tmp_path: Path, filename="temp.spacy") -> Path:
    """Creates a temporary SpaCy file for testing.  Now takes a list of docs."""
    temp_file = tmp_path / filename
    db = DocBin(docs=docs,  # Use the provided list of docs
                 store_user_data=True)
    db.to_disk(temp_file)
    return temp_file


def create_blank_spacy_model() -> spacy.Language:
    """Creates a blank SpaCy model for testing."""
    nlp = spacy.blank("en")
    return nlp


@pytest.fixture(scope="module")
def blank_nlp():
    """Fixture for a blank SpaCy model."""
    return create_blank_spacy_model()


@pytest.fixture
def temp_yaml_file(tmp_path: Path):
    """Fixture for a temporary YAML file with example data."""
    yaml_data = {
        "prompt": "Generate a product description for a {product_type}...",
        "examples": [
            {"product_type": "smartphone"},
            {"product_type": "shoes"},
        ],
        "entity_types": ["PRODUCT_TYPE"],
    }
    return create_temp_yaml(yaml_data, tmp_path)


@pytest.fixture
def temp_spacy_file(tmp_path: Path, blank_nlp):
    """Fixture for a temporary SpaCy file with example data."""
    doc1 = blank_nlp("This is a test.")
    doc2 = blank_nlp("Another test sentence.")
    docs = [doc1, doc2]
    return create_temp_spacy_file(docs, tmp_path)


def test_generate_data_success(temp_yaml_file, tmp_path, blank_nlp):
    """Test successful execution of the generate command."""
    result = runner.invoke(
        app,
        [
            "generate",
            "--llm",
            "test-llm",
            "--prompt-config",
            str(temp_yaml_file),
            "--n-samples",
            "2",
            "--output-path",
            str(tmp_path / "output.spacy"),
            "--task",
            "ner",
            "--labels",
            "TEST",
        ],
    )
    assert result.exit_code == 0
    assert "Generating training data..." in result.output
    assert "Saved SpaCy NER data to" in result.output
    assert (tmp_path / "output.spacy").exists()  # Check if file was created


def test_generate_data_no_prompt_config(tmp_path):
    """Test generate command with missing prompt config."""
    result = runner.invoke(
        app,
        [
            "generate",
            "--llm",
            "test-llm",
            "--n-samples",
            "2",
            "--output-path",
            str(tmp_path / "output.spacy"),
            "--task",
            "ner",
            "--labels",
            "TEST",
        ],
    )
    assert result.exit_code == 2
    assert "Error: " in result.output


def test_generate_data_invalid_task(temp_yaml_file, tmp_path):
    """Test generate command with invalid task."""
    result = runner.invoke(
        app,
        [
            "generate",
            "--llm",
            "test-llm",
            "--prompt-config",
            str(temp_yaml_file),
            "--n-samples",
            "2",
            "--output-path",
            str(tmp_path / "output.spacy"),
            "--task",
            "invalid",
            "--labels",
            "TEST",
        ],
    )
    assert result.exit_code == 1
    assert "Error: Invalid task" in result.output


def test_generate_data_no_labels_for_ner(temp_yaml_file, tmp_path):
    """Test generate command with no labels for NER task."""
    result = runner.invoke(
        app,
        [
            "generate",
            "--llm",
            "test-llm",
            "--prompt-config",
            str(temp_yaml_file),
            "--n-samples",
            "2",
            "--output-path",
            str(tmp_path / "output.spacy"),
            "--task",
            "ner",
        ],
    )
    assert result.exit_code == 1
    assert "Error: Labels are required for NER task" in result.output


def test_visualize_data_success(temp_spacy_file):
    """Test successful execution of the visualize command."""
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input-path",
            str(temp_spacy_file),
            "--task",
            "ner",
            "--n-samples",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "Visualizing generated data..." in result.output
    assert "Serving visualization on port" in result.output


def test_visualize_data_file_not_found(tmp_path):
    """Test visualize command with non-existent input file."""
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input-path",
            str(tmp_path / "nonexistent.spacy"),
            "--task",
            "ner",
            "--n-samples",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert "Error: Input data file not found" in result.output



def test_visualize_data_invalid_task(temp_spacy_file, tmp_path):
    """Test visualize command with invalid task."""
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input-path",
            str(temp_spacy_file),
            "--task",
            "invalid",
            "--n-samples",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "Error: Invalid task" in result.output


def test_visualize_data_invalid_n_samples(temp_spacy_file, tmp_path):
    """Test visualize command with invalid n_samples."""
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input-path",
            str(temp_spacy_file),
            "--task",
            "ner",
            "--n-samples",
            "0",
        ],
    )
    assert result.exit_code == 3
    assert "Error: Number of samples must be greater than zero." in result.output


def test_train_pipeline_success(temp_spacy_file, tmp_path, blank_nlp):
    """Test successful execution of the train command."""
    result = runner.invoke(
        app,
        [
            "train",
            "--train-data",
            str(temp_spacy_file),
            "--base-model",
            "en_core_web_sm",  # Or a small model that you have installed
            "--output-model",
            str(tmp_path / "trained_model"),
            "--n-iter",
            "1",
            "--dropout",
            "0.1",
        ],
    )
    assert result.exit_code == 0
    assert "Training SpaCy pipeline..." in result.output
    assert "Trained pipeline saved to" in result.output
    assert (tmp_path / "trained_model").exists()  # Check if model dir was created


def test_train_pipeline_data_not_found(tmp_path):
    """Test train command with non-existent training data."""
    result = runner.invoke(
        app,
        [
            "train",
            "--train-data",
            str(tmp_path / "nonexistent.spacy"),
            "--base-model",
            "en_core_web_sm",
            "--output-model",
            str(tmp_path / "trained_model"),
            "--n-iter",
            "1",
            "--dropout",
            "0.1",
        ],
    )
    assert result.exit_code == 1
    assert "Error: Training data file not found" in result.output


def test_train_pipeline_invalid_n_iter(temp_spacy_file, tmp_path):
    """Test train command with invalid n_iter."""
    result = runner.invoke(
        app,
        [
            "train",
            "--train-data",
            str(temp_spacy_file),
            "--base-model",
            "en_core_web_sm",
            "--output-model",
            str(tmp_path / "trained_model"),
            "--n-iter",
            "0",
            "--dropout",
            "0.1",
        ],
    )
    assert result.exit_code == 2
    assert "Error: Number of iterations must be greater than zero." in result.output


def test_train_pipeline_invalid_dropout(temp_spacy_file, tmp_path):
    """Test train command with invalid dropout."""
    result = runner.invoke(
        app,
        [
            "train",
            "--train-data",
            str(temp_spacy_file),
            "--base-model",
            "en_core_web_sm",
            "--output-model",
            str(tmp_path / "trained_model"),
            "--n-iter",
            "1",
            "--dropout",
            "1.1",
        ],
    )
    assert result.exit_code == 3
    assert "Error: Dropout must be between 0 and 1." in result.output
