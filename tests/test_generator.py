from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import spacy
from spacy.tokens import DocBin

from spacylize.generator import DataGenerator


@pytest.fixture(autouse=True)
def mock_prompt_rendering():
    """Render a fixed prompt instead of reading a prompt config file."""
    prompt = MagicMock()
    prompt.user.content = "user prompt"
    prompt.system.content = "system prompt"
    with (
        patch(
            "spacylize.generator.load_structured_config",
            return_value=MagicMock(labels=["PERSON"]),
        ),
        patch("spacylize.generator.render_prompt", return_value=prompt),
    ):
        yield


def test_parse_annotated_text_single_entity():
    from spacylize.generator import NERParser

    text = "Hello [John Doe](PERSON), welcome!"
    clean_text, entities = NERParser.parse(text)

    assert clean_text == "Hello John Doe, welcome!"
    assert entities == [(6, 14, "PERSON")]


def test_parse_annotated_text_multiple_entities():
    from spacylize.generator import NERParser

    text = "[Alice](PERSON) works at [OpenAI](ORG)."
    clean_text, entities = NERParser.parse(text)

    assert clean_text == "Alice works at OpenAI."
    assert entities == [
        (0, 5, "PERSON"),
        (15, 21, "ORG"),
    ]


@patch("spacylize.generator.load_llm_config")
@patch("spacylize.generator.load_prompt_config")
@patch("spacylize.generator.LLMClient")
@patch.object(DocBin, "to_disk")
def test_run_generates_docs(
    mock_to_disk,
    mock_llm_client_cls,
    mock_load_prompt_config,
    mock_load_llm_config,
    tmp_path,
):
    # --- Mock LLM config ---
    mock_load_llm_config.return_value = MagicMock(
        model="test-model",
        api_key="test-key",
        api_base="http://test",
        max_tokens=100,
    )

    # --- Mock prompt config ---
    mock_prompt = MagicMock()
    mock_prompt.user.content = "user prompt"
    mock_prompt.system.content = "system prompt"
    mock_prompt.temperature = 0.9
    mock_load_prompt_config.return_value = mock_prompt

    # --- Mock LLM client ---
    mock_llm_client = MagicMock()
    mock_llm_client.generate.return_value = "Hello [John](PERSON)."
    mock_llm_client_cls.return_value = mock_llm_client

    output_path = tmp_path / "docs.spacy"

    generator = DataGenerator(
        llm_config_path=Path("llm.yaml"),
        prompt_config_path=Path("prompt.yaml"),
        n_samples=2,
        output_path=output_path,
        task="ner",
    )

    generator.run()

    assert mock_llm_client.generate.call_count == 2
    mock_llm_client.generate.assert_called_with(
        "user prompt", "system prompt", temperature=0.9
    )
    mock_to_disk.assert_called_once_with(output_path)


@patch("spacylize.generator.load_llm_config")
@patch("spacylize.generator.load_prompt_config")
@patch("spacylize.generator.LLMClient")
def test_data_generator_saves_rendered_prompts(
    mock_llm_client_cls,
    mock_load_prompt_config,
    mock_load_llm_config,
    tmp_path,
):
    """Test that DataGenerator saves rendered prompts to output folder."""
    # --- Mock LLM config ---
    mock_load_llm_config.return_value = MagicMock(
        model="test-model",
        api_key="test-key",
        api_base="http://test",
        max_tokens=100,
    )

    # --- Mock prompt config ---
    mock_prompt = MagicMock()
    mock_prompt.user.content = "user prompt"
    mock_prompt.system.content = "system prompt"
    mock_load_prompt_config.return_value = mock_prompt

    # --- Mock LLM client ---
    mock_llm_client = MagicMock()
    mock_llm_client_cls.return_value = mock_llm_client

    output_path = tmp_path / "docs.spacy"

    DataGenerator(
        llm_config_path=Path("llm.yaml"),
        prompt_config_path=Path("prompt.yaml"),
        n_samples=1,
        output_path=output_path,
        task="ner",
    )

    # Verify load_prompt_config was called with output_folder parameter
    mock_load_prompt_config.assert_called_once()
    call_args = mock_load_prompt_config.call_args

    # Check that output_folder was passed as keyword argument
    assert "output_folder" in call_args.kwargs
    assert call_args.kwargs["output_folder"] == tmp_path


@patch("spacylize.generator.load_llm_config")
@patch("spacylize.generator.load_prompt_config")
@patch("spacylize.generator.LLMClient")
def test_run_skips_failed_samples_and_saves_on_abort(
    mock_llm_client_cls, mock_load_prompt_config, mock_load_llm_config, tmp_path
):
    mock_llm_client = MagicMock()
    mock_llm_client.generate.side_effect = [
        "Hello [John](PERSON).",
        RuntimeError("connection refused"),
        "Hi [Anna](PERSON).",
        RuntimeError("connection refused"),
        RuntimeError("connection refused"),
        RuntimeError("connection refused"),
    ]
    mock_llm_client_cls.return_value = mock_llm_client
    output_path = tmp_path / "docs.spacy"

    generator = DataGenerator(
        llm_config_path=Path("llm.yaml"),
        prompt_config_path=Path("prompt.yaml"),
        n_samples=10,
        output_path=output_path,
        task="ner",
    )

    # One failure is skipped, three in a row abort the run
    with pytest.raises(RuntimeError):
        generator.run()

    docs = list(DocBin().from_disk(output_path).get_docs(spacy.blank("en").vocab))
    assert [doc.text for doc in docs] == ["Hello John.", "Hi Anna."]


@patch(
    "spacylize.generator.topic_score",
    MagicMock(side_effect=[0.9, ConnectionError("nimble not available")]),
)
@patch("spacylize.generator.load_llm_config")
@patch("spacylize.generator.load_prompt_config")
@patch("spacylize.generator.LLMClient")
def test_run_scores_topic_and_skips_when_unavailable(
    mock_llm_client_cls, mock_load_prompt_config, mock_load_llm_config, tmp_path
):
    mock_llm_client = MagicMock()
    mock_llm_client.generate.side_effect = [
        "Hello [John](PERSON).",
        "Hi [Anna](PERSON).",
    ]
    mock_llm_client_cls.return_value = mock_llm_client
    output_path = tmp_path / "docs.spacy"

    DataGenerator(
        llm_config_path=Path("llm.yaml"),
        prompt_config_path=Path("prompt.yaml"),
        n_samples=2,
        output_path=output_path,
        task="ner",
        topic_check=True,
    ).run()

    # The first sample is scored, then the topic check is skipped
    docs = list(DocBin().from_disk(output_path).get_docs(spacy.blank("en").vocab))
    assert [doc.user_data.get("topic_score") for doc in docs] == [0.9, None]


@patch("spacylize.generator.load_llm_config")
@patch("spacylize.generator.load_prompt_config")
@patch("spacylize.generator.LLMClient")
def test_run_drops_unconfigured_labels(
    mock_llm_client_cls, mock_load_prompt_config, mock_load_llm_config, tmp_path
):
    mock_llm_client = MagicMock()
    mock_llm_client.generate.return_value = "[John](PERSON) uses [iOS](OS)."
    mock_llm_client_cls.return_value = mock_llm_client
    output_path = tmp_path / "docs.spacy"

    DataGenerator(
        llm_config_path=Path("llm.yaml"),
        prompt_config_path=Path("prompt.yaml"),
        n_samples=1,
        output_path=output_path,
        task="ner",
    ).run()

    doc = next(DocBin().from_disk(output_path).get_docs(spacy.blank("en").vocab))
    assert [(e.text, e.label_) for e in doc.ents] == [("John", "PERSON")]


@patch("spacylize.generator.load_llm_config")
@patch("spacylize.generator.load_prompt_config")
@patch("spacylize.generator.LLMClient")
def test_run_appends_to_existing_file(
    mock_llm_client_cls, mock_load_prompt_config, mock_load_llm_config, tmp_path
):
    nlp = spacy.blank("en")
    output_path = tmp_path / "docs.spacy"
    DocBin(docs=[nlp.make_doc("Old sample.")]).to_disk(output_path)
    mock_llm_client = MagicMock()
    mock_llm_client.generate.return_value = "Hello [John](PERSON)."
    mock_llm_client_cls.return_value = mock_llm_client

    DataGenerator(
        llm_config_path=Path("llm.yaml"),
        prompt_config_path=Path("prompt.yaml"),
        n_samples=1,
        output_path=output_path,
        task="ner",
        append=True,
    ).run()

    docs = list(DocBin().from_disk(output_path).get_docs(nlp.vocab))
    assert [doc.text for doc in docs] == ["Old sample.", "Hello John."]
