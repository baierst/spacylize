import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import spacy
from spacy.tokens import DocBin

from spacylize.generator import DataGenerator


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
    mock_to_disk.assert_called_once_with(output_path)
