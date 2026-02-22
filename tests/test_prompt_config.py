"""Tests for structured prompt configuration loading and validation."""

import pytest
import tempfile
from pathlib import Path

from spacylize.prompt_config import (
    load_prompt_config,
    NERStructuredConfig,
    TextCatStructuredConfig,
    NERExample,
    TextCatCategory,
    TextCatExample,
)
from pydantic import ValidationError


def test_ner_structured_config_valid():
    """Test valid NER structured config."""
    config = NERStructuredConfig(
        task="ner",
        entities=["PERSON", "ORG"],
        domain="news articles",
        tone="professional",
        length="2-3 sentences",
        language="en",
        temperature=0.7,
        constraints=["Be realistic"],
        examples=[NERExample(text="[John](PERSON) works at [OpenAI](ORG).")],
    )

    assert config.task == "ner"
    assert config.entities == ["PERSON", "ORG"]
    assert config.domain == "news articles"
    assert config.temperature == 0.7


def test_ner_structured_config_no_entities():
    """Test that NER config requires at least one entity."""
    with pytest.raises(ValidationError, match="List should have at least 1 item"):
        NERStructuredConfig(
            task="ner",
            entities=[],
            domain="test",
        )


def test_textcat_structured_config_valid():
    """Test valid TextCat structured config."""
    config = TextCatStructuredConfig(
        task="textcat",
        categories=[
            TextCatCategory(name="Electronics", description="tech devices"),
            TextCatCategory(name="Clothing", description="apparel"),
        ],
        domain="product descriptions",
        tone="marketing",
        length="2-3 sentences",
        language="en",
        temperature=0.8,
        constraints=["Use brand names"],
        examples=[
            TextCatExample(text="A wireless keyboard.", category="Electronics")
        ],
    )

    assert config.task == "textcat"
    assert len(config.categories) == 2
    assert config.categories[0].name == "Electronics"
    assert config.temperature == 0.8


def test_textcat_structured_config_insufficient_categories():
    """Test that TextCat config requires at least two categories."""
    with pytest.raises(ValidationError, match="List should have at least 2 items"):
        TextCatStructuredConfig(
            task="textcat",
            categories=[TextCatCategory(name="Cat1", description="desc1")],
            domain="test",
        )


def test_load_prompt_config_ner_from_yaml(tmp_path):
    """Test loading NER structured config from YAML file."""
    config_file = tmp_path / "prompt.yaml"
    config_file.write_text(
        """
task: ner
entities:
  - PERSON
  - ORG
domain: "news articles"
tone: "professional"
length: "2-3 sentences"
language: "en"
temperature: 0.7
constraints:
  - "Be realistic"
examples:
  - text: "[John](PERSON) works here."
    explanation: "Simple example"
"""
    )

    prompt_config = load_prompt_config(config_file)

    # Check that prompts were rendered
    assert prompt_config.system.role == "system"
    assert prompt_config.user.role == "user"
    assert "Named Entity Recognition" in prompt_config.system.content
    assert "PERSON" in prompt_config.user.content
    assert "ORG" in prompt_config.user.content


def test_load_prompt_config_textcat_from_yaml(tmp_path):
    """Test loading TextCat structured config from YAML file."""
    config_file = tmp_path / "prompt.yaml"
    config_file.write_text(
        """
task: textcat
categories:
  - name: Electronics
    description: "tech devices"
  - name: Clothing
    description: "apparel"
domain: "product descriptions"
tone: "marketing"
length: "2-3 sentences"
language: "en"
temperature: 0.8
"""
    )

    prompt_config = load_prompt_config(config_file)

    # Check that prompts were rendered
    assert prompt_config.system.role == "system"
    assert prompt_config.user.role == "user"
    assert "Text Classification" in prompt_config.system.content
    assert "Electronics" in prompt_config.user.content
    assert "Clothing" in prompt_config.user.content


def test_load_prompt_config_missing_task(tmp_path):
    """Test that loading config without task field raises error."""
    config_file = tmp_path / "prompt.yaml"
    config_file.write_text(
        """
entities:
  - PERSON
domain: "test"
"""
    )

    with pytest.raises(RuntimeError, match="Missing 'task' field in config"):
        load_prompt_config(config_file)


def test_load_prompt_config_invalid_task(tmp_path):
    """Test that loading config with invalid task raises error."""
    config_file = tmp_path / "prompt.yaml"
    config_file.write_text(
        """
task: invalid
entities:
  - PERSON
domain: "test"
"""
    )

    with pytest.raises(RuntimeError, match="Unsupported task: invalid"):
        load_prompt_config(config_file)


def test_load_prompt_config_saves_rendered_prompts(tmp_path):
    """Test that rendered prompts are saved to output folder."""
    config_file = tmp_path / "prompt.yaml"
    config_file.write_text(
        """
task: ner
entities:
  - PERSON
domain: "test"
tone: "casual"
length: "1 sentence"
"""
    )

    output_folder = tmp_path / "output"
    prompt_config = load_prompt_config(config_file, output_folder=output_folder)

    # Check that rendered prompt files were created
    system_file = output_folder / "system_prompt.txt"
    user_file = output_folder / "user_prompt.txt"

    assert system_file.exists()
    assert user_file.exists()

    # Check content matches the PromptConfig
    assert system_file.read_text() == prompt_config.system.content
    assert user_file.read_text() == prompt_config.user.content


def test_load_prompt_config_invalid_yaml_format(tmp_path):
    """Test that invalid YAML raises appropriate error."""
    config_file = tmp_path / "prompt.yaml"
    config_file.write_text(
        """
task: ner
# Missing required fields like entities and domain
"""
    )

    with pytest.raises(RuntimeError, match="Invalid structured config"):
        load_prompt_config(config_file)
