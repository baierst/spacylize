"""Tests for template rendering system."""

import pytest
from spacylize.templates import TemplateRegistry, NERTemplate, TextCatTemplate


def test_template_registry_get_ner():
    """Test that registry returns NER template class."""
    template_cls = TemplateRegistry.get_template("ner")
    assert template_cls == NERTemplate


def test_template_registry_get_textcat():
    """Test that registry returns TextCat template class."""
    template_cls = TemplateRegistry.get_template("textcat")
    assert template_cls == TextCatTemplate


def test_template_registry_invalid_task():
    """Test that registry raises error for invalid task."""
    with pytest.raises(ValueError, match="No template for task 'invalid'"):
        TemplateRegistry.get_template("invalid")


def test_ner_template_render_basic():
    """Test NER template rendering with basic config."""
    config = {
        "task": "ner",
        "entities": ["PERSON", "ORG"],
        "domain": "news articles",
        "tone": "professional",
        "length": "2-3 sentences",
        "language": "en",
        "temperature": 0.7,
        "constraints": [],
        "examples": [],
    }

    system_prompt, user_prompt = NERTemplate.render(config)

    # Check system prompt contains key information
    assert "Named Entity Recognition" in system_prompt
    assert "news articles" in system_prompt
    assert "professional" in system_prompt
    assert "[TEXT](LABEL)" in system_prompt

    # Check user prompt contains entities
    assert "PERSON" in user_prompt
    assert "ORG" in user_prompt
    assert "news articles" in user_prompt


def test_ner_template_render_with_examples():
    """Test NER template rendering with examples."""
    config = {
        "task": "ner",
        "entities": ["PERSON"],
        "domain": "test domain",
        "tone": "casual",
        "length": "1 sentence",
        "language": "en",
        "temperature": 0.5,
        "constraints": ["Be realistic"],
        "examples": [
            {"text": "[John](PERSON) is here.", "explanation": "Simple example"}
        ],
    }

    system_prompt, user_prompt = NERTemplate.render(config)

    # Check examples are included
    assert "[John](PERSON) is here." in user_prompt
    assert "Simple example" in user_prompt

    # Check constraints are included
    assert "Be realistic" in user_prompt


def test_ner_template_render_non_english():
    """Test NER template with non-English language."""
    config = {
        "task": "ner",
        "entities": ["PERSON"],
        "domain": "news",
        "tone": "formal",
        "length": "1-2 sentences",
        "language": "de",
        "temperature": 0.7,
        "constraints": [],
        "examples": [],
    }

    system_prompt, user_prompt = NERTemplate.render(config)

    # Check language is mentioned
    assert "de" in system_prompt or "de" in user_prompt


def test_textcat_template_render_basic():
    """Test TextCat template rendering with basic config."""
    config = {
        "task": "textcat",
        "categories": [
            {"name": "Electronics", "description": "tech devices"},
            {"name": "Clothing", "description": "apparel"},
        ],
        "domain": "product descriptions",
        "tone": "marketing",
        "length": "2-3 sentences",
        "language": "en",
        "temperature": 0.8,
        "constraints": [],
        "examples": [],
    }

    system_prompt, user_prompt = TextCatTemplate.render(config)

    # Check system prompt
    assert "Text Classification" in system_prompt
    assert "product descriptions" in system_prompt
    assert "LABEL:" in system_prompt

    # Check user prompt contains categories
    assert "Electronics" in user_prompt
    assert "tech devices" in user_prompt
    assert "Clothing" in user_prompt
    assert "apparel" in user_prompt


def test_textcat_template_render_with_examples():
    """Test TextCat template rendering with examples."""
    config = {
        "task": "textcat",
        "categories": [
            {"name": "Tech", "description": "technology"},
            {"name": "Fashion", "description": "clothing"},
        ],
        "domain": "products",
        "tone": "professional",
        "length": "1 sentence",
        "language": "en",
        "temperature": 0.7,
        "constraints": ["Be specific"],
        "examples": [
            {"text": "A wireless keyboard.", "category": "Tech"}
        ],
    }

    system_prompt, user_prompt = TextCatTemplate.render(config)

    # Check examples are included
    assert "A wireless keyboard." in user_prompt
    assert "Tech" in user_prompt

    # Check constraints are included
    assert "Be specific" in user_prompt


def test_textcat_template_render_non_english():
    """Test TextCat template with non-English language."""
    config = {
        "task": "textcat",
        "categories": [
            {"name": "Cat1", "description": "desc1"},
            {"name": "Cat2", "description": "desc2"},
        ],
        "domain": "test",
        "tone": "casual",
        "length": "1 sentence",
        "language": "es",
        "temperature": 0.7,
        "constraints": [],
        "examples": [],
    }

    system_prompt, user_prompt = TextCatTemplate.render(config)

    # Check language is mentioned
    assert "es" in system_prompt or "es" in user_prompt
