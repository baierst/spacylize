import pytest
from unittest.mock import patch

from spacylize.llm import LLMClient


@pytest.fixture
def mock_completion_response():
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a mocked response."
                }
            }
        ]
    }


@patch("spacylize.llm.completion")
def test_generate_without_system_prompt(
    mock_completion,
    mock_completion_response
):
    mock_completion.return_value = mock_completion_response

    client = LLMClient(
        model="gpt-4o-mini",
        api_key="test-key",
        max_tokens=100,
    )

    result = client.generate("Hello")

    # Assertions
    assert result == "This is a mocked response."

    mock_completion.assert_called_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        api_key="test-key",
        api_base=None,
        max_tokens=100,
    )


@patch("spacylize.llm.completion")
def test_generate_with_system_prompt(
    mock_completion,
    mock_completion_response
):
    mock_completion.return_value = mock_completion_response

    client = LLMClient(
        model="anthropic/claude-opus-4-5-20251101",
        api_key="test-key",
        max_tokens=200,
    )

    result = client.generate(
        prompt="Tell me a joke.",
        system_prompt="You are a funny assistant."
    )

    assert result == "This is a mocked response."

    mock_completion.assert_called_once_with(
        model="anthropic/claude-opus-4-5-20251101",
        messages=[
            {"role": "system", "content": "You are a funny assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
        api_key="test-key",
        api_base=None,
        max_tokens=200,
    )


@patch("spacylize.llm.completion")
def test_generate_with_api_base(
    mock_completion,
    mock_completion_response
):
    mock_completion.return_value = mock_completion_response

    client = LLMClient(
        model="ollama/guanaco-7b",
        api_base="http://localhost:11434",
        max_tokens=50,
    )

    result = client.generate("Test local model")

    assert result == "This is a mocked response."

    mock_completion.assert_called_once()
    _, kwargs = mock_completion.call_args

    assert kwargs["api_base"] == "http://localhost:11434"
