"""Prompt configuration loading module.

This module provides functionality to load and validate prompt configurations
from YAML files for LLM data generation tasks.
"""

from pathlib import Path
from typing import Literal, Any
import os
import re
import yaml
from pydantic import BaseModel, ValidationError


class PromptMessage(BaseModel):
    """A single prompt message with role and content.

    Attributes:
        role: The role of the message (system, user, or assistant).
        content: The text content of the message.
    """

    role: Literal["system", "user", "assistant"]
    content: str

    model_config = {"extra": "forbid"}


class PromptConfig(BaseModel):
    """Configuration model for prompt templates.

    Defines the structure for prompt configurations including system
    and user messages for LLM interactions.

    Attributes:
        system: System prompt message for setting LLM behavior.
        user: User prompt message for the main task instruction.
    """

    system: PromptMessage
    user: PromptMessage

    model_config = {"extra": "forbid"}


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variable references in configuration values.

    Replaces ${VAR_NAME} patterns with their environment variable values.
    Supports nested dictionaries and lists.

    Args:
        value: Configuration value to process (str, dict, list, or other).

    Returns:
        The value with environment variables expanded.
    """
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.fullmatch(value)
        if match:
            return os.getenv(match.group(1))
        return value

    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]

    return value


def load_prompt_config(path: Path) -> PromptConfig:
    """Load and validate prompt configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        PromptConfig: Validated prompt configuration object.

    Raises:
        RuntimeError: If the configuration is invalid or fails validation.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    expanded = _expand_env_vars(raw)

    try:
        return PromptConfig.model_validate(expanded)
    except ValidationError as e:
        raise RuntimeError(f"Invalid prompt configuration:\n{e}") from e
