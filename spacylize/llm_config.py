"""LLM configuration loading module.

This module provides functionality to load and validate LLM configuration
from YAML files, including environment variable expansion.
"""

import os
import re
from pathlib import Path
from typing import Optional, Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class LLMConfig(BaseModel):
    """Configuration model for LLM settings.

    Defines the structure and validation rules for LLM configuration
    including model selection, authentication, and generation parameters.

    Attributes:
        model: The model identifier for LiteLLM.
        api_key: Optional API key for authentication.
        api_base: Optional custom API base URL.
        max_tokens: Maximum number of tokens to generate.
    """

    model: str
    api_key: Optional[str] = Field(default=None)
    api_base: Optional[str] = None
    max_tokens: int = 1024

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


def load_llm_config(path: Path) -> LLMConfig:
    """Load and validate LLM configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        LLMConfig: Validated LLM configuration object.

    Raises:
        RuntimeError: If the configuration is invalid or fails validation.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    expanded = _expand_env_vars(raw)

    try:
        return LLMConfig.model_validate(expanded)
    except ValidationError as e:
        raise RuntimeError(f"Invalid LLM configuration:\n{e}") from e
