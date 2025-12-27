import os
import re
from pathlib import Path
from typing import Optional, Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class LLMConfig(BaseModel):
    model: str
    api_key: Optional[str] = Field(default=None)
    api_base: Optional[str] = None
    max_tokens: int = 1024

    model_config = {"extra": "forbid"}


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: Any) -> Any:
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
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    expanded = _expand_env_vars(raw)

    try:
        return LLMConfig.model_validate(expanded)
    except ValidationError as e:
        raise RuntimeError(f"Invalid LLM configuration:\n{e}") from e
